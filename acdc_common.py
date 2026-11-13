"""
Shared evaluation protocol for the ACDC baselines (random forest, MLP).

Everything in this module reproduces, step by step and with the same constants and seeds, the protocol of
the Tsetlin Machine notebook (TorchTsetlin_model_optuna.ipynb), so that the TM, the RF and the MLP differ
only in the model itself:

    data      same CSV files, same column handling, all-NaN features dropped, median imputation (train only)
    selection Optuna, TPE(seed, 10 startup trials), MedianPruner(10, warm-up 1), 30 trials, objective =
              5-fold stratified CV accuracy (StratifiedKFold(random_state=seed)) averaged over 2 model seeds
              per fold; for iterative models the fold curve is smoothed over +/-2 epochs and the argmax
              gives the number of epochs
    nested CV outer StratifiedKFold(random_state=SEED+1); the whole selection is repeated in every outer
              fold (selection seed SEED+10+f); 3 model seeds per outer fold
    calib.    temperature scaling, tau fitted on out-of-fold scores of 3 x 5-fold CV (random_state SEED+100+r)
    test      retrain on all 100 training patients with 10 seeds (SEED*1000+1000+r), evaluate once on the 50
              test patients; mean +/- SD (ddof=1) over seeds; bootstrap 95% CI over patients (2000 resamples);
              ECE with 10 equal-width bins; Brier (multiclass, [0, 2]); NLL; per-class recall; mean confusion
              matrix; representative run = accuracy closest to the mean
    output    results.json with the same keys as the TM notebook, test_predictions.csv (per patient: error rate
              over seeds and mean calibrated probabilities), runs_metrics.csv, optuna_trials.csv, nested_cv.csv

A model plugs in through three callables:
    suggest(trial)                         -> params dict
    curve(Xtr, ytr, Xva, yva, params, seed) -> validation accuracy per epoch (length 1 if not iterative)
    fit_scores(Xtr, ytr, Xev, params, seed) -> (scores (n_ev, n_classes), fitted model); argmax(scores)
                                               is the prediction and softmax(scores / tau) the probability
"""

import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]

# ---- protocol constants: identical to the TM notebook -----------------------------------------------
N_TRIALS = 30          # Optuna trials per selection (main and each outer fold)
N_STARTUP = 10         # random trials before TPE; no pruning before this many trials
FOLDS = 5              # CV folds (inner and outer)
N_SEL_SEEDS = 2        # model seeds averaged per fold during selection
SMOOTH = 2             # epoch selection on a moving average of +/- SMOOTH epochs
N_OUTER_RUNS = 3       # model seeds per outer fold in nested CV
N_CAL_REPEATS = 3      # repeated CV for temperature calibration
N_RUNS = 10            # independent seeds of the final model
N_BOOT = 2000          # bootstrap resamples over test patients
ECE_BINS = 10          # equal-width confidence bins
SEED = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)


def sd(x):
    """Sample standard deviation (ddof = 1), the convention used everywhere."""
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1)) if x.size > 1 else 0.0


def versions(pkgs):
    v = {}
    for p in pkgs:
        try:
            v[p] = version(p)
        except Exception:
            pass
    v["python"] = platform.python_version()
    return v


# ---- data (same handling as the TM notebook, Section 2) -----------------------------------------------
def load_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv).rename(columns={"pid": "patient", "label": "group"})
    test_df = pd.read_csv(test_csv).rename(columns={"pid": "patient", "label": "group"})
    for name, _df in (("train", train_df), ("test", test_df)):
        _df["patient"] = _df["patient"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        unknown = sorted(set(_df["group"]) - set(CLASSES))
        if unknown:
            raise ValueError(f"{name} set has unexpected labels {unknown}; expected {CLASSES}")
    if set(train_df.patient) & set(test_df.patient):
        raise ValueError("Train and test sets share patient IDs")
    meta = ("patient", "group")
    feats = [c for c in train_df.columns if c not in meta and train_df[c].notna().any()]
    dropped = [c for c in train_df.columns if c not in meta and c not in feats]
    missing = [c for c in feats if c not in test_df.columns]
    if missing:
        raise ValueError(f"Test set is missing columns: {missing}")
    med = train_df[feats].median()
    n_missing = int(train_df[feats].isna().sum().sum() + test_df[feats].isna().sum().sum())
    X_train = train_df[feats].fillna(med).to_numpy(np.float64)
    X_test = test_df[feats].fillna(med).to_numpy(np.float64)
    y_train = train_df["group"].map(CLASSES.index).to_numpy(np.int64)
    y_test = test_df["group"].map(CLASSES.index).to_numpy(np.int64)
    print(f"Train: {len(y_train)} | Test: {len(y_test)} | {len(feats)} features "
          f"({len(dropped)} all-NaN features dropped, {n_missing} missing values imputed)")
    return dict(train_csv=str(train_csv), test_csv=str(test_csv), train_df=train_df, test_df=test_df, feats=feats, dropped=dropped, n_missing=n_missing,
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


# ---- metrics (verbatim from the TM notebook, Section 8) -------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece_score(probs, y, n_bins=ECE_BINS):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def brier(probs, y):
    return np.mean(np.sum((probs - np.eye(len(CLASSES))[y]) ** 2, axis=1))


def bootstrap_ci(preds_list, y, n_boot=N_BOOT, seed=SEED):
    """95% percentile CI of seed-averaged accuracy and macro-F1, resampling patients."""
    rng = np.random.default_rng(seed)
    P = np.asarray(preds_list)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        accs.append(np.mean([accuracy_score(y[idx], p[idx]) for p in P]))
        f1s.append(np.mean([f1_score(y[idx], p[idx], average="macro", labels=range(len(CLASSES)),
                                     zero_division=0) for p in P]))
    return {"accuracy": np.percentile(accs, [2.5, 97.5]).tolist(),
            "macro_f1": np.percentile(f1s, [2.5, 97.5]).tolist()}


def smooth(m, k=SMOOTH):
    pad = np.pad(m, k, mode="edge")
    return np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")


# ---- selection (TM notebook, Section 4) ------------------------------------------------------------------
def select_hparams(X, y, seed, suggest, curve, verbose=False):
    """Optuna search on (X, y). Returns (params incl. 'epochs', CV curve of the best trial, study)."""
    splits = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed).split(X, y))

    def objective(trial):
        p = suggest(trial)
        fold_curves = []
        for f, (tr, va) in enumerate(splits):
            cs = [curve(X[tr], y[tr], X[va], y[va], p, seed * 1000 + 100 * f + k) for k in range(N_SEL_SEEDS)]
            fold_curves.append(np.mean(cs, axis=0))
            trial.report(float(smooth(np.mean(fold_curves, axis=0)).max()), f)
            if f < FOLDS - 1 and trial.should_prune():
                raise optuna.TrialPruned()
        fc = np.array(fold_curves)
        m, sd_ = fc.mean(0), fc.std(0, ddof=1)
        sm = smooth(m)
        e = int(np.argmax(sm))
        trial.set_user_attr("epochs", e + 1)
        trial.set_user_attr("raw_mean", float(m[e]))
        trial.set_user_attr("raw_std", float(sd_[e]))
        trial.set_user_attr("curve", {"mean": m.tolist(), "std": sd_.tolist(), "smooth": sm.tolist()})
        return float(sm[e])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_STARTUP),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP, n_warmup_steps=1))
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    best = study.best_trial
    params = dict(best.params, epochs=best.user_attrs["epochs"])
    curve_best = {k: np.array(v) for k, v in best.user_attrs["curve"].items()}
    if verbose:
        n_pruned = sum(t_.state == optuna.trial.TrialState.PRUNED for t_ in study.trials)
        print(f"  {len(study.trials)} trials ({n_pruned} pruned); best trial #{best.number}: "
              f"smoothed CV acc {best.value:.3f}")
    return params, curve_best, study


# ---- full protocol -----------------------------------------------------------------------------------------
def run_protocol(name, D, suggest, curve, fit_scores, out, tau_bounds=(0.05, 20.0), extra_config=None,
                 pkgs=()):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = D["X_train"], D["X_test"], D["y_train"], D["y_test"]
    test_df, feats = D["test_df"], D["feats"]
    timings = {}

    # 1. selection on the full training set
    t0 = time.time()
    print(f"[{name}] Optuna selection on training set (1-100):")
    best_params, best_curve, study = select_hparams(X_train, y_train, SEED, suggest, curve, verbose=True)
    best_epoch = best_params["epochs"]
    sel_cv = float(best_curve["mean"][best_epoch - 1])
    sel_cv_std = float(best_curve["std"][best_epoch - 1])
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    trials_df = trials_df.drop(columns=[c for c in trials_df.columns if c.endswith("curve")])
    timings["selection_s"] = time.time() - t0
    print(f"[{name}] Selected {best_params} (selection CV acc {sel_cv:.3f} ± {sel_cv_std:.3f}, biased) "
          f"[{timings['selection_s']:.0f}s]")

    # 2. nested CV
    t0 = time.time()
    outer = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + 1)
    nested = []
    for f, (tr, va) in enumerate(outer.split(X_train, y_train)):
        p, _, _ = select_hparams(X_train[tr], y_train[tr], SEED + 10 + f, suggest, curve)
        accs = []
        for r in range(N_OUTER_RUNS):
            z, _ = fit_scores(X_train[tr], y_train[tr], X_train[va], p, SEED * 7 + 31 * f + r)
            accs.append(accuracy_score(y_train[va], z.argmax(1)))
        nested.append({"fold": f + 1, **p, "accuracy": float(np.mean(accs))})
        print(f"[{name}] outer fold {f + 1}: {p} -> held-out acc {np.mean(accs):.3f}")
    nested_df = pd.DataFrame(nested)
    NESTED_ACC, NESTED_STD = float(nested_df.accuracy.mean()), sd(nested_df.accuracy)
    timings["nested_s"] = time.time() - t0
    print(f"[{name}] Nested CV accuracy: {NESTED_ACC:.3f} ± {NESTED_STD:.3f} [{timings['nested_s']:.0f}s]")

    # 3. temperature on out-of-fold scores
    oof_z, oof_y = [], []
    for r in range(N_CAL_REPEATS):
        skf_r = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + 100 + r)
        for f, (tr, va) in enumerate(skf_r.split(X_train, y_train)):
            z, _ = fit_scores(X_train[tr], y_train[tr], X_train[va], best_params, SEED * 13 + 10 * r + f)
            oof_z.append(z)
            oof_y.append(y_train[va])
    oof_z, oof_y = np.vstack(oof_z), np.concatenate(oof_y)

    def nll_t(log_tau):
        p_ = softmax(oof_z / np.exp(log_tau))
        return -np.mean(np.log(p_[np.arange(len(oof_y)), oof_y] + 1e-12))

    res = minimize_scalar(nll_t, bounds=tuple(np.log(tau_bounds)), method="bounded")
    TAU = float(np.exp(res.x))
    OOF_ACC = float(np.mean(oof_z.argmax(1) == oof_y))
    print(f"[{name}] Fitted temperature tau = {TAU:.3f} (OOF NLL {res.fun:.3f}, OOF accuracy {OOF_ACC:.3f})")

    # 4. test
    runs, all_probs, all_preds, models = [], [], [], []
    for r in range(N_RUNS):
        z, model = fit_scores(X_train, y_train, X_test, best_params, SEED * 1000 + 1000 + r)
        y_pred = z.argmax(1)
        probs, raw = softmax(z / TAU), softmax(z)
        runs.append({"run": r + 1, "accuracy": accuracy_score(y_test, y_pred),
                     "macro_f1": f1_score(y_test, y_pred, average="macro"),
                     "ece": ece_score(probs, y_test), "brier": brier(probs, y_test),
                     "nll": log_loss(y_test, probs, labels=range(len(CLASSES))),
                     "ece_uncalibrated": ece_score(raw, y_test), "brier_uncalibrated": brier(raw, y_test),
                     "nll_uncalibrated": log_loss(y_test, np.clip(raw, 1e-12, 1), labels=range(len(CLASSES)))})
        all_probs.append(probs); all_preds.append(y_pred); models.append(model)
        print(f"[{name}] run {r + 1:>2}: acc {runs[-1]['accuracy']:.3f} | macro-F1 {runs[-1]['macro_f1']:.3f} | "
              f"ECE {runs[-1]['ece']:.3f}")
    runs_df = pd.DataFrame(runs)
    summary = runs_df.drop(columns="run").agg(["mean", sd]).T.rename(columns={"sd": "std"})
    CI = bootstrap_ci(all_preds, y_test)
    recalls = np.array([[np.mean(p[y_test == c] == c) for c in range(len(CLASSES))] for p in all_preds])
    cm_mean = np.mean([confusion_matrix(y_test, p, labels=range(len(CLASSES))) for p in all_preds], axis=0)
    err_rate = np.mean([p != y_test for p in all_preds], axis=0)
    wrong = [(int(pid), CLASSES[t], f"{e:.0%}") for pid, t, e in zip(test_df["patient"], y_test, err_rate) if e > 0]
    rep = int(np.argmin(np.abs(runs_df.accuracy - runs_df.accuracy.mean())))
    print(f"\n[{name}] TEST accuracy: {runs_df.accuracy.mean():.3f} ± {sd(runs_df.accuracy):.3f} across {N_RUNS} "
          f"seeds; 95% bootstrap CI [{CI['accuracy'][0]:.3f}, {CI['accuracy'][1]:.3f}]")
    print(summary.round(3))
    print("Mean confusion matrix:\n", cm_mean.round(1))
    print("Misclassified:", wrong)

    # 5. save (same layout and keys as the TM notebook)
    runs_df.to_csv(out / "runs_metrics.csv", index=False)
    trials_df.to_csv(out / "optuna_trials.csv", index=False)
    nested_df.to_csv(out / "nested_cv.csv", index=False)
    mean_probs = np.mean(all_probs, axis=0)
    pd.DataFrame({"patient": test_df["patient"], "true": [CLASSES[i] for i in y_test],
                  "error_rate_over_seeds": err_rate,
                  **{f"p_{c}": mean_probs[:, i] for i, c in enumerate(CLASSES)}}
                 ).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame({"patient": test_df["patient"].values, "true": [CLASSES[i] for i in y_test],
                  **{f"run{r + 1}": [CLASSES[i] for i in p] for r, p in enumerate(all_preds)}}
                 ).to_csv(out / "test_predictions_runs.csv", index=False)
    results = {
        "model": name, "versions": versions(pkgs),
        "config": {"tuning": "optuna", "n_trials": N_TRIALS, "n_startup_trials": N_STARTUP,
                   "sampler": "TPESampler(seed)", "pruner": "MedianPruner(n_warmup_steps=1)",
                   "objective": "smoothed 5-fold CV accuracy", "folds": FOLDS, "n_sel_seeds": N_SEL_SEEDS,
                   "smooth_halfwidth": SMOOTH, "n_outer_runs": N_OUTER_RUNS, "n_cal_repeats": N_CAL_REPEATS,
                   "n_runs": N_RUNS, "n_boot": N_BOOT, "ece_bins": ECE_BINS, "seed": SEED,
                   "std_convention": "ddof=1", **(extra_config or {})},
        "data": {"train_csv": Path(D["train_csv"]).name, "test_csv": Path(D["test_csv"]).name,
                 "n_train": int(len(y_train)), "n_test": int(len(y_test)), "features": feats,
                 "dropped_features": D["dropped"], "missing_values_imputed": D["n_missing"]},
        "selection": {"best_params": best_params, "best_epoch": best_epoch,
                      "n_trials_pruned": int((trials_df.state == "PRUNED").sum()),
                      "selection_cv_accuracy_biased": sel_cv, "selection_cv_std": sel_cv_std},
        "nested_cv": {"accuracy_mean": NESTED_ACC, "accuracy_std": NESTED_STD, "folds": nested},
        "calibration": {"temperature": TAU, "oof_nll": float(res.fun), "oof_accuracy": OOF_ACC},
        "test": {"metrics_mean_std_over_seeds": {k: {"mean": float(v["mean"]), "std": float(v["std"])}
                                                 for k, v in summary.iterrows()},
                 "accuracy_bootstrap_ci95": CI["accuracy"], "macro_f1_bootstrap_ci95": CI["macro_f1"],
                 "per_class_recall": {c: {"mean": float(recalls[:, i].mean()), "std": sd(recalls[:, i])}
                                      for i, c in enumerate(CLASSES)},
                 "mean_confusion_matrix": cm_mean.tolist(), "misclassified": wrong,
                 "representative_run": rep + 1},
        "timings_s": timings,
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # 6. figures (same as the TM notebook, Section 9)
    _figures(name, out, trials_df, best_curve, best_params, runs_df, cm_mean, CI, all_probs, y_test, TAU)
    return dict(results=results, models=models, all_preds=all_preds, all_probs=all_probs, rep=rep,
                best_params=best_params, tau=TAU)


def _figures(name, out, trials_df, best_curve, best_params, runs_df, cm_mean, CI, all_probs, y_test, TAU):
    iterative = len(best_curve["mean"]) > 1
    fig, axes = plt.subplots(1, 2 if iterative else 1, figsize=(13 if iterative else 6.5, 4.8), squeeze=False)
    ax = axes[0, 0]
    done = trials_df[trials_df.state == "COMPLETE"]
    pruned = trials_df[trials_df.state == "PRUNED"]
    ax.scatter(done.number, done.value, s=22, label="Complete trial")
    ax.scatter(pruned.number, pruned.value, s=14, marker="x", color="gray", label="Pruned (score when stopped)")
    ax.plot(done.number, done.value.cummax(), color="k", lw=1.5, label="Best so far")
    ax.set(xlabel="Trial", ylabel="Smoothed CV accuracy", title=f"{name}: Optuna history ({N_TRIALS} trials)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    if iterative:
        ax = axes[0, 1]
        ep = np.arange(1, len(best_curve["mean"]) + 1)
        ax.plot(ep, best_curve["mean"], lw=0.9, alpha=0.6, label="Mean over folds")
        ax.fill_between(ep, best_curve["mean"] - best_curve["std"], best_curve["mean"] + best_curve["std"],
                        alpha=0.12)
        ax.plot(ep, best_curve["smooth"], lw=2.2, label=f"Smoothed (±{SMOOTH} epochs)")
        ax.axvline(best_params["epochs"], ls="--", c="gray")
        ax.set(xlabel="Epoch", ylabel=f"Validation accuracy ({FOLDS}-fold × {N_SEL_SEEDS} seeds)",
               title="Best trial CV curve")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "cv_curves.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm_mean, cmap="Blues")
    ax.set_xticks(range(5), CLASSES); ax.set_yticks(range(5), CLASSES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm_mean[i, j]:.1f}", ha="center", va="center",
                    color="white" if cm_mean[i, j] > cm_mean.max() / 2 else "black")
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"{name}, test set (101–150), mean of {N_RUNS} seeds\naccuracy {runs_df.accuracy.mean():.3f} "
                 f"(95% CI {CI['accuracy'][0]:.2f}–{CI['accuracy'][1]:.2f})")
    fig.tight_layout(); fig.savefig(out / "test_confusion.png", dpi=200); plt.close(fig)

    P = np.vstack(all_probs); Y = np.tile(y_test, N_RUNS)
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, ECE_BINS + 1); centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    acc_bin, conf_bin, n_bin = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        n_bin.append(int(m.sum()))
        acc_bin.append((pred[m] == Y[m]).mean() if m.any() else np.nan)
        conf_bin.append(conf[m].mean() if m.any() else np.nan)
    acc_bin, conf_bin = np.array(acc_bin), np.array(conf_bin)
    ok = ~np.isnan(acc_bin)
    ax.bar(centers[ok], acc_bin[ok], width=1 / ECE_BINS, edgecolor="k", alpha=0.75, label="Observed accuracy")
    ax.bar(centers[ok], (conf_bin - acc_bin)[ok], bottom=acc_bin[ok], width=1 / ECE_BINS,
           color="tab:red", alpha=0.3, edgecolor="tab:red", hatch="//", label="Gap to confidence")
    for x, a, n in zip(centers[ok], acc_bin[ok], np.array(n_bin)[ok]):
        ax.text(x, a + 0.02, str(n), ha="center", fontsize=8)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set(xlim=(0, 1), ylim=(0, 1.08), xlabel="Predicted confidence (top class)", ylabel="Accuracy",
           title=f"Reliability diagram – ECE {runs_df.ece.mean():.3f} ± {sd(runs_df.ece):.3f}")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    for c, cname in enumerate(CLASSES):
        pc, yc = P[:, c], (Y == c)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pc > lo) & (pc <= hi) if lo > 0 else (pc >= lo) & (pc <= hi)
            if m.sum() >= 3:
                xs.append(pc[m].mean()); ys.append(yc[m].mean())
        ax.plot(xs, ys, "o-", ms=4, label=cname)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Observed frequency",
           title="One-vs-rest reliability per class")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{name}: calibration on the test set (101–150), {N_RUNS} seeds pooled, τ = {TAU:.2f}")
    fig.tight_layout(); fig.savefig(out / "calibration_plot.png", dpi=200, bbox_inches="tight"); plt.close(fig)
