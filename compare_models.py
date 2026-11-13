"""
Paired comparison of the Tsetlin Machine, random forest and MLP on the same 50 test patients.

Inputs: the result folders written by the TM notebook and by acdc_rf.py / acdc_dnn.py (results.json and
test_predictions.csv in each). All three models are evaluated on the same patients, so differences are
tested with paired methods:

* Paired bootstrap of the difference in seed-averaged accuracy (patients resampled with the same indices for
  both models, 2000 resamples): 95% percentile CI and two-sided bootstrap p-value.
* Exact McNemar test on the seed-ensemble predictions (argmax of the seed-averaged calibrated probabilities).
* Holm correction over the three pairwise comparisons of each feature set.

Run:
    python compare_models.py --sets f26=results/f26 f43=results/f43 f57=results/f57 --out results/comparison
    (each folder contains the subfolders tm/, rf/ and dnn/)
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

CLASSES = ["DCM", "HCM", "MINF", "NOR", "RV"]
MODELS = {"tm": "Tsetlin Machine", "rf": "Random forest", "dnn": "MLP"}


def load(folder):
    folder = Path(folder)
    res = json.load(open(folder / "results.json"))
    pred = pd.read_csv(folder / "test_predictions.csv").sort_values("patient").reset_index(drop=True)
    y = pred["true"].map(CLASSES.index).to_numpy()
    correct = 1.0 - pred["error_rate_over_seeds"].to_numpy()            # seed-averaged correctness
    P = pred[[f"p_{c}" for c in CLASSES]].to_numpy()
    ens = None if np.isnan(P).any() else P.argmax(1) == y               # seed-ensemble correctness
    return res, pred["patient"].to_numpy(), y, correct, ens


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def fmt(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help="name=folder (folder contains tm/, rf/, dnn/)")
    ap.add_argument("--out", default="results/comparison")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    summary, paired = [], []
    for spec in args.sets:
        name, folder = spec.split("=", 1)
        L = {k: load(Path(folder) / k) for k in MODELS if (Path(folder) / k / "results.json").exists()}
        pids = {k: v[1] for k, v in L.items()}
        ref = next(iter(pids.values()))
        assert all(np.array_equal(ref, p) for p in pids.values()), "test patients differ between models"
        for k, (res, _, y, correct, ens) in L.items():
            t = res["test"]; mm = t["metrics_mean_std_over_seeds"]
            summary.append({
                "feature_set": name, "model": MODELS[k], "n_features": len(res["data"]["features"]),
                "nested_cv_acc": fmt(res["nested_cv"]["accuracy_mean"], res["nested_cv"]["accuracy_std"]),
                "test_acc": fmt(mm["accuracy"]["mean"], mm["accuracy"]["std"]),
                "test_acc_ci95": "[{:.3f}, {:.3f}]".format(*t["accuracy_bootstrap_ci95"]),
                "test_macro_f1": fmt(mm["macro_f1"]["mean"], mm["macro_f1"]["std"]),
                "test_macro_f1_ci95": "[{:.3f}, {:.3f}]".format(*t["macro_f1_bootstrap_ci95"]),
                "ece": fmt(mm["ece"]["mean"], mm["ece"]["std"]),
                "brier": fmt(mm["brier"]["mean"], mm["brier"]["std"]),
                "nll": fmt(mm["nll"]["mean"], mm["nll"]["std"]),
                "ensemble_acc": float(ens.mean()) if ens is not None else float("nan"),
                **{f"recall_{c}": fmt(v["mean"], v["std"], 2) for c, v in t["per_class_recall"].items()},
                "temperature": res["calibration"]["temperature"],
            })
        rows = []
        for a, b in itertools.combinations(L, 2):
            ca, cb = L[a][3], L[b][3]
            d = ca - cb
            rng = np.random.default_rng(args.seed)
            boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.n_boot)])
            p_boot = min(1.0, 2 * (min((boots <= 0).sum(), (boots >= 0).sum()) + 1) / (args.n_boot + 1))  # never 0
            ea, eb = L[a][4], L[b][4]
            if ea is None or eb is None:                               # probabilities not available
                n01 = n10 = None; p_mc = float("nan")
            else:
                n01, n10 = int((~ea & eb).sum()), int((ea & ~eb).sum())   # only b right / only a right
                p_mc = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            rows.append({"feature_set": name, "comparison": f"{MODELS[a]} − {MODELS[b]}",
                         "acc_diff": float(d.mean()),
                         "acc_diff_ci95": "[{:+.3f}, {:+.3f}]".format(*np.percentile(boots, [2.5, 97.5])),
                         "p_bootstrap": p_boot,
                         "mcnemar_only_first_correct": n10, "mcnemar_only_second_correct": n01,
                         "p_mcnemar_exact": p_mc})
        for key in ("p_bootstrap", "p_mcnemar_exact"):
            ok = [i for i, r in enumerate(rows) if not np.isnan(r[key])]
            adj = holm([rows[i][key] for i in ok]) if ok else []
            for r in rows:
                r[key + "_holm"] = float("nan")
            for i, v in zip(ok, adj):
                rows[i][key + "_holm"] = float(v)
        paired += rows

    S, P = pd.DataFrame(summary), pd.DataFrame(paired)
    S.to_csv(out / "summary.csv", index=False)
    P.to_csv(out / "paired_tests.csv", index=False)
    with open(out / "comparison.md", "w") as f:
        f.write("## Test-set and nested-CV results (mean ± SD over 10 seeds; 95% bootstrap CI over patients)\n\n")
        f.write(S.drop(columns=[c for c in S.columns if c.startswith("recall_")]).to_markdown(index=False))
        f.write("\n\n## Per-class recall\n\n")
        f.write(S[["feature_set", "model"] + [f"recall_{c}" for c in CLASSES]].to_markdown(index=False))
        f.write("\n\n## Paired tests (same 50 test patients)\n\n")
        f.write(P.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    print(open(out / "comparison.md").read())


if __name__ == "__main__":
    main()
