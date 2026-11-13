"""
Random forest baseline for ACDC cardiac pathology, evaluated with exactly the same protocol as the
Tsetlin Machine notebook (see acdc_common.py for the protocol; only the model differs).

Model-specific choices
* No feature scaling (trees are invariant to monotone transforms).
* Search space: n_estimators, max_depth, min_samples_leaf, max_features (no epochs: the CV "curve" has length 1).
* Scores for temperature scaling: log(predict_proba + 1e-6); argmax is the RF prediction and
  softmax(scores / tau) is the calibrated probability (tau = 1 gives back predict_proba up to the epsilon).
* Explainability: SHAP (TreeExplainer, uncalibrated class probabilities) of the representative run on the
  test patients: bar plot, beeswarm per class, waterfall plots for selected patients.

Run:
    python acdc_rf.py --train features_train.csv --test features_test.csv --out results_rf
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import acdc_common as C

EPS = 1e-6


def suggest(trial):
    return {"n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 1.0])}


def make_rf(p, seed, n_jobs):
    return RandomForestClassifier(**{k: v for k, v in p.items() if k != "epochs"}, random_state=int(seed),
                                  n_jobs=n_jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="features_train.csv")
    ap.add_argument("--test", default="features_test.csv")
    ap.add_argument("--out", default="results_rf")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--shap_display", type=int, default=15)
    ap.add_argument("--waterfall_patients", type=int, nargs="+", default=[101, 117, 132, 147])
    ap.add_argument("--no_shap", action="store_true")
    args = ap.parse_args()

    D = C.load_data(args.train, args.test)

    def curve(Xtr, ytr, Xva, yva, p, seed):
        rf = make_rf(p, seed, args.n_jobs).fit(Xtr, ytr)
        return np.array([np.mean(rf.predict(Xva) == yva)])

    def fit_scores(Xtr, ytr, Xev, p, seed):
        rf = make_rf(p, seed, args.n_jobs).fit(Xtr, ytr)
        proba = np.zeros((len(Xev), len(C.CLASSES)))
        proba[:, rf.classes_] = rf.predict_proba(Xev)
        return np.log(proba + EPS), rf

    R = C.run_protocol("Random forest", D, suggest, curve, fit_scores, args.out, tau_bounds=(0.05, 20.0),
                       extra_config={"model": "RandomForestClassifier", "score": "log(predict_proba + 1e-6)"},
                       pkgs=["scikit-learn", "optuna", "numpy", "pandas", "scipy", "shap", "matplotlib"])
    if args.no_shap:
        return

    # ---- SHAP (representative run, test patients, probability output) ------------------------------
    import shap
    out, feats, X_test, y_test = args.out, D["feats"], D["X_test"], D["y_test"]
    ids = D["test_df"]["patient"].tolist()
    rep, model, preds = R["rep"], R["models"][R["rep"]], R["all_preds"][R["rep"]]

    def save(name):
        plt.gcf().savefig(os.path.join(out, name), dpi=300, bbox_inches="tight"); plt.close("all")

    sv = shap.TreeExplainer(model)(X_test)
    sv = shap.Explanation(sv.values, sv.base_values, data=X_test, feature_names=feats)
    pd.DataFrame(np.abs(sv.values).mean(0), index=feats, columns=C.CLASSES).assign(
        overall=lambda d: d.sum(axis=1)).sort_values("overall", ascending=False).to_csv(
        os.path.join(out, "rf_shap_importance.csv"), index_label="feature")
    shap.summary_plot([sv.values[:, :, k] for k in range(len(C.CLASSES))], X_test, feature_names=feats,
                      class_names=C.CLASSES, plot_type="bar", max_display=args.shap_display, show=False)
    save("rf_shap_bar.png")
    for k, c in enumerate(C.CLASSES):
        shap.plots.beeswarm(sv[:, :, k], max_display=args.shap_display, show=False)
        save(f"rf_shap_beeswarm_{c}.png")
    for pid in args.waterfall_patients:
        if pid not in ids:
            continue
        i = ids.index(pid); q = preds[i]
        shap.plots.waterfall(sv[i, :, q], max_display=args.shap_display, show=False)
        save(f"rf_shap_waterfall_{pid}_true-{C.CLASSES[y_test[i]]}_pred-{C.CLASSES[q]}.png")
    print(f"SHAP plots saved (representative run {rep + 1})")


if __name__ == "__main__":
    main()
