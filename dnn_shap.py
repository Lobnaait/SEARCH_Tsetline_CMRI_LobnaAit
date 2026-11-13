"""Post-hoc SHAP for the DNN: retrain the representative run (same seed as acdc_dnn.py, verified to reproduce
its test accuracy) and explain its class probabilities on the test patients with a model-agnostic permutation
explainer (background = training set). Saves the same outputs as the RF script: importance CSV, bar plot,
one beeswarm per class and waterfall plots (predicted class) for selected patients."""
import json, os, sys
import numpy as np, pandas as pd, torch, shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import acdc_common as C, acdc_dnn as N

torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
tr, te, out = sys.argv[1:4]
WATERFALL = [101, 117, 132, 147]; DISPLAY = 15
D = C.load_data(tr, te); R = json.load(open(f"{out}/results.json"))
p = R["selection"]["best_params"]; rep = R["test"]["representative_run"] - 1
Xtr, Xte, feats = D["X_train"], D["X_test"], D["feats"]
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
model, opt = N.make_model(Xtr.shape[1], p, C.SEED * 1000 + 1000 + rep)
Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); yt = torch.tensor(D["y_train"])
for _ in range(p["epochs"]): N.train_epoch(model, opt, Xt, yt, p["batch_size"])
f = lambda X: C.softmax(N.logits_of(model, torch.tensor((X - mu) / sd, dtype=torch.float32)))
pred = f(Xte).argmax(1)
acc = (pred == D["y_test"]).mean()
assert abs(acc - pd.read_csv(f"{out}/runs_metrics.csv").accuracy[rep]) < 1e-9, "representative run not reproduced"

ex = shap.PermutationExplainer(f, shap.maskers.Independent(Xtr, max_samples=100), seed=C.SEED)  # fixed seed: reproducible values
sv = ex(Xte, max_evals=2 * Xte.shape[1] * 10 + 1, silent=True)
sv = shap.Explanation(sv.values, sv.base_values, data=Xte, feature_names=feats)
pd.DataFrame(np.abs(sv.values).mean(0), index=feats, columns=C.CLASSES).assign(
    overall=lambda d: d.sum(1)).sort_values("overall", ascending=False).to_csv(f"{out}/dnn_shap_importance.csv", index_label="feature")

def save(name):
    plt.gcf().savefig(os.path.join(out, name), dpi=300, bbox_inches="tight"); plt.close("all")
shap.summary_plot([sv.values[:, :, k] for k in range(len(C.CLASSES))], Xte, feature_names=feats,
                  class_names=C.CLASSES, plot_type="bar", max_display=DISPLAY, show=False)
save("dnn_shap_bar.png")
for k, c in enumerate(C.CLASSES):
    shap.plots.beeswarm(sv[:, :, k], max_display=DISPLAY, show=False); save(f"dnn_shap_beeswarm_{c}.png")
ids = D["test_df"]["patient"].tolist()
for pid in WATERFALL:
    i = ids.index(pid); q = pred[i]
    shap.plots.waterfall(sv[i, :, q], max_display=DISPLAY, show=False)
    save(f"dnn_shap_waterfall_{pid}_true-{C.CLASSES[D['y_test'][i]]}_pred-{C.CLASSES[q]}.png")
print(out, f"rep run {rep + 1} reproduced (acc {acc:.3f}); figures saved")
