"""Publication figures (vector PDF): mean confusion matrices and global feature importance for TM, RF, DNN."""
import json, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "axes.titlesize": 8,
                     "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "xtick.color": "#444", "ytick.color": "#444", "axes.labelcolor": "#222"})
CL = ["DCM", "HCM", "MINF", "NOR", "RV"]
COL = dict(zip(CL, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))   # validated categorical palette
SETS = [("f26", 26), ("f43", 43), ("f57", 57)]
MODELS = [("tm", "TM"), ("rf", "RF"), ("dnn", "DNN")]
TM26_CM = np.array([[8.4, 0, 1.6, 0, 0], [0, 9.8, 0, 0.2, 0], [2.2, 0.3, 7.5, 0, 0],
                    [0, 1.5, 0.1, 7.3, 1.1], [0, 0.3, 0, 0, 9.7]])          # printed in the TM notebook (26 features)
short = lambda f: re.sub(r"^[WIK]__", "", f)

# ---- 1. confusion matrices (3 feature sets x 3 models) -------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(6.6, 6.6), constrained_layout=True)
for r, (d, n) in enumerate(SETS):
    for c, (m, name) in enumerate(MODELS):
        R = json.load(open(f"results/{d}/{m}/results.json"))
        cm = TM26_CM if (d, m) == ("f26", "tm") else np.array(R["test"]["mean_confusion_matrix"])
        acc = R["test"]["metrics_mean_std_over_seeds"]["accuracy"]["mean"]
        ax = axes[r, c]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=10)
        for i in range(5):
            for j in range(5):
                v = cm[i, j]
                if v > 0.05:
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                            color="white" if v > 5 else "#222")
        ax.set_xticks(range(5), CL); ax.set_yticks(range(5), CL)
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(f"{name}, {n} features (acc. {acc:.3f})")
        if r == 2: ax.set_xlabel("Predicted")
        if c == 0: ax.set_ylabel("True")
fig.colorbar(im, ax=axes, shrink=0.5, label="Patients (mean of 10 seeds)")
fig.savefig("figures/fig_confusion.pdf"); plt.close(fig)

# ---- 2. feature importance, one figure per feature set (TM | RF | DNN) ---------------------------------
TOP = 12
for d, n in SETS:
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.3), constrained_layout=True)
    for ax, (m, name) in zip(axes, MODELS):
        if m == "tm":
            df = pd.read_csv(f"results/{d}/tm/feature_importance_clauses.csv", index_col=0)[CL]
            xlabel = "Share of positive clause weight"
        else:
            df = pd.read_csv(f"results/{d}/{m}/{'rf' if m == 'rf' else 'dnn'}_shap_importance.csv", index_col=0)[CL]
            xlabel = "Share of mean |SHAP|"
        share = df / df.values.sum()                       # each model: total importance = 1
        share = share.loc[share.sum(axis=1).sort_values(ascending=False).index[:TOP]][::-1]
        left = np.zeros(len(share))
        for cl in CL:
            ax.barh(range(len(share)), share[cl], left=left, color=COL[cl], height=0.72,
                    edgecolor="white", linewidth=0.6, label=cl)
            left += share[cl].values
        ax.axvline(1 / n, color="#666", lw=0.8, ls=(0, (3, 2)))
        ax.set_yticks(range(len(share)), [short(f) for f in share.index])
        ax.set_xlabel(xlabel); ax.set_title(name, fontweight="bold")
        ax.grid(axis="x", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, max(left.max() * 1.08, 1.5 / n))
    h, l = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    fig.legend(h + [Line2D([], [], color="#666", lw=0.8, ls=(0, (3, 2)))], l + ["Uniform share"],
               loc="outside lower center", ncol=6, frameon=False)
    fig.savefig(f"figures/fig_importance_{n}.pdf"); plt.close(fig)
print("ok")
