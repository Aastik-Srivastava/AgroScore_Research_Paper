#!/usr/bin/env python3
"""Generate all AgroScore figures from machine-readable results."""
from pathlib import Path
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R = Path(os.environ.get("AGROSCORE_RESULTS_DIR", ROOT / "results2"))
O = Path(os.environ.get("AGROSCORE_FIGURE_DIR", ROOT / "figures"))
O.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.size":8.5,"axes.labelsize":8.5,"axes.titlesize":9,
                     "legend.fontsize":7,"xtick.labelsize":7.5,"ytick.labelsize":7.5,
                     "figure.dpi":160,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False,
                     "pdf.fonttype":42,"ps.fonttype":42})
def save(fig,name):
    fig.tight_layout(); fig.savefig(O/f"{name}.pdf",bbox_inches="tight"); plt.close(fig); print(name)

# 1 ROC all models
roc = pd.read_csv(R/"roc_all.csv"); bench = pd.read_csv(R/"benchmark_random.csv").set_index("model")
bench_oot = pd.read_csv(R/"benchmark_oot.csv").set_index("model")
fig, ax = plt.subplots(figsize=(3.4,2.7))
for name, g in roc[roc.split=="Random test"].groupby("model"):
    ax.plot(g.fpr, g.tpr, lw=1.3, label=f"{name} ({bench.auc_roc[name]:.3f})")
g = roc[roc.split=="Out-of-time"]
oot_auc = bench_oot.loc["AgroScore stack", "auc_roc"]
ax.plot(g.fpr, g.tpr, lw=1.6, ls=":", color="k", label=f"Stack, out-of-time ({oot_auc:.3f})")
ax.plot([0,1],[0,1],color="0.6",lw=.9,ls="--")
ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0,1), ylim=(0,1))
ax.grid(alpha=.18); ax.legend(frameon=False, loc="lower right", fontsize=6)
save(fig,"fig_roc_all")

# 2 forest plot delta vs LR
d = pd.read_csv(R/"delta_vs_logistic.csv").sort_values("delta_auc_vs_lr")
y = np.arange(len(d)); xe = np.vstack([d.delta_auc_vs_lr-d.ci_low, d.ci_high-d.delta_auc_vs_lr])
fig, ax = plt.subplots(figsize=(3.4,2.3))
ax.errorbar(d.delta_auc_vs_lr, y, xerr=xe, fmt="o", ms=4, color="0.15", capsize=2.5, lw=1.1)
ax.axvline(0,color="0.55",ls="--",lw=1)
ax.set_yticks(y, d.model); ax.set(xlabel=r"$\Delta$AUC vs. logistic regression")
ax.grid(axis="x",alpha=.18); save(fig,"fig_model_delta")

# 3 calibration
cal = pd.read_csv(R/"calibration_bins.csv")
fig, ax = plt.subplots(figsize=(3.4,2.6))
for s,g in cal.groupby("split"):
    ax.plot(g.mean_predicted_pd, g.observed_default_rate, marker="o", ms=3.5, lw=1.3, label=s)
ax.plot([0,.45],[0,.45],color="0.6",lw=.9,ls="--",label="Ideal")
ax.set(xlabel="Mean predicted PD", ylabel="Observed default rate", xlim=(0,.45), ylim=(0,.45))
ax.grid(alpha=.18); ax.legend(frameon=False,loc="upper left"); save(fig,"fig_calibration")

# 4 correlation heatmap
c = pd.read_csv(R/"corr_matrix.csv", index_col=0)
fig, ax = plt.subplots(figsize=(6.9,5.6))
im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=5.5)
ax.set_yticks(range(len(c)), c.index, fontsize=5.5)
fig.colorbar(im, ax=ax, shrink=.8, label="Spearman correlation")
save(fig,"fig_corr")

# 5 t-SNE two panels
t = pd.read_csv(R/"tsne.csv")
fig, axes = plt.subplots(1,2,figsize=(6.9,2.9))
axes[0].scatter(t.x,t.y,c=t["default"],cmap="coolwarm",s=5,alpha=.7)
axes[0].set_title("Coloured by simulated default"); 
for r in sorted(t.region.unique()):
    k=t.region==r; axes[1].scatter(t.x[k],t.y[k],s=5,alpha=.7,label=r)
axes[1].set_title("Coloured by region")
axes[1].legend(frameon=False,fontsize=5.5,ncol=2,markerscale=2)
for a in axes: a.set(xlabel="t-SNE 1", ylabel="t-SNE 2"); a.set_xticks([]); a.set_yticks([])
save(fig,"fig_tsne")

# 6 PCA scree
p = pd.read_csv(R/"pca.csv")
fig, ax = plt.subplots(figsize=(3.4,2.3))
ax.bar(p.component, p.explained, color="0.6", label="Individual")
ax2 = ax.twinx(); ax2.plot(p.component, p.cumulative, marker="o", ms=3.5, color="0.1", label="Cumulative")
ax.set(xlabel="Principal component", ylabel="Explained variance ratio")
ax2.set(ylabel="Cumulative", ylim=(0,1)); ax.set_xticks(p.component)
save(fig,"fig_pca")

# 7 KS + detection
k = pd.read_csv(R/"ks_two_cohorts.csv").sort_values("ks")
fig, ax = plt.subplots(figsize=(3.4,3.4))
ax.barh(range(len(k)), k.ks, color="0.55")
ax.set_yticks(range(len(k)), k.feature, fontsize=5.5)
ax.axvline(.05,color="crimson",ls="--",lw=1,label="0.05 reference")
ax.set(xlabel="KS statistic, two independent cohorts")
ax.legend(frameon=False,loc="lower right"); save(fig,"fig_ks")

# 8 multiseed ablation box
m = pd.read_csv(R/"multiseed_ablation.csv"); m = m[m.experiment!="Full"]
order = m.groupby("experiment").delta.mean().sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(3.4,2.5))
ax.boxplot([m.delta[m.experiment==e] for e in order], vert=False, widths=.55,
           patch_artist=True, boxprops=dict(facecolor="0.85"), medianprops=dict(color="0.1"))
ax.axvline(0,color="0.55",ls="--",lw=1)
ax.set_yticks(range(1,len(order)+1), order)
ax.set(xlabel=r"$\Delta$AUC vs. full model (10 seeds)")
ax.grid(axis="x",alpha=.18); save(fig,"fig_multiseed")

# 9 learning curve
lc = pd.read_csv(R/"learning_curve.csv")
fig, ax = plt.subplots(figsize=(3.4,2.3))
for e,g in lc.groupby("experiment"):
    ax.plot(g.n, g.auc, marker="o", ms=3.5, lw=1.3, label=e)
ax.set_xscale("log"); ax.set(xlabel="Cohort size (log scale)", ylabel="AUC-ROC")
ax.grid(alpha=.18); ax.legend(frameon=False); save(fig,"fig_learning")

# 10 temporal
tp = pd.read_csv(R/"rolling_temporal.csv")
fig, ax1 = plt.subplots(figsize=(3.4,2.4))
ax1.plot(tp.test_year, tp.auc_roc, marker="o", lw=1.4, color="0.1", label="AUC")
ax1.set(xlabel="Held-out year", ylabel="AUC-ROC", ylim=(.55,.8)); ax1.set_xticks(tp.test_year)
ax2 = ax1.twinx()
ax2.plot(tp.test_year, tp.default_rate, marker="s", ls="--", lw=1.2, color="0.5", label="Default rate")
ax2.plot(tp.test_year, tp.ece_10, marker="^", ls=":", lw=1.2, color="crimson", label="ECE")
ax2.set(ylabel="Default rate / ECE", ylim=(0,.22))
ln = ax1.lines+ax2.lines; ax1.legend(ln,[l.get_label() for l in ln],frameon=False,fontsize=6.5,loc="lower left")
ax1.grid(alpha=.18); save(fig,"fig_temporal")

# 11 regional
lg = pd.read_csv(R/"logo.csv"); rg = pd.read_csv(R/"regional.csv")
mg = lg.merge(rg[["region","auc_roc"]].rename(columns={"region":"held_out_region","auc_roc":"rand"}),on="held_out_region")
mg = mg.sort_values("auc_roc"); y = np.arange(len(mg))
fig, ax = plt.subplots(figsize=(3.4,2.7))
ax.scatter(mg.auc_roc,y,s=26,marker="o",label="Leave-one-region-out")
ax.scatter(mg["rand"],y,s=30,marker="x",label="Random-split subgroup")
ax.axvline(.5,color="0.6",ls="--",lw=.9)
ax.set_yticks(y,mg.held_out_region); ax.set(xlabel="AUC-ROC",xlim=(.5,.9))
ax.grid(axis="x",alpha=.18); ax.legend(frameon=False,loc="lower right"); save(fig,"fig_regional")

# 12 marginals grid
from agroscore_reproducible_study import generate_simulation
df = generate_simulation(8000, 20260916)
sel = ["debt_to_income","payment_history","credit_utilization","annual_income_log",
       "drought_risk","ndvi_current","soil_moisture","land_size_log",
       "price_volatility","climate_debt_stress","market_distance_log","loan_to_land"]
fig, axes = plt.subplots(3,4,figsize=(6.9,4.2))
for a,f in zip(axes.ravel(), sel):
    a.hist(df[f],bins=40,color="0.6",edgecolor="none")
    a.set_title(f.replace("_"," "),fontsize=6.5); a.set_yticks([]); a.tick_params(labelsize=5.5)
save(fig,"fig_marginals")

# 13 threshold / decision curve
th = pd.read_csv(R/"threshold_diagnostics.csv")
fig, axes = plt.subplots(1,2,figsize=(6.9,2.5))
for s,g in th.groupby("split"):
    axes[0].plot(g.threshold,g.net_benefit,marker="o",ms=3.5,lw=1.3,label=s)
    axes[1].plot(g.threshold,g.flag_rate,marker="s",ms=3.5,lw=1.3,label=s)
axes[0].set(xlabel="PD threshold",ylabel="Standardised net benefit")
axes[1].set(xlabel="PD threshold",ylabel="Flag rate")
for a in axes: a.grid(alpha=.18); a.legend(frameon=False)
save(fig,"fig_threshold")
print("ALL FIGURES DONE")
