#!/usr/bin/env python3
"""Interpretability-to-mechanism conformance diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, XGBClassifier

from agroscore_reproducible_study import (
    FEATURES,
    RANDOM_SEED,
    additive_logit_structural_weights,
    generate_simulation,
)

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("AGROSCORE_RESULTS_DIR", ROOT / "results2"))
FIG = Path(os.environ.get("AGROSCORE_FIGURE_DIR", ROOT / "figures"))
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def xgboost_shap(model: XGBClassifier, x_test: pd.DataFrame) -> np.ndarray:
    booster = model.get_booster()
    contrib = booster.predict(DMatrix(x_test, feature_names=FEATURES), pred_contribs=True)
    return np.asarray(contrib[:, :-1])


def hist_gradient_shap(model: HistGradientBoostingClassifier, x_test: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_test)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def rank_table(true_importance: pd.Series, shap_importance: pd.Series, model: str) -> pd.DataFrame:
    frame = pd.DataFrame({
        "feature": FEATURES,
        "model": model,
        "true_abs_weight": true_importance.loc[FEATURES].to_numpy(),
        "mean_abs_shap": shap_importance.loc[FEATURES].to_numpy(),
    })
    frame["true_rank"] = frame["true_abs_weight"].rank(method="average", ascending=False)
    frame["shap_rank"] = frame["mean_abs_shap"].rank(method="average", ascending=False)
    return frame


def main() -> None:
    df = generate_simulation(8000, RANDOM_SEED, outcome_regime="additive_logit")
    y = df.default_flag.to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=.2, random_state=RANDOM_SEED, stratify=y
    )
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    true_weights = pd.Series(additive_logit_structural_weights(), dtype=float)
    true_importance = true_weights.abs()

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8,
            colsample_bytree=.8, reg_lambda=1.5, eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=.055, max_leaf_nodes=23,
            min_samples_leaf=24, l2_regularization=.5, random_state=RANDOM_SEED,
        ),
    }

    metrics: dict[str, object] = {
        "target": "mean_abs_shap_vs_abs_additive_logit_structural_weight",
        "n_features": len(FEATURES),
        "models": {},
    }
    tables = []
    for name, model in models.items():
        model.fit(train_df[FEATURES], train_df.default_flag)
        if name == "XGBoost":
            shap_values = xgboost_shap(model, test_df[FEATURES])
        else:
            shap_values = hist_gradient_shap(model, test_df[FEATURES])
        shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES)
        rho, pvalue = spearmanr(true_importance.loc[FEATURES], shap_importance.loc[FEATURES])
        metrics["models"][name] = {
            "spearman_rho": float(rho),
            "spearman_pvalue": float(pvalue),
            "top_true_features": (
                true_importance.sort_values(ascending=False).head(8).index.tolist()
            ),
            "top_shap_features": (
                shap_importance.sort_values(ascending=False).head(8).index.tolist()
            ),
        }
        tables.append(rank_table(true_importance, shap_importance, name))
        print("shap", name, float(rho), flush=True)

    ranking = pd.concat(tables, ignore_index=True)
    metrics["rankings"] = ranking.to_dict(orient="records")
    with open(OUT / "shap_conformance.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    plot_conformance(ranking)


def plot_conformance(ranking: pd.DataFrame) -> None:
    top = (
        ranking.groupby("feature", as_index=False)["true_abs_weight"].first()
        .sort_values("true_abs_weight", ascending=False)
        .head(12)["feature"]
        .tolist()
    )
    plot_df = ranking[ranking["feature"].isin(top)].copy()
    plot_df["feature"] = pd.Categorical(plot_df["feature"], categories=top[::-1], ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), sharey=True)
    for ax, (model, g) in zip(axes, plot_df.groupby("model", sort=True)):
        g = g.sort_values("feature")
        y = np.arange(len(g))
        ax.barh(y - .18, g["true_abs_weight"], height=.34, color="0.75", label="True abs. weight")
        scaled = g["mean_abs_shap"] / max(g["mean_abs_shap"].max(), 1e-12)
        scaled = scaled * max(g["true_abs_weight"].max(), 1e-12)
        ax.barh(y + .18, scaled, height=.34, color="0.20", label="Mean abs. SHAP, scaled")
        ax.set_yticks(y, g["feature"])
        ax.set_title(model)
        ax.grid(axis="x", alpha=.18)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.supxlabel("Relative mechanism importance")
    fig.tight_layout()
    fig.savefig(FIG / "fig_shap_conformance.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
