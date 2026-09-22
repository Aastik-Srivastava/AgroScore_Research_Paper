#!/usr/bin/env python3
"""Counterfactual OOD stress tests for the AgroScore benchmark."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from agroscore_reproducible_study import (
    FEATURES,
    RANDOM_SEED,
    expected_calibration_error,
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


def models(seed: int = RANDOM_SEED) -> dict[str, object]:
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=8,
            random_state=seed, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=.055, max_leaf_nodes=23,
            min_samples_leaf=24, l2_regularization=.5, random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8,
            colsample_bytree=.8, reg_lambda=1.5, eval_metric="logloss",
            random_state=seed, n_jobs=-1,
        ),
    }


def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float = 0.15) -> float:
    pred = p >= threshold
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    return float(tp / len(y) - fp / len(y) * threshold / (1.0 - threshold))


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc_roc": float(roc_auc_score(y, p)),
        "ece_10": expected_calibration_error(y, p, bins=10),
        "net_benefit_015": net_benefit(y, p, threshold=0.15),
    }


def main() -> None:
    df = generate_simulation(8000, RANDOM_SEED)
    y = df.default_flag.to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=.2, random_state=RANDOM_SEED, stratify=y
    )
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    y_test = test_df.default_flag.to_numpy()

    shock_frames = {
        "baseline": test_df.copy(),
        "severity_shock": test_df.copy(),
        "decoupling_shock": test_df.copy(),
    }
    shock_frames["severity_shock"]["drought_risk"] = 0.85
    shock_frames["decoupling_shock"]["irrigation_access"] = 0.0

    rows: list[dict[str, float | str]] = []
    for model_name, model in models().items():
        model.fit(train_df[FEATURES], train_df.default_flag)
        base_prob = model.predict_proba(shock_frames["baseline"][FEATURES])[:, 1]
        base = evaluate(y_test, base_prob)
        for scenario, frame in shock_frames.items():
            prob = model.predict_proba(frame[FEATURES])[:, 1]
            met = evaluate(y_test, prob)
            rows.append({
                "model": model_name,
                "scenario": scenario,
                **met,
                "delta_auc": met["auc_roc"] - base["auc_roc"],
                "delta_ece_10": met["ece_10"] - base["ece_10"],
                "delta_net_benefit_015": (
                    met["net_benefit_015"] - base["net_benefit_015"]
                ),
            })
        print("stress", model_name, flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "stress_test.csv", index=False)
    plot_stress(result)


def plot_stress(result: pd.DataFrame) -> None:
    scenarios = ["severity_shock", "decoupling_shock"]
    metrics = [
        ("delta_auc", r"$\Delta$AUC"),
        ("delta_ece_10", r"$\Delta$ECE"),
        ("delta_net_benefit_015", r"$\Delta$Net benefit, $\tau=0.15$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), sharex=False)
    x = np.arange(len(result["model"].unique()))
    width = 0.36
    for ax, (metric, label) in zip(axes, metrics):
        for offset, scenario in [(-width / 2, scenarios[0]), (width / 2, scenarios[1])]:
            g = result[result["scenario"] == scenario].set_index("model")
            models_order = list(result["model"].unique())
            ax.bar(x + offset, g.loc[models_order, metric], width=width,
                   label=scenario.replace("_", " "))
        ax.axhline(0, color="0.45", lw=.9, ls="--")
        ax.set_title(label)
        ax.set_xticks(x, result["model"].unique(), rotation=35, ha="right")
        ax.grid(axis="y", alpha=.18)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_stress_test.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
