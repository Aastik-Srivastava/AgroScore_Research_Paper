#!/usr/bin/env python3
"""Repeated, equal-input benchmark over prespecified AgroScore outcome regimes."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from agroscore_reproducible_study import FEATURES, OUTCOME_REGIMES, generate_simulation

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results2"
OUT.mkdir(exist_ok=True)
SEEDS = range(20260916, 20260926)


def models(seed):
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)),
        "Random forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=6, max_features="sqrt", n_jobs=-1, random_state=seed),
        "Extra trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=6, max_features="sqrt", n_jobs=-1, random_state=seed + 1),
        "Histogram boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=.5, random_state=seed),
    }


def main():
    rows = []
    for regime in OUTCOME_REGIMES:
        for seed in SEEDS:
            cohort = generate_simulation(8000, seed, regime)
            y = cohort.default_flag.to_numpy()
            train, test = train_test_split(np.arange(len(cohort)), test_size=.2, stratify=y, random_state=seed)
            for name, model in models(seed).items():
                model.fit(cohort[FEATURES].iloc[train], y[train])
                probability = model.predict_proba(cohort[FEATURES].iloc[test])[:, 1]
                rows.append({"regime": regime, "seed": seed, "model": name,
                             "test_prevalence": float(y[test].mean()),
                             "auc_roc": float(roc_auc_score(y[test], probability))})
            print(regime, seed, flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "regime_benchmark_raw.csv", index=False)
    summary = raw.groupby(["regime", "model"], as_index=False).agg(
        mean_auc=("auc_roc", "mean"), sd_auc=("auc_roc", "std"),
        min_auc=("auc_roc", "min"), max_auc=("auc_roc", "max"))
    summary["rank"] = summary.groupby("regime")["mean_auc"].rank(method="min", ascending=False).astype(int)
    summary.sort_values(["regime", "rank", "model"]).to_csv(OUT / "regime_benchmark_summary.csv", index=False)
    print(summary.sort_values(["regime", "rank", "model"]).to_string(index=False))


if __name__ == "__main__":
    main()
