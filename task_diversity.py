#!/usr/bin/env python3
"""Supplementary LGD and time-to-default task benchmarks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from agroscore_reproducible_study import FEATURES, RANDOM_SEED, generate_simulation

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("AGROSCORE_RESULTS_DIR", ROOT / "results2"))
OUT.mkdir(parents=True, exist_ok=True)


def cox_ph_fit(x: np.ndarray, time: np.ndarray, event: np.ndarray, l2: float = 1.0) -> np.ndarray:
    order = np.argsort(-time)
    x_ord = x[order]
    event_ord = event[order].astype(bool)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(x_ord @ beta, -30.0, 30.0)
        exp_eta = np.exp(eta)
        cum_risk = np.cumsum(exp_eta)
        weighted_x = np.cumsum(exp_eta[:, None] * x_ord, axis=0)
        mean_risk_x = weighted_x / cum_risk[:, None]
        loglik = np.sum(eta[event_ord] - np.log(cum_risk[event_ord]))
        grad = np.sum(x_ord[event_ord] - mean_risk_x[event_ord], axis=0)
        penalty = 0.5 * l2 * np.sum(beta ** 2)
        return float(-loglik + penalty), -grad + l2 * beta

    result = minimize(
        fun=lambda b: objective(b)[0],
        x0=np.zeros(x.shape[1], dtype=float),
        jac=lambda b: objective(b)[1],
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Cox optimization failed: {result.message}")
    return np.asarray(result.x)


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    permissible = 0
    concordant = 0.0
    n = len(time)
    for i in range(n):
        if not event[i]:
            continue
        comparable = time[i] < time
        ties = comparable & (risk[i] == risk)
        wins = comparable & (risk[i] > risk)
        permissible += int(comparable.sum())
        concordant += float(wins.sum()) + 0.5 * float(ties.sum())
    if permissible == 0:
        return float("nan")
    return float(concordant / permissible)


def main() -> None:
    df = generate_simulation(8000, RANDOM_SEED, include_task_targets=True)
    y = df.default_flag.to_numpy()
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=.2, random_state=RANDOM_SEED, stratify=y
    )
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    rows: list[dict[str, float | str]] = []

    lgd_train = train_df[train_df.default_flag == 1].copy()
    lgd_test = test_df[test_df.default_flag == 1].copy()
    regressors = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "XGBoost regressor": XGBRegressor(
            n_estimators=250, max_depth=3, learning_rate=.04, subsample=.85,
            colsample_bytree=.85, reg_lambda=2.0, objective="reg:squarederror",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
    }
    for name, model in regressors.items():
        model.fit(lgd_train[FEATURES], lgd_train["lgd"])
        pred = np.clip(model.predict(lgd_test[FEATURES]), 0.05, 0.95)
        scores = {
            "r2": float(r2_score(lgd_test["lgd"], pred)),
            "mae": float(mean_absolute_error(lgd_test["lgd"], pred)),
        }
        for metric, value in scores.items():
            rows.append({
                "task": "lgd",
                "model": name,
                "metric": metric,
                "value": value,
                "n_train": int(len(lgd_train)),
                "n_test": int(len(lgd_test)),
            })
        print("lgd", name, flush=True)

    scaler = StandardScaler().fit(train_df[FEATURES])
    x_train = scaler.transform(train_df[FEATURES])
    x_test = scaler.transform(test_df[FEATURES])
    beta = cox_ph_fit(
        x_train,
        train_df["time_to_default"].to_numpy(),
        train_df["default_event_observed"].to_numpy(),
        l2=1.0,
    )
    risk = x_test @ beta
    rows.append({
        "task": "time_to_default",
        "model": "Cox PH ridge",
        "metric": "c_index",
        "value": harrell_c_index(
            test_df["time_to_default"].to_numpy(),
            test_df["default_event_observed"].to_numpy().astype(bool),
            risk,
        ),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    })
    pd.DataFrame(rows).to_csv(OUT / "task_diversity.csv", index=False)
    print("survival Cox PH ridge", flush=True)


if __name__ == "__main__":
    main()
