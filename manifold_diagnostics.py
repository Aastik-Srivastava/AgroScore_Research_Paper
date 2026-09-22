#!/usr/bin/env python3
"""Manifold and embedding stability diagnostics for independent cohorts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
from sklearn.preprocessing import StandardScaler

from agroscore_reproducible_study import FEATURES, generate_simulation

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("AGROSCORE_RESULTS_DIR", ROOT / "results2"))
OUT.mkdir(parents=True, exist_ok=True)


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    cov_a = np.cov(a, rowvar=False)
    cov_b = np.cov(b, rowvar=False)
    covmean = sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = np.sum((mu_a - mu_b) ** 2) + np.trace(cov_a + cov_b - 2.0 * covmean)
    return float(max(value, 0.0))


def main() -> None:
    s1 = generate_simulation(8000, 20260916)
    s2 = generate_simulation(8000, 20260917)
    scaler = StandardScaler().fit(s1[FEATURES])
    x1 = scaler.transform(s1[FEATURES])
    x2 = scaler.transform(s2[FEATURES])

    rng = np.random.default_rng(314159)
    w = rng.normal(0.0, 1.0 / np.sqrt(len(FEATURES)), size=(len(FEATURES), 64))
    z1 = x1 @ w
    z2 = x2 @ w

    audit = {
        "seed_1": 20260916,
        "seed_2": 20260917,
        "n_1": int(len(s1)),
        "n_2": int(len(s2)),
        "feature_dim": int(len(FEATURES)),
        "latent_dim": 64,
        "projection_seed": 314159,
        "frechet_distance": frechet_distance(z1, z2),
        "mean_norm_1": float(np.linalg.norm(z1.mean(axis=0))),
        "mean_norm_2": float(np.linalg.norm(z2.mean(axis=0))),
        "trace_cov_1": float(np.trace(np.cov(z1, rowvar=False))),
        "trace_cov_2": float(np.trace(np.cov(z2, rowvar=False))),
    }
    with open(OUT / "manifold_audit.json", "w") as handle:
        json.dump(audit, handle, indent=2)
    print("manifold FD", audit["frechet_distance"], flush=True)


if __name__ == "__main__":
    main()
