#!/usr/bin/env python3
"""Deterministic unit tests for the AgroScore reproduction artifact."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from agroscore_reproducible_study import (
    FEATURES,
    OUTCOME_REGIMES,
    PDOScore,
    additive_logit_structural_weights,
    generate_simulation,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results2"
FIGURES = ROOT / "figures"


class GeneratorTests(unittest.TestCase):
    def test_feature_count_and_schema(self) -> None:
        cohort = generate_simulation(128, 20260916)
        self.assertEqual(len(FEATURES), 28)
        self.assertTrue(set(FEATURES).issubset(cohort.columns))
        self.assertFalse(cohort[FEATURES].isna().any().any())

    def test_generation_is_deterministic(self) -> None:
        left = generate_simulation(128, 20260916)
        right = generate_simulation(128, 20260916)
        self.assertTrue(left.equals(right))

    def test_output_is_time_ordered(self) -> None:
        cohort = generate_simulation(128, 20260916)
        self.assertTrue(cohort["disbursement_date"].is_monotonic_increasing)

    def test_every_outcome_regime_is_deterministic_and_non_degenerate(self) -> None:
        for regime in OUTCOME_REGIMES:
            left = generate_simulation(512, 20260916, regime)
            right = generate_simulation(512, 20260916, regime)
            self.assertTrue(left.equals(right))
            self.assertEqual(left["outcome_regime"].iat[0], regime)
            self.assertGreater(left["default_flag"].mean(), .04)
            self.assertLess(left["default_flag"].mean(), .30)

    def test_unknown_outcome_regime_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            generate_simulation(64, 20260916, "unknown")

    def test_optional_task_targets_are_deterministic_and_bounded(self) -> None:
        left = generate_simulation(512, 20260916, include_task_targets=True)
        right = generate_simulation(512, 20260916, include_task_targets=True)
        self.assertTrue(left.equals(right))
        for column in ["lgd", "time_to_default", "default_event_observed", "latent_risk_index"]:
            self.assertIn(column, left.columns)
        default_rows = left["default_flag"] == 1
        self.assertFalse(left.loc[default_rows, "lgd"].isna().any())
        self.assertTrue(left.loc[default_rows, "lgd"].between(.05, .95).all())
        self.assertTrue(left["time_to_default"].between(.25, 60.0).all())
        self.assertTrue(set(left["default_event_observed"].unique()).issubset({0, 1}))

    def test_additive_structural_weights_cover_feature_schema(self) -> None:
        weights = additive_logit_structural_weights()
        self.assertEqual(set(weights), set(FEATURES))
        self.assertGreater(abs(weights["payment_history"]), abs(weights["wpi_inflation"]))
        self.assertEqual(weights["repo_rate"], 0.0)


class PDOScoreTests(unittest.TestCase):
    def test_six_reference_probabilities(self) -> None:
        probabilities = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 0.95])
        expected = np.array([695, 600, 557, 510, 430, 300])
        np.testing.assert_array_equal(PDOScore().transform(probabilities), expected)

    def test_anchor_and_points_to_double_odds(self) -> None:
        score = PDOScore()
        anchor = score.anchor_pd
        doubled_bad_odds_pd = (2 * anchor / (1 - anchor)) / (1 + 2 * anchor / (1 - anchor))
        self.assertEqual(int(score.transform(anchor)), 600)
        self.assertEqual(int(score.transform(doubled_bad_odds_pd)), 560)


class NewArtifactTests(unittest.TestCase):
    def test_stress_test_artifact_schema(self) -> None:
        path = RESULTS / "stress_test.csv"
        self.assertTrue(path.exists(), f"missing {path}")
        frame = pd.read_csv(path)
        self.assertEqual(len(frame), 12)
        self.assertEqual(set(frame["scenario"]), {"baseline", "severity_shock", "decoupling_shock"})
        numeric = frame.select_dtypes(include=[np.number])
        self.assertFalse(numeric.isna().any().any())
        self.assertTrue((FIGURES / "fig_stress_test.pdf").exists())

    def test_shap_conformance_artifact_schema(self) -> None:
        path = RESULTS / "shap_conformance.json"
        self.assertTrue(path.exists(), f"missing {path}")
        with open(path) as handle:
            data = json.load(handle)
        self.assertEqual(data["n_features"], len(FEATURES))
        self.assertEqual(set(data["models"]), {"XGBoost", "HistGradientBoosting"})
        for payload in data["models"].values():
            self.assertTrue(np.isfinite(payload["spearman_rho"]))
            self.assertEqual(len(payload["top_true_features"]), 8)
            self.assertEqual(len(payload["top_shap_features"]), 8)
        self.assertEqual(len(data["rankings"]), 2 * len(FEATURES))
        self.assertTrue((FIGURES / "fig_shap_conformance.pdf").exists())

    def test_manifold_audit_artifact_schema(self) -> None:
        path = RESULTS / "manifold_audit.json"
        self.assertTrue(path.exists(), f"missing {path}")
        with open(path) as handle:
            data = json.load(handle)
        self.assertEqual(data["n_1"], 8000)
        self.assertEqual(data["n_2"], 8000)
        self.assertEqual(data["feature_dim"], len(FEATURES))
        self.assertEqual(data["latent_dim"], 64)
        self.assertTrue(np.isfinite(data["frechet_distance"]))
        self.assertGreaterEqual(data["frechet_distance"], 0.0)

    def test_task_diversity_artifact_schema(self) -> None:
        path = RESULTS / "task_diversity.csv"
        self.assertTrue(path.exists(), f"missing {path}")
        frame = pd.read_csv(path)
        self.assertEqual(set(frame["task"]), {"lgd", "time_to_default"})
        self.assertIn("Cox PH ridge", set(frame["model"]))
        self.assertEqual((frame["task"] == "lgd").sum(), 4)
        self.assertEqual((frame["task"] == "time_to_default").sum(), 1)
        self.assertEqual(set(frame["metric"]), {"r2", "mae", "c_index"})
        for column in ["n_train", "n_test"]:
            self.assertTrue((frame[column] > 0).all())
        self.assertFalse(frame.isna().any().any())
        self.assertTrue(np.isfinite(frame["value"]).all())


if __name__ == "__main__":
    unittest.main()
