#!/usr/bin/env python3
"""Reproducible simulation study for AgroScore.

This module intentionally makes no claim of clinical, regulatory, or production
validation.  It generates a documented synthetic cohort, evaluates identical
models on fixed splits, writes machine-readable results, and implements a
correctly anchored points-to-double-the-odds (PDO) transformation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

RANDOM_SEED = 20260916
REGIONS = ("Punjab", "Maharashtra", "UP", "Karnataka", "AP", "WB", "Gujarat", "MP")

FINANCIAL = [
    "debt_to_income", "loan_to_land", "payment_history", "savings_ratio",
    "credit_utilization", "prior_default", "annual_income_log",
]
CLIMATE = [
    "drought_risk", "soil_moisture", "ndvi_current", "ndvi_anomaly",
    "rainfall_deviation", "temperature_stress", "flood_risk",
    "climate_debt_stress",
]
FARM = [
    "farmer_age", "education_level", "family_size", "land_size_log",
    "irrigation_access", "insurance_coverage", "crop_diversification",
    "soil_health", "market_distance_log", "price_volatility", "price_trend",
]
MACRO = ["repo_rate", "wpi_inflation"]
FEATURES = FINANCIAL + CLIMATE + FARM + MACRO
OUTCOME_REGIMES = ("additive_logit", "threshold", "interaction", "hybrid")


def additive_logit_structural_weights() -> dict[str, float]:
    """Feature-level coefficients in the additive-logit outcome mechanism.

    Nonlinear transformed terms are represented by the coefficient multiplying
    the transformed feature contribution, so this mapping is an importance
    target rather than a linear scorecard that exactly reconstructs the logit.
    """
    weights = {feature: 0.0 for feature in FEATURES}
    weights.update({
        "payment_history": -2.45,
        "debt_to_income": 1.45,
        "prior_default": 0.85,
        "climate_debt_stress": 0.85,
        "price_volatility": 0.50,
        "insurance_coverage": -0.45,
        "irrigation_access": -0.35,
        "soil_health": -0.30,
        "price_trend": 0.28,
        "wpi_inflation": 0.15 / 3.0,
    })
    return weights


@dataclass(frozen=True)
class PDOScore:
    """Map probability of default to score with a good:bad odds anchor."""

    anchor_score: int = 600
    anchor_good_to_bad_odds: float = 19.0
    pdo: int = 40
    minimum: int = 300
    maximum: int = 900

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2.0)

    @property
    def anchor_pd(self) -> float:
        return 1.0 / (1.0 + self.anchor_good_to_bad_odds)

    def transform(self, probability_default: float | np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability_default, dtype=float), 1e-7, 1 - 1e-7)
        bad_odds = p / (1 - p)
        anchor_bad_odds = self.anchor_pd / (1 - self.anchor_pd)
        score = self.anchor_score - self.factor * np.log(bad_odds / anchor_bad_odds)
        return np.clip(np.rint(score), self.minimum, self.maximum).astype(int)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _calibrate_intercept(systematic: np.ndarray, target_rate: float = 0.123) -> float:
    """Choose an intercept giving a common expected prevalence across regimes."""
    lo, hi = -12.0, 4.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(systematic + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate_simulation(
    n: int = 8000,
    seed: int = RANDOM_SEED,
    outcome_regime: str = "additive_logit",
    include_task_targets: bool = False,
) -> pd.DataFrame:
    """Generate a temporally ordered, region-structured synthetic cohort.

    The generator is a simulation test bed, not a calibrated representation of
    an Indian loan portfolio.  Outcome generation includes unobserved noise and
    a time shock so random-split performance cannot be mistaken for deployment
    performance. ``outcome_regime`` selects one of four prespecified structural
    outcome mechanisms, so no single learner family is privileged by the
    benchmark definition.
    """

    rng = np.random.default_rng(seed)
    region = rng.choice(REGIONS, n, p=[.12, .14, .15, .12, .12, .11, .12, .12])
    month = rng.integers(0, 60, n)
    date = pd.Timestamp("2019-01-01") + pd.to_timedelta(month * 30, unit="D")

    drought_base = {"Punjab": .18, "Maharashtra": .55, "UP": .28, "Karnataka": .62,
                    "AP": .48, "WB": .20, "Gujarat": .42, "MP": .65}
    irrigation_p = {"Punjab": .91, "Maharashtra": .45, "UP": .72, "Karnataka": .38,
                    "AP": .54, "WB": .78, "Gujarat": .64, "MP": .34}
    land_mean = {"Punjab": 4.0, "Maharashtra": 2.7, "UP": 1.8, "Karnataka": 2.9,
                 "AP": 2.4, "WB": 1.5, "Gujarat": 3.3, "MP": 4.2}

    reg_drought = np.array([drought_base[x] for x in region])
    irrigation = rng.binomial(1, np.array([irrigation_p[x] for x in region])).astype(float)
    seasonal = .12 * np.sin(2 * np.pi * month / 12) + .003 * month
    drought = np.clip(reg_drought + seasonal + rng.normal(0, .13, n) - .18 * irrigation, 0, 1)
    soil_moisture = np.clip(.78 - .63 * drought + .12 * irrigation + rng.normal(0, .09, n), 0, 1)
    ndvi_anomaly = np.clip(-.75 * drought + .35 * irrigation + rng.normal(0, .32, n), -2, 2)
    ndvi_current = np.clip(.52 + .13 * ndvi_anomaly + .12 * soil_moisture + rng.normal(0, .06, n), .05, .92)
    rainfall_deviation = np.clip(rng.normal(-.45 * drought, .35, n), -1, 1)
    flood_risk = np.clip(rng.beta(1.5, 7, n) + np.maximum(rainfall_deviation, 0) * .45, 0, 1)
    temperature_stress = np.clip(.20 + .42 * drought + rng.normal(0, .14, n), 0, 1)

    land = np.clip(np.array([land_mean[x] for x in region]) * rng.lognormal(0, .45, n), .3, 20)
    education = np.clip(np.rint(rng.normal(2.8, 1.1, n)), 1, 5)
    age = np.clip(rng.normal(45, 12, n), 21, 75)
    family = np.clip(rng.poisson(4.2, n), 1, 11)
    insurance = rng.binomial(1, np.clip(.24 + .08 * education + .12 * irrigation, .1, .85)).astype(float)
    diversification = np.clip(rng.beta(2.4, 4.2, n) + .05 * np.log1p(land), 0, 1)
    soil_health = np.clip(.48 + .18 * irrigation + .12 * soil_moisture + rng.normal(0, .12, n), 0, 1)
    market_distance = np.clip(rng.gamma(2.4, 7.0, n) * (1.12 - .05 * education), .5, 120)
    price_volatility = np.clip(rng.beta(2.2, 7, n) + .10 * drought, .02, .85)
    price_trend = np.clip(rng.normal(.01 - .08 * drought, .09, n), -.35, .35)

    income = np.clip(45_000 * land * (1 + .35 * irrigation) * (1 - .22 * drought)
                     * rng.lognormal(0, .35, n), 25_000, 3_000_000)
    debt_to_income = np.clip(rng.beta(2.2, 4.5, n) + .12 * drought, 0, 1.5)
    loan_to_land = np.clip(debt_to_income * income / (150_000 * land + 1), 0, 3)
    payment_history = np.clip(.86 - .40 * debt_to_income - .14 * price_volatility
                              + rng.normal(0, .12, n), .05, 1)
    savings_ratio = np.clip(rng.beta(2.0, 8.0, n) - .10 * debt_to_income, 0, .6)
    credit_utilization = np.clip(.16 + .78 * debt_to_income + rng.normal(0, .11, n), 0, 1)
    prior_default = rng.binomial(1, _sigmoid(-3.2 + 2.0 * debt_to_income - 1.8 * payment_history))
    climate_debt = np.sqrt(np.clip(drought * np.clip(debt_to_income, 0, 1), 0, 1)) \
        * (1 + .2 * np.maximum(-ndvi_anomaly, 0))
    repo = 5.0 + .025 * month + rng.normal(0, .18, n)
    wpi = np.clip(3.1 + .055 * month + rng.normal(0, .75, n), .2, 10)

    if outcome_regime not in OUTCOME_REGIMES:
        raise ValueError(f"Unknown outcome_regime={outcome_regime!r}; expected one of {OUTCOME_REGIMES}")

    shock = .35 * (month >= 48)
    if outcome_regime == "additive_logit":
        systematic = (-3.65 + 2.45 * (1 - payment_history) + 1.45 * debt_to_income
                      + .85 * prior_default + .85 * climate_debt + .50 * price_volatility
                      - .45 * insurance - .35 * irrigation - .30 * soil_health
                      + .28 * np.maximum(-price_trend, 0) + .15 * (wpi - 4) / 3
                      + shock)
        latent = systematic + rng.normal(0, .78, n)
    else:
        if outcome_regime == "threshold":
            systematic = (
                1.55 * (debt_to_income > .52)
                + 1.35 * (payment_history < .58)
                + .95 * ((drought > .62) & (irrigation == 0))
                + .75 * ((credit_utilization > .72) & (savings_ratio < .08))
                + .60 * prior_default
                - .55 * ((insurance == 1) & (soil_health > .62))
                + .40 * (market_distance > 22) + shock
            )
        elif outcome_regime == "interaction":
            systematic = (
                2.10 * (debt_to_income > .48) * (payment_history < .62)
                + 1.65 * (drought > .58) * (irrigation == 0) * (insurance == 0)
                + 1.35 * (ndvi_anomaly < -.30) * (price_volatility > .30)
                + 1.10 * (flood_risk > .32) * (market_distance > 20)
                + .85 * prior_default
                - .90 * (savings_ratio > .18) * (soil_health > .62)
                + shock
            )
        else:
            systematic = (
                1.20 * (1 - payment_history) + .80 * debt_to_income
                + 1.10 * (debt_to_income > .50) * (payment_history < .62)
                + 1.40 * drought * price_volatility * (1 - insurance)
                + .75 * np.maximum(-ndvi_anomaly, 0) * credit_utilization
                + .55 * np.sin(np.pi * np.clip(soil_moisture, 0, 1))
                + .45 * ((land > 2.0) & (market_distance < 18))
                + .55 * prior_default + shock
            )
        systematic = _calibrate_intercept(systematic) + systematic
        latent = systematic + rng.normal(0, .78, n)
    probability = _sigmoid(latent)
    default = rng.binomial(1, probability)

    data = pd.DataFrame({
        "farmer_id": [f"SIM-{i:06d}" for i in range(n)], "region": region,
        "disbursement_date": date, "default_flag": default,
        "outcome_regime": outcome_regime,
        "debt_to_income": debt_to_income, "loan_to_land": loan_to_land,
        "payment_history": payment_history, "savings_ratio": savings_ratio,
        "credit_utilization": credit_utilization, "prior_default": prior_default,
        "annual_income_log": np.log1p(income), "drought_risk": drought,
        "soil_moisture": soil_moisture, "ndvi_current": ndvi_current,
        "ndvi_anomaly": ndvi_anomaly, "rainfall_deviation": rainfall_deviation,
        "temperature_stress": temperature_stress, "flood_risk": flood_risk,
        "climate_debt_stress": climate_debt, "farmer_age": age,
        "education_level": education, "family_size": family,
        "land_size_log": np.log1p(land), "irrigation_access": irrigation,
        "insurance_coverage": insurance, "crop_diversification": diversification,
        "soil_health": soil_health, "market_distance_log": np.log1p(market_distance),
        "price_volatility": price_volatility, "price_trend": price_trend,
        "repo_rate": repo, "wpi_inflation": wpi,
    })
    if include_task_targets:
        lgd_raw = (0.18 + 0.18 * loan_to_land + 0.09 * np.log1p(market_distance)
                   + 0.10 * debt_to_income + rng.normal(0, 0.09, n))
        lgd = np.where(default == 1, np.clip(lgd_raw, 0.05, 0.95), np.nan)

        duration_scale = 18.0 * np.exp(-0.35 * systematic)
        event_time = duration_scale * rng.weibull(1.45, n)
        censor_time = rng.uniform(6.0, 60.0, n)
        observed_time = np.minimum(event_time, censor_time)
        event_observed = event_time <= censor_time

        data["lgd"] = lgd
        data["time_to_default"] = np.clip(observed_time, 0.25, 60.0)
        data["default_event_observed"] = event_observed.astype(int)
        data["latent_risk_index"] = systematic
    return data.sort_values(["disbursement_date", "farmer_id"]).reset_index(drop=True)


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    return float(ks_2samp(p[y == 1], p[y == 0], alternative="two-sided", mode="auto").statistic)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc_roc": float(roc_auc_score(y, p)),
        "auc_pr": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "gini": float(2 * roc_auc_score(y, p) - 1),
        "brier": float(brier_score_loss(y, p)),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Equal-width expected calibration error used by the manuscript."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    membership = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = membership == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration intercept and slope on the logit of predictions."""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    recalibrator = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    recalibrator.fit(logit_p, y)
    return float(recalibrator.intercept_[0]), float(recalibrator.coef_[0, 0])
