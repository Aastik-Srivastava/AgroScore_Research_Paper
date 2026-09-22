#!/usr/bin/env python3
"""Extended fidelity, benchmarking and stability analysis for AgroScore."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from agroscore_reproducible_study import (
    generate_simulation, FEATURES, FINANCIAL, CLIMATE, FARM, MACRO,
    metrics, expected_calibration_error, calibration_intercept_slope,
    RANDOM_SEED,
)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ks_2samp, skew, kurtosis, spearmanr
from xgboost import XGBClassifier

OUT = Path(os.environ.get("AGROSCORE_RESULTS_DIR", Path(__file__).resolve().parent / "results2"))
OUT.mkdir(parents=True, exist_ok=True)
SEED = RANDOM_SEED


def models(seed=SEED):
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
        "Random forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=8, random_state=seed, n_jobs=-1),
        "Extra trees": ExtraTreesClassifier(n_estimators=300, max_depth=11, min_samples_leaf=6, random_state=seed + 1, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=.055, max_leaf_nodes=23, min_samples_leaf=24, l2_regularization=.5, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.8, reg_lambda=1.5, eval_metric="logloss", random_state=seed, n_jobs=-1),
        "MLP (64-32)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=600, early_stopping=True, random_state=seed)),
        "AgroScore stack": StackingClassifier(
            estimators=[("rf", RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=8, random_state=seed, n_jobs=-1)),
                        ("et", ExtraTreesClassifier(n_estimators=200, max_depth=11, min_samples_leaf=6, random_state=seed + 1, n_jobs=-1)),
                        ("hgb", HistGradientBoostingClassifier(max_iter=200, learning_rate=.055, max_leaf_nodes=23, min_samples_leaf=24, l2_regularization=.5, random_state=seed))],
            final_estimator=make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)),
            stack_method="predict_proba", cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            passthrough=True, n_jobs=-1),
    }


def paired_delta_ci(y, pa, pb, draws=2000, seed=SEED):
    """CI for AUC(b) - AUC(a)."""
    rng = np.random.default_rng(seed); v = []
    for _ in range(draws):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        v.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
    v = np.array(v)
    return float(v.mean()), float(np.quantile(v, .025)), float(np.quantile(v, .975)), float((v <= 0).mean())


def boot_ci(y, p, draws=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = {**{k: [] for k in metrics(y, p)}, "ece_10": []}
    for _ in range(draws):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        for k, v in metrics(y[i], p[i]).items(): vals[k].append(v)
        vals["ece_10"].append(expected_calibration_error(y[i], p[i], bins=10))
    return {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in vals.items()}


def main():
    df = generate_simulation(8000, SEED)
    y = df.default_flag.to_numpy()
    tr_i, te_i = train_test_split(np.arange(len(df)), test_size=.2, random_state=SEED, stratify=y)
    tr, te = df.iloc[tr_i], df.iloc[te_i]
    cut = int(.8 * len(df)); otr, ote = df.iloc[:cut], df.iloc[cut:]

    # ---------- 1. Benchmark ----------
    rows, probs, oot_rows, oot_probs = [], {}, [], {}
    for name, m in models().items():
        m.fit(tr[FEATURES], tr.default_flag)
        p = m.predict_proba(te[FEATURES])[:, 1]
        probs[name] = p
        i, s = calibration_intercept_slope(te.default_flag.to_numpy(), p)
        rows.append({"model": name, **metrics(te.default_flag.to_numpy(), p),
                     "ece_10": expected_calibration_error(te.default_flag.to_numpy(), p, bins=10),
                     "cal_intercept": i, "cal_slope": s})
        m2 = models()[name]; m2.fit(otr[FEATURES], otr.default_flag)
        po = m2.predict_proba(ote[FEATURES])[:, 1]; oot_probs[name] = po
        oi, os = calibration_intercept_slope(ote.default_flag.to_numpy(), po)
        oot_rows.append({"model": name, **metrics(ote.default_flag.to_numpy(), po),
                         "ece_10": expected_calibration_error(ote.default_flag.to_numpy(), po, bins=10),
                         "cal_intercept": oi, "cal_slope": os})
        print("done", name, flush=True)
    bench = pd.DataFrame(rows); bench.to_csv(OUT / "benchmark_random.csv", index=False)
    pd.DataFrame(oot_rows).to_csv(OUT / "benchmark_oot.csv", index=False)

    # pairwise vs logistic
    ref = probs["Logistic regression"]
    cmp_rows = []
    for name, p in probs.items():
        if name == "Logistic regression": continue
        d, lo, hi, pr = paired_delta_ci(te.default_flag.to_numpy(), ref, p)
        cmp_rows.append({"model": name, "delta_auc_vs_lr": d, "ci_low": lo, "ci_high": hi, "p_one_sided": pr})
    pd.DataFrame(cmp_rows).to_csv(OUT / "delta_vs_logistic.csv", index=False)

    # CIs for best model
    json.dump({"stack_random_ci": boot_ci(te.default_flag.to_numpy(), probs["AgroScore stack"]),
               "stack_oot_ci": boot_ci(ote.default_flag.to_numpy(), oot_probs["AgroScore stack"])},
              open(OUT / "cis.json", "w"), indent=2)

    # ROC curves for all models
    rc = []
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(te.default_flag, p)
        k = max(1, len(fpr) // 300)
        rc += [{"split": "Random test", "model": name, "fpr": float(a), "tpr": float(b)} for a, b in zip(fpr[::k], tpr[::k])]
    fpr, tpr, _ = roc_curve(ote.default_flag, oot_probs["AgroScore stack"])
    k = max(1, len(fpr) // 300)
    rc += [{"split": "Out-of-time", "model": "AgroScore stack", "fpr": float(a), "tpr": float(b)} for a, b in zip(fpr[::k], tpr[::k])]
    pd.DataFrame(rc).to_csv(OUT / "roc_all.csv", index=False)

    # ---------- 2. Fidelity: marginals ----------
    ref_spec = {  # (lo, hi) plausible ranges from agricultural-finance literature//policy sources
        "debt_to_income": (.15, .75), "payment_history": (.4, 1.0), "credit_utilization": (.1, .95),
        "drought_risk": (0, 1), "ndvi_current": (.1, .9), "soil_moisture": (.05, 1),
        "irrigation_access": (0, 1), "insurance_coverage": (0, 1), "price_volatility": (.02, .85),
    }
    mrows = []
    for f in FEATURES:
        v = df[f].to_numpy()
        lo, hi = ref_spec.get(f, (np.nan, np.nan))
        inside = np.nan if np.isnan(lo) else float(((v >= lo) & (v <= hi)).mean())
        mrows.append({"feature": f, "mean": v.mean(), "sd": v.std(ddof=1), "p05": np.quantile(v, .05),
                      "median": np.median(v), "p95": np.quantile(v, .95), "skew": float(skew(v)),
                      "kurtosis": float(kurtosis(v)), "in_reference_range": inside})
    pd.DataFrame(mrows).to_csv(OUT / "marginals.csv", index=False)

    # correlation matrix
    corr = df[FEATURES].corr(method="spearman")
    corr.to_csv(OUT / "corr_matrix.csv")

    # expected sign conformance from the DGP
    expected = {("drought_risk", "ndvi_anomaly"): -1, ("drought_risk", "soil_moisture"): -1,
                ("debt_to_income", "payment_history"): -1, ("debt_to_income", "credit_utilization"): 1,
                ("irrigation_access", "drought_risk"): -1, ("soil_moisture", "soil_health"): 1,
                ("drought_risk", "temperature_stress"): 1, ("drought_risk", "price_volatility"): 1,
                ("ndvi_anomaly", "ndvi_current"): 1, ("drought_risk", "rainfall_deviation"): -1}
    srows = []
    for (a, b), sgn in expected.items():
        r = float(corr.loc[a, b]); srows.append({"pair": f"{a} vs {b}", "expected_sign": sgn,
                                                 "observed_rho": r, "conforms": bool(np.sign(r) == sgn)})
    pd.DataFrame(srows).to_csv(OUT / "sign_conformance.csv", index=False)

    # ---------- 3. Fidelity: generator stability (detection test) ----------
    det_rows = []
    for s2 in [SEED + 1, SEED + 2, SEED + 3]:
        df2 = generate_simulation(8000, s2)
        Z = pd.concat([df[FEATURES], df2[FEATURES]]); lab = np.r_[np.zeros(len(df)), np.ones(len(df2))]
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=0)
        auc = cross_val_score(clf, Z, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()
        det_rows.append({"compare_seed": s2, "detection_auc": float(auc),
                         "default_rate": float(df2.default_flag.mean())})
    pd.DataFrame(det_rows).to_csv(OUT / "detection_auc.csv", index=False)

    # KS per feature between two independent cohorts
    df2 = generate_simulation(8000, SEED + 1)
    ks_rows = [{"feature": f, "ks": float(ks_2samp(df[f], df2[f]).statistic),
                "pvalue": float(ks_2samp(df[f], df2[f]).pvalue)} for f in FEATURES]
    pd.DataFrame(ks_rows).to_csv(OUT / "ks_two_cohorts.csv", index=False)

    # privacy: distance to closest record
    scaler = StandardScaler().fit(df[FEATURES])
    A = scaler.transform(df[FEATURES])
    B = scaler.transform(df2[FEATURES])
    nn = NearestNeighbors(n_neighbors=2).fit(A)
    d_cross, _ = NearestNeighbors(n_neighbors=1).fit(A).kneighbors(B)
    d_within, _ = nn.kneighbors(A)
    json.dump({"median_dcr_cross": float(np.median(d_cross)),
               "median_nn_within": float(np.median(d_within[:, 1])),
               "dcr_ratio": float(np.median(d_cross) / np.median(d_within[:, 1]))},
              open(OUT / "near_duplicate_audit.json", "w"), indent=2)

    # ---------- 4. Embeddings ----------
    idx = np.random.default_rng(0).choice(len(df), 1500, replace=False)
    Xs = StandardScaler().fit_transform(df[FEATURES].iloc[idx])
    pca = PCA(n_components=10).fit(StandardScaler().fit_transform(df[FEATURES]))
    pd.DataFrame({"component": np.arange(1, 11), "explained": pca.explained_variance_ratio_,
                  "cumulative": np.cumsum(pca.explained_variance_ratio_)}).to_csv(OUT / "pca.csv", index=False)
    ts = TSNE(n_components=2, perplexity=30, init="pca", random_state=0,
              max_iter=500).fit_transform(Xs)
    pd.DataFrame({"x": ts[:, 0], "y": ts[:, 1], "default": df.default_flag.iloc[idx].to_numpy(),
                  "region": df.region.iloc[idx].to_numpy()}).to_csv(OUT / "tsne.csv", index=False)

    # ---------- 5. Multi-seed ablation stability ----------
    abl_sets = {"Full": FEATURES, "No climate": [f for f in FEATURES if f not in CLIMATE],
                "No financial": [f for f in FEATURES if f not in FINANCIAL],
                "No compound": [f for f in FEATURES if f != "climate_debt_stress"],
                "No farm/market": [f for f in FEATURES if f not in FARM],
                "No macro": [f for f in FEATURES if f not in MACRO]}
    srs = []
    for s in range(20260916, 20260916 + 10):
        d = generate_simulation(8000, s); yy = d.default_flag.to_numpy()
        a, b = train_test_split(np.arange(len(d)), test_size=.2, random_state=s, stratify=yy)
        base = {}
        for nm, fs in abl_sets.items():
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s))
            m.fit(d[fs].iloc[a], yy[a]); base[nm] = roc_auc_score(yy[b], m.predict_proba(d[fs].iloc[b])[:, 1])
        for nm in abl_sets:
            srs.append({"seed": s, "experiment": nm, "auc": base[nm], "delta": base[nm] - base["Full"]})
        print("seed", s, flush=True)
    pd.DataFrame(srs).to_csv(OUT / "multiseed_ablation.csv", index=False)

    # ---------- 6. Sample-size learning curve ----------
    lc = []
    for n in [1000, 2000, 4000, 8000, 16000, 32000]:
        d = generate_simulation(n, SEED); yy = d.default_flag.to_numpy()
        a, b = train_test_split(np.arange(n), test_size=.2, random_state=SEED, stratify=yy)
        for nm, fs in [("Full", FEATURES),
                       ("No climate", [f for f in FEATURES if f not in CLIMATE]),
                       ("No financial", [f for f in FEATURES if f not in FINANCIAL])]:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
            m.fit(d[fs].iloc[a], yy[a])
            lc.append({"n": n, "experiment": nm, "auc": roc_auc_score(yy[b], m.predict_proba(d[fs].iloc[b])[:, 1])})
        print("lc", n, flush=True)
    pd.DataFrame(lc).to_csv(OUT / "learning_curve.csv", index=False)

    print("ALL DONE")


if __name__ == "__main__":
    main()
