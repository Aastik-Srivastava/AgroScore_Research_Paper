import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/mnt/user-data/uploads")
from agroscore_reproducible_study import generate_simulation, FEATURES, CLIMATE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("/home/claude/results2")
TEST_SEED = 999999   # reserved, never used for training, disjoint from all other seeds
TRAIN_SEEDS = list(range(101, 109))  # 8 independent training-draw replicates per n
NO_CLIMATE = [f for f in FEATURES if f not in CLIMATE]

def fit_auc(train_df, test_df, feats, seed):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    m.fit(train_df[feats], train_df.default_flag)
    p = m.predict_proba(test_df[feats])[:, 1]
    return roc_auc_score(test_df.default_flag, p)

# fixed, large, held-out common test population -- never touched by any training seed
test_df = generate_simulation(20000, TEST_SEED)

rows = []
for n in [1000, 2000, 4000, 8000, 16000, 32000]:
    for s in TRAIN_SEEDS:
        train_df = generate_simulation(n, s)
        auc_full = fit_auc(train_df, test_df, FEATURES, s)
        auc_noclim = fit_auc(train_df, test_df, NO_CLIMATE, s)
        rows.append({"n": n, "seed": s, "experiment": "Full", "auc": auc_full})
        rows.append({"n": n, "seed": s, "experiment": "No climate", "auc": auc_noclim})
    print("n =", n, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "learning_curve_v2_raw.csv", index=False)

summary = df.groupby(["n", "experiment"]).auc.agg(["mean", "std", "count"]).reset_index()
summary.to_csv(OUT / "learning_curve_v2_summary.csv", index=False)
print(summary.round(4).to_string())
