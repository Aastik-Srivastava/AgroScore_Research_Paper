import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/mnt/user-data/uploads")
from agroscore_reproducible_study import generate_simulation, FEATURES
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path

OUT = Path("/home/claude/results2")
SEED = 20260916
df = generate_simulation(8000, SEED)
y = df.default_flag.to_numpy()
tr_i, te_i = train_test_split(np.arange(len(df)), test_size=.2, random_state=SEED, stratify=y)
tr, te = df.iloc[tr_i], df.iloc[te_i]

pipe = make_pipeline(StandardScaler(), MLPClassifier(max_iter=800, early_stopping=True, random_state=SEED))
grid = {
    "mlpclassifier__hidden_layer_sizes": [(16,), (32,), (64,), (64, 32), (32, 16)],
    "mlpclassifier__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
}
gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
gs.fit(tr[FEATURES], tr.default_flag)
best = gs.best_estimator_
p = best.predict_proba(te[FEATURES])[:, 1]
test_auc = roc_auc_score(te.default_flag, p)

pd.DataFrame([{
    "best_params": str(gs.best_params_),
    "cv_auc": gs.best_score_,
    "test_auc": test_auc,
}]).to_csv(OUT / "mlp_tuned.csv", index=False)
print("best params:", gs.best_params_)
print("CV AUC:", round(gs.best_score_, 4))
print("Test AUC:", round(test_auc, 4))
