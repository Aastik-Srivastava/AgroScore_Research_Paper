import os, json, numpy as np, pandas as pd
from agroscore_reproducible_study import (generate_simulation, FEATURES, metrics,
                                          calibration_intercept_slope, expected_calibration_error)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score, brier_score_loss
from scipy.stats import wasserstein_distance
from pathlib import Path
OUT = Path(os.environ.get("AGROSCORE_RESULTS_DIR", Path(__file__).resolve().parent / "results2"))
OUT.mkdir(parents=True, exist_ok=True)
S = 20260916
def M(s=S): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=s))
df = generate_simulation(8000, S); y = df.default_flag.to_numpy()
a, b = train_test_split(np.arange(len(df)), test_size=.2, random_state=S, stratify=y)
tr, te = df.iloc[a], df.iloc[b]; cut = int(.8*len(df)); otr, ote = df.iloc[:cut], df.iloc[cut:]
m = M(); m.fit(tr[FEATURES], tr.default_flag); pr = m.predict_proba(te[FEATURES])[:,1]
m2 = M(); m2.fit(otr[FEATURES], otr.default_flag); po = m2.predict_proba(ote[FEATURES])[:,1]
pd.DataFrame({"farmer_id": te.farmer_id.values, "region": te.region.values,
              "default_flag": te.default_flag.values, "predicted_pd": pr}).to_csv(OUT/"pred_random.csv", index=False)
pd.DataFrame({"farmer_id": ote.farmer_id.values, "region": ote.region.values,
              "default_flag": ote.default_flag.values, "predicted_pd": po}).to_csv(OUT/"pred_oot.csv", index=False)
def cal(yy, p, split, bins=10):
    f = pd.DataFrame({"o": yy, "p": p}); f["bin"] = pd.qcut(f.p, bins, labels=False, duplicates="drop")
    g = f.groupby("bin").agg(n=("o","size"), mean_predicted_pd=("p","mean"), observed_default_rate=("o","mean")).reset_index()
    g.insert(0,"split",split); return g
pd.concat([cal(te.default_flag.values, pr, "Random test"), cal(ote.default_flag.values, po, "Out-of-time test")]
          ).to_csv(OUT/"calibration_bins.csv", index=False)
rows=[]
for split, yy, p in [("Random test", te.default_flag.values, pr), ("Out-of-time test", ote.default_flag.values, po)]:
    for t in [.05,.10,.15,.20,.25,.30]:
        pred=(p>=t).astype(int); tn,fp,fn,tp = confusion_matrix(yy,pred,labels=[0,1]).ravel()
        rows.append({"split":split,"threshold":t,"flag_rate":pred.mean(),
                     "sensitivity":recall_score(yy,pred,zero_division=0),"specificity":tn/(tn+fp),
                     "precision":precision_score(yy,pred,zero_division=0),
                     "net_benefit":tp/len(yy)-fp/len(yy)*t/(1-t)})
pd.DataFrame(rows).to_csv(OUT/"threshold_diagnostics.csv", index=False)
tr_rows=[]; years = pd.to_datetime(df.disbursement_date).dt.year
for ty in sorted(years.unique())[2:]:
    a1=df.loc[years<ty]; b1=df.loc[years==ty]
    mm=M(); mm.fit(a1[FEATURES], a1.default_flag); p=mm.predict_proba(b1[FEATURES])[:,1]
    tr_rows.append({"test_year":int(ty),"train_n":len(a1),"test_n":len(b1),
                    "default_rate":float(b1.default_flag.mean()), **metrics(b1.default_flag.values,p),
                    "ece_10":expected_calibration_error(b1.default_flag.values,p,bins=10)})
pd.DataFrame(tr_rows).to_csv(OUT/"rolling_temporal.csv", index=False)
logo=LeaveOneGroupOut(); lr=[]
for i,j in logo.split(df[FEATURES], y, df.region.values):
    mm=M(); mm.fit(df[FEATURES].iloc[i], y[i]); p=mm.predict_proba(df[FEATURES].iloc[j])[:,1]
    lr.append({"held_out_region":str(df.region.values[j][0]),"n":len(j),**metrics(y[j],p),
               "ece_10":expected_calibration_error(y[j],p,bins=10)})
pd.DataFrame(lr).sort_values("held_out_region").to_csv(OUT/"logo.csv", index=False)
sub=[]
for g in np.unique(te.region):
    k=te.region.values==g
    sub.append({"region":g,"n":int(k.sum()),"default_rate":float(te.default_flag.values[k].mean()),
                "mean_predicted_pd":float(pr[k].mean()),"auc_roc":float(roc_auc_score(te.default_flag.values[k],pr[k])),
                "brier":float(brier_score_loss(te.default_flag.values[k],pr[k]))})
sub=pd.DataFrame(sub); sub.to_csv(OUT/"regional.csv", index=False)
u=np.unique(te.region)
mw=max(wasserstein_distance(pr[te.region.values==x],pr[te.region.values==z]) for i,x in enumerate(u) for z in u[i+1:])
json.dump({"random":{**metrics(te.default_flag.values,pr),
                     "ece_10":expected_calibration_error(te.default_flag.values,pr,bins=10)},
           "oot":{**metrics(ote.default_flag.values,po),
                  "ece_10":expected_calibration_error(ote.default_flag.values,po,bins=10)},
           "cal_random":calibration_intercept_slope(te.default_flag.values,pr),
           "cal_oot":calibration_intercept_slope(ote.default_flag.values,po),
           "logo_mean":float(pd.DataFrame(lr).auc_roc.mean()),"logo_sd":float(pd.DataFrame(lr).auc_roc.std(ddof=1)),
           "max_wasserstein":float(mw)}, open(OUT/"summary2.json","w"), indent=2)
print("DONE")
