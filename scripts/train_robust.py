"""Train models that generalize across space. Use spatial CV for model selection,
trim features to local-only, heavily regularize."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb

DATA = Path(__file__).parent
train = pd.read_csv(DATA / "train_features.csv")
test = pd.read_csv(DATA / "test_features.csv")
glcm_tr = pd.read_csv(DATA / "train_glcm.csv")
glcm_te = pd.read_csv(DATA / "test_glcm.csv")

train = pd.concat([train.reset_index(drop=True), glcm_tr.reset_index(drop=True)], axis=1)
test = pd.concat([test.reset_index(drop=True), glcm_te.reset_index(drop=True)], axis=1)

classes = sorted(train["class"].unique())
c2i = {c: i for i, c in enumerate(classes)}
i2c = {i: c for c, i in c2i.items()}
y = np.array([c2i[c] for c in train["class"].values])
NC = len(classes)
xy = train[["x", "y"]].values

DROP = ["class", "ID", "x", "y"]
ALL_FEATS = [c for c in train.columns if c not in DROP]
# Local-only features (no large-window context)
LOCAL_FEATS = [c for c in ALL_FEATS if not any(s in c for s in ["_r51", "_r101", "_r25", "_w32"])]
print(f"ALL feats: {len(ALL_FEATS)}   LOCAL feats: {len(LOCAL_FEATS)}")

# Spatial groups
N_CLUSTERS = 20
km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
groups = km.fit_predict(xy)

configs = [
    ("baseline", ALL_FEATS, dict(num_leaves=63, min_data_in_leaf=20, lambda_l2=1.0, feature_fraction=0.9)),
    ("reg_mild", ALL_FEATS, dict(num_leaves=31, min_data_in_leaf=50, lambda_l2=5.0, feature_fraction=0.7)),
    ("reg_hard", ALL_FEATS, dict(num_leaves=15, min_data_in_leaf=100, lambda_l2=20.0, feature_fraction=0.5)),
    ("local", LOCAL_FEATS, dict(num_leaves=31, min_data_in_leaf=30, lambda_l2=3.0, feature_fraction=0.8)),
    ("local_hard", LOCAL_FEATS, dict(num_leaves=15, min_data_in_leaf=80, lambda_l2=10.0, feature_fraction=0.5)),
]

def run_cv(X, y, groups, cv_type, params_extra):
    p = dict(objective="multiclass", num_class=NC, learning_rate=0.05,
             metric="multi_logloss", verbose=-1, class_weight="balanced", seed=42,
             bagging_fraction=0.9, bagging_freq=5)
    p.update(params_extra)
    oof = np.zeros((len(X), NC))
    if cv_type == "random":
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = kf.split(X, y)
    else:
        kf = GroupKFold(n_splits=5)
        splits = kf.split(X, y, groups)
    for tr_i, va_i in splits:
        dtr = lgb.Dataset(X[tr_i], y[tr_i])
        dva = lgb.Dataset(X[va_i], y[va_i], reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof[va_i] = m.predict(X[va_i], num_iteration=m.best_iteration)
    return oof


print(f"\n{'config':<15}{'feats':<8}{'randCV':<10}{'spatCV':<10}{'delta':<10}")
print("-" * 55)
results = []
for name, feats, params_extra in configs:
    X = train[feats].values.astype(np.float32)
    oof_r = run_cv(X, y, groups, "random", params_extra)
    oof_s = run_cv(X, y, groups, "spatial", params_extra)
    f_r = f1_score(y, oof_r.argmax(1), average="macro")
    f_s = f1_score(y, oof_s.argmax(1), average="macro")
    results.append((name, feats, params_extra, f_r, f_s))
    print(f"{name:<15}{len(feats):<8}{f_r:<10.4f}{f_s:<10.4f}{f_r-f_s:<10.4f}")

# Train the best spatial CV config on full data + generate test preds
results.sort(key=lambda x: -x[4])  # Sort by spatCV desc
best_name, best_feats, best_params, best_r, best_s = results[0]
print(f"\nBest spatial CV config: {best_name} (randCV={best_r:.4f} spatCV={best_s:.4f})")

X = train[best_feats].values.astype(np.float32)
Xte = test[best_feats].values.astype(np.float32)

# 5-seed bag with random folds
p = dict(objective="multiclass", num_class=NC, learning_rate=0.05,
         metric="multi_logloss", verbose=-1, class_weight="balanced",
         bagging_fraction=0.9, bagging_freq=5)
p.update(best_params)

pte = np.zeros((len(Xte), NC))
for seed in [42, 7, 123, 2024, 31]:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_i, va_i in kf.split(X, y):
        pp = dict(p, seed=seed)
        dtr = lgb.Dataset(X[tr_i], y[tr_i])
        dva = lgb.Dataset(X[va_i], y[va_i], reference=dtr)
        m = lgb.train(pp, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        pte += m.predict(Xte, num_iteration=m.best_iteration) / (5 * 5)

np.save(DATA / f"pte_robust_{best_name}.npy", pte)
sub = pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in pte.argmax(1)]})
sub.to_csv(DATA / f"sub_robust_{best_name}.csv", index=False)
base = pd.read_csv(DATA / "sub_lgb_baseline.csv").sort_values("ID").reset_index(drop=True)
s = sub.sort_values("ID").reset_index(drop=True)
diffs = s[s["class"] != base["class"]]
print(f"\nRobust {best_name} sub: {len(diffs)} flips from baseline")
print(diffs.merge(test[["ID","x","y"]], on="ID"))