"""Train LightGBM on v1 features + GLCM textures, 5-seed bag, compare to baseline."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb

DATA = Path(__file__).parent
tr_v1 = pd.read_csv(DATA / "train_features.csv")
te_v1 = pd.read_csv(DATA / "test_features.csv")
tr_g = pd.read_csv(DATA / "train_glcm.csv")
te_g = pd.read_csv(DATA / "test_glcm.csv")

# Combine
tr = pd.concat([tr_v1.reset_index(drop=True), tr_g.reset_index(drop=True)], axis=1)
te = pd.concat([te_v1.reset_index(drop=True), te_g.reset_index(drop=True)], axis=1)
DROP = ["class", "ID", "x", "y"]
FEATS = [c for c in tr.columns if c not in DROP]
print(f"features: {len(FEATS)} ({len(tr_v1.columns)-4} v1 + {len(tr_g.columns)} glcm)")

classes = sorted(tr["class"].unique())
c2i = {c: i for i, c in enumerate(classes)}; i2c = {i: c for c, i in c2i.items()}
y = np.array([c2i[c] for c in tr["class"].values])
NC = len(classes)

X = tr[FEATS].values.astype(np.float32)
Xte = te[FEATS].values.astype(np.float32)

params = dict(
    objective="multiclass", num_class=NC, learning_rate=0.05,
    num_leaves=63, min_data_in_leaf=20, feature_fraction=0.9,
    bagging_fraction=0.9, bagging_freq=5, lambda_l2=1.0,
    metric="multi_logloss", verbose=-1, class_weight="balanced",
)

SEEDS = [42, 7, 123, 2024, 31]
oof = np.zeros((len(X), NC))
pte = np.zeros((len(Xte), NC))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_i, va_i in skf.split(X, y):
        p = dict(params, seed=seed)
        dtr = lgb.Dataset(X[tr_i], y[tr_i])
        dva = lgb.Dataset(X[va_i], y[va_i], reference=dtr)
        m = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof[va_i] += m.predict(X[va_i], num_iteration=m.best_iteration) / len(SEEDS)
        pte += m.predict(Xte, num_iteration=m.best_iteration) / (5 * len(SEEDS))

oof_pred = oof.argmax(1)
print(f"\nOOF macroF1: {f1_score(y, oof_pred, average='macro'):.4f}")
print(f"OOF weightedF1: {f1_score(y, oof_pred, average='weighted'):.4f}")
print(classification_report(y, oof_pred, target_names=classes, digits=4))

np.save(DATA / "oof_lgb_v1glcm.npy", oof)
np.save(DATA / "pte_lgb_v1glcm.npy", pte)

sub = pd.DataFrame({"ID": te["ID"].values, "class": [i2c[i] for i in pte.argmax(1)]})
sub.to_csv(DATA / "sub_lgb_v1glcm.csv", index=False)

# Compare to baseline and current 3-flip leader
base = pd.read_csv(DATA / "sub_lgb_baseline.csv").sort_values("ID").reset_index(drop=True)
flip3 = pd.read_csv(DATA / "sub_3flips_safe.csv").sort_values("ID").reset_index(drop=True)
s = sub.sort_values("ID").reset_index(drop=True)
diff_base = (s["class"] != base["class"]).sum()
diff_flip3 = (s["class"] != flip3["class"]).sum()
print(f"\nsub_lgb_v1glcm.csv dist: {s['class'].value_counts().to_dict()}")
print(f"  differs from baseline: {diff_base} points")
print(f"  differs from 3-flip: {diff_flip3} points")

# Show the diffs from 3-flip
import numpy as np
d = s[s["class"] != flip3["class"]].copy()
d["flip3"] = flip3.loc[d.index, "class"].values
print("\nDiffs vs 3-flip sub:")
print(d[["ID", "class", "flip3"]])