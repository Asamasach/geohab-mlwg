"""Train LightGBM on v2 features with stratified 5-fold CV and a 10-seed bag."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import lightgbm as lgb

DATA = Path(__file__).parent

train = pd.read_csv(DATA / "train_features_v2.csv")
test = pd.read_csv(DATA / "test_features_v2.csv")

DROP = ["class", "ID"]
FEATS = [c for c in train.columns if c not in DROP]
print(f"{len(FEATS)} features")

y_str = train["class"].values
classes = sorted(np.unique(y_str))
c2i = {c: i for i, c in enumerate(classes)}
i2c = {i: c for c, i in c2i.items()}
y = np.array([c2i[c] for c in y_str])

X = train[FEATS].values.astype(np.float32)
Xte = test[FEATS].values.astype(np.float32)

params = dict(
    objective="multiclass",
    num_class=len(classes),
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_data_in_leaf=15,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=1.0,
    metric="multi_logloss",
    verbose=-1,
    class_weight="balanced",
)

SEEDS = [42, 7, 123, 2024, 31]
oof = np.zeros((len(X), len(classes)), dtype=np.float64)
pte = np.zeros((len(Xte), len(classes)), dtype=np.float64)

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        p = dict(params, seed=seed)
        dtr = lgb.Dataset(X[tr_idx], y[tr_idx])
        dva = lgb.Dataset(X[va_idx], y[va_idx], reference=dtr)
        model = lgb.train(
            p, dtr, num_boost_round=5000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )
        oof[va_idx] += model.predict(X[va_idx], num_iteration=model.best_iteration) / len(SEEDS)
        pte += model.predict(Xte, num_iteration=model.best_iteration) / (5 * len(SEEDS))
    # print CV for this seed
    pred = oof.argmax(1)
    print(f"seed {seed}: running OOF macroF1={f1_score(y, pred, average='macro'):.4f}")

oof_pred = oof.argmax(1)
print(f"\nFinal OOF macroF1: {f1_score(y, oof_pred, average='macro'):.4f}")
print(f"Final OOF weightedF1: {f1_score(y, oof_pred, average='weighted'):.4f}")
print(f"Final OOF accuracy: {f1_score(y, oof_pred, average='micro'):.4f}")
print("\nPer-class F1:")
print(classification_report(y, oof_pred, target_names=classes, digits=4))
print("Confusion:")
print(pd.DataFrame(confusion_matrix(y, oof_pred), index=classes, columns=classes))

np.save(DATA / "oof_lgb_v2.npy", oof)
np.save(DATA / "pte_lgb_v2.npy", pte)

sub = pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in pte.argmax(1)]})
sub.to_csv(DATA / "sub_lgb_v2.csv", index=False)
print(f"\nsub_lgb_v2.csv: {sub['class'].value_counts().to_dict()}")