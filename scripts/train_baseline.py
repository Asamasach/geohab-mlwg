"""Baseline: LightGBM with stratified 5-fold CV, macro-F1, class weights."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import lightgbm as lgb

DATA = Path(__file__).parent
SEED = 42
N_FOLDS = 5

train = pd.read_csv(DATA / "train_features.csv")
test = pd.read_csv(DATA / "test_features.csv")

LABEL = "class"
DROP = ["class", "ID", "x", "y"]
FEATS = [c for c in train.columns if c not in DROP]
print(f"{len(FEATS)} features")

y_str = train[LABEL].values
classes = sorted(np.unique(y_str))
c2i = {c: i for i, c in enumerate(classes)}
i2c = {i: c for c, i in c2i.items()}
y = np.array([c2i[c] for c in y_str])
print("classes:", classes)
print("counts:", np.bincount(y))

X = train[FEATS].values.astype(np.float32)
Xte = test[FEATS].values.astype(np.float32)

params = dict(
    objective="multiclass",
    num_class=len(classes),
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_data_in_leaf=20,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    lambda_l2=1.0,
    metric="multi_logloss",
    verbose=-1,
    seed=SEED,
    class_weight="balanced",
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros((len(X), len(classes)), dtype=np.float64)
pte = np.zeros((len(Xte), len(classes)), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    dtr = lgb.Dataset(Xtr, ytr)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=3000,
        valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof[va_idx] = model.predict(Xva, num_iteration=model.best_iteration)
    pte += model.predict(Xte, num_iteration=model.best_iteration) / N_FOLDS
    fold_pred = oof[va_idx].argmax(1)
    fold_f1 = f1_score(yva, fold_pred, average="macro")
    print(f"fold {fold}: best_iter={model.best_iteration} macroF1={fold_f1:.4f}")

oof_pred = oof.argmax(1)
cv_macro = f1_score(y, oof_pred, average="macro")
cv_weighted = f1_score(y, oof_pred, average="weighted")
cv_micro = f1_score(y, oof_pred, average="micro")
print(f"\nOOF macro-F1: {cv_macro:.4f}")
print(f"OOF weighted-F1: {cv_weighted:.4f}")
print(f"OOF micro-F1 (=accuracy): {cv_micro:.4f}")
print("\nClassification report (OOF):")
print(classification_report(y, oof_pred, target_names=classes, digits=4))
print("\nConfusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y, oof_pred)
print(pd.DataFrame(cm, index=classes, columns=classes))

# Save OOF + test probs for ensembling
np.save(DATA / "oof_lgb.npy", oof)
np.save(DATA / "pte_lgb.npy", pte)

# Submission
sub = pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in pte.argmax(1)]})
sub.to_csv(DATA / "sub_lgb_baseline.csv", index=False)
print(f"\nwrote sub_lgb_baseline.csv ({len(sub)} rows)")
print(sub["class"].value_counts())