"""Ensemble on v1 features (which proved strongest on the public LB).
Multi-model x multi-seed bagging, averaged predictions."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

DATA = Path(__file__).parent
train = pd.read_csv(DATA / "train_features.csv")
test = pd.read_csv(DATA / "test_features.csv")
DROP = ["class", "ID", "x", "y"]
FEATS = [c for c in train.columns if c not in DROP]
print(f"v1 features: {len(FEATS)}")

y_str = train["class"].values
classes = sorted(np.unique(y_str))
c2i = {c: i for i, c in enumerate(classes)}; i2c = {i: c for c, i in c2i.items()}
y = np.array([c2i[c] for c in y_str])
X = train[FEATS].values.astype(np.float32)
Xte = test[FEATS].values.astype(np.float32)
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X); Xte_imp = imp.transform(Xte)

NC = len(classes)
SEEDS = [42, 7, 123, 2024, 31]
N_FOLDS = 5


def run_lgb():
    oof = np.zeros((len(X), NC)); pte = np.zeros((len(Xte), NC))
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(X, y):
            p = dict(objective="multiclass", num_class=NC, learning_rate=0.05,
                     num_leaves=63, min_data_in_leaf=20, feature_fraction=0.9,
                     bagging_fraction=0.9, bagging_freq=5, lambda_l2=1.0,
                     metric="multi_logloss", verbose=-1, class_weight="balanced", seed=seed)
            dtr = lgb.Dataset(X[tr_i], y[tr_i]); dva = lgb.Dataset(X[va_i], y[va_i], reference=dtr)
            m = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            oof[va_i] += m.predict(X[va_i], num_iteration=m.best_iteration) / len(SEEDS)
            pte += m.predict(Xte, num_iteration=m.best_iteration) / (N_FOLDS * len(SEEDS))
    return oof, pte


def run_xgb():
    oof = np.zeros((len(X), NC)); pte = np.zeros((len(Xte), NC))
    cw = np.bincount(y); w = (cw.sum() / (NC * cw))[y]
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(X, y):
            m = xgb.XGBClassifier(
                n_estimators=2000, learning_rate=0.05, max_depth=7, min_child_weight=3,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective="multi:softprob", num_class=NC, eval_metric="mlogloss",
                tree_method="hist", early_stopping_rounds=100, random_state=seed, n_jobs=-1,
            )
            m.fit(X[tr_i], y[tr_i], sample_weight=w[tr_i], eval_set=[(X[va_i], y[va_i])], verbose=False)
            oof[va_i] += m.predict_proba(X[va_i]) / len(SEEDS)
            pte += m.predict_proba(Xte) / (N_FOLDS * len(SEEDS))
    return oof, pte


def run_cat():
    oof = np.zeros((len(X), NC)); pte = np.zeros((len(Xte), NC))
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(X, y):
            m = CatBoostClassifier(
                iterations=2000, learning_rate=0.05, depth=7, l2_leaf_reg=3.0,
                loss_function="MultiClass", auto_class_weights="Balanced",
                random_seed=seed, verbose=0, early_stopping_rounds=100, allow_writing_files=False,
            )
            m.fit(X[tr_i], y[tr_i], eval_set=(X[va_i], y[va_i]), verbose=False)
            oof[va_i] += m.predict_proba(X[va_i]) / len(SEEDS)
            pte += m.predict_proba(Xte) / (N_FOLDS * len(SEEDS))
    return oof, pte


def run_et():
    oof = np.zeros((len(X), NC)); pte = np.zeros((len(Xte), NC))
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(X_imp, y):
            m = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2,
                                      class_weight="balanced", n_jobs=-1, random_state=seed)
            m.fit(X_imp[tr_i], y[tr_i])
            oof[va_i] += m.predict_proba(X_imp[va_i]) / len(SEEDS)
            pte += m.predict_proba(Xte_imp) / (N_FOLDS * len(SEEDS))
    return oof, pte


def rep(n, o):
    p = o.argmax(1)
    print(f"{n}: macroF1={f1_score(y,p,average='macro'):.4f} weightedF1={f1_score(y,p,average='weighted'):.4f}")


print("LGB v1..."); oof_l, pte_l = run_lgb(); rep("LGB", oof_l)
print("XGB v1..."); oof_x, pte_x = run_xgb(); rep("XGB", oof_x)
print("CAT v1..."); oof_c, pte_c = run_cat(); rep("CAT", oof_c)
print("ET v1..."); oof_e, pte_e = run_et(); rep("ET", oof_e)

np.save(DATA / "oof_lgb_v1.npy", oof_l); np.save(DATA / "pte_lgb_v1.npy", pte_l)
np.save(DATA / "oof_xgb_v1.npy", oof_x); np.save(DATA / "pte_xgb_v1.npy", pte_x)
np.save(DATA / "oof_cat_v1.npy", oof_c); np.save(DATA / "pte_cat_v1.npy", pte_c)
np.save(DATA / "oof_et_v1.npy", oof_e); np.save(DATA / "pte_et_v1.npy", pte_e)

ens4 = (oof_l + oof_x + oof_c + oof_e) / 4
ens4_te = (pte_l + pte_x + pte_c + pte_e) / 4
rep("ENS4", ens4)

ens3 = (oof_l + oof_x + oof_c) / 3
ens3_te = (pte_l + pte_x + pte_c) / 3
rep("ENS3(GBDT)", ens3)

print("\nPer-class ENS3:")
print(classification_report(y, ens3.argmax(1), target_names=classes, digits=4))

pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in ens3_te.argmax(1)]}).to_csv(DATA / "sub_ens3_v1.csv", index=False)
pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in ens4_te.argmax(1)]}).to_csv(DATA / "sub_ens4_v1.csv", index=False)
pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in pte_l.argmax(1)]}).to_csv(DATA / "sub_lgb_v1_bag.csv", index=False)
print("\nwrote sub_ens3_v1.csv, sub_ens4_v1.csv, sub_lgb_v1_bag.csv")