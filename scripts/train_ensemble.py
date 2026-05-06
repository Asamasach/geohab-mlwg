"""Ensemble: LightGBM + XGBoost + CatBoost + ExtraTrees on v2 features.
Each with 5-fold CV x 3 seeds. Average probabilities. Pick argmax."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

DATA = Path(__file__).parent
train = pd.read_csv(DATA / "train_features_v2.csv")
test = pd.read_csv(DATA / "test_features_v2.csv")
DROP = ["class", "ID"]
FEATS = [c for c in train.columns if c not in DROP]
y_str = train["class"].values
classes = sorted(np.unique(y_str))
c2i = {c: i for i, c in enumerate(classes)}; i2c = {i: c for c, i in c2i.items()}
y = np.array([c2i[c] for c in y_str])
X = train[FEATS].values.astype(np.float32)
Xte = test[FEATS].values.astype(np.float32)
# For sklearn models, impute nans
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X)
Xte_imp = imp.transform(Xte)

NC = len(classes)
SEEDS = [42, 7, 123]
N_FOLDS = 5


def run_lgb():
    oof = np.zeros((len(X), NC)); pte = np.zeros((len(Xte), NC))
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(X, y):
            p = dict(objective="multiclass", num_class=NC, learning_rate=0.03,
                     num_leaves=63, min_data_in_leaf=15, feature_fraction=0.8,
                     bagging_fraction=0.8, bagging_freq=5, lambda_l1=0.1, lambda_l2=1.0,
                     metric="multi_logloss", verbose=-1, class_weight="balanced", seed=seed)
            dtr = lgb.Dataset(X[tr_i], y[tr_i]); dva = lgb.Dataset(X[va_i], y[va_i], reference=dtr)
            m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
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
                n_estimators=3000, learning_rate=0.03, max_depth=8, min_child_weight=2,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1,
                objective="multi:softprob", num_class=NC, eval_metric="mlogloss",
                tree_method="hist", early_stopping_rounds=150, random_state=seed, n_jobs=-1,
            )
            m.fit(X[tr_i], y[tr_i], sample_weight=w[tr_i],
                  eval_set=[(X[va_i], y[va_i])], verbose=False)
            oof[va_i] += m.predict_proba(X[va_i]) / len(SEEDS)
            pte += m.predict_proba(Xte) / (N_FOLDS * len(SEEDS))
    return oof, pte


def run_cat():
    oof = np.zeros((len(X), NC)); pte = np.zeros((len(Xte), NC))
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(X, y):
            m = CatBoostClassifier(
                iterations=3000, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                loss_function="MultiClass", eval_metric="TotalF1", bootstrap_type="Bernoulli",
                subsample=0.8, auto_class_weights="Balanced", random_seed=seed, verbose=0,
                early_stopping_rounds=150, allow_writing_files=False,
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


def report(name, oof, y):
    pred = oof.argmax(1)
    print(f"{name}: macroF1={f1_score(y, pred, average='macro'):.4f} "
          f"weightedF1={f1_score(y, pred, average='weighted'):.4f}")


print("LGB..."); oof_l, pte_l = run_lgb(); report("LGB", oof_l, y)
print("XGB..."); oof_x, pte_x = run_xgb(); report("XGB", oof_x, y)
print("CAT..."); oof_c, pte_c = run_cat(); report("CAT", oof_c, y)
print("ET..."); oof_e, pte_e = run_et(); report("ET", oof_e, y)

np.save(DATA / "oof_lgb_v2.npy", oof_l); np.save(DATA / "pte_lgb_v2.npy", pte_l)
np.save(DATA / "oof_xgb_v2.npy", oof_x); np.save(DATA / "pte_xgb_v2.npy", pte_x)
np.save(DATA / "oof_cat_v2.npy", oof_c); np.save(DATA / "pte_cat_v2.npy", pte_c)
np.save(DATA / "oof_et_v2.npy", oof_e); np.save(DATA / "pte_et_v2.npy", pte_e)

# Simple average ensemble
ens_oof = (oof_l + oof_x + oof_c + oof_e) / 4
ens_pte = (pte_l + pte_x + pte_c + pte_e) / 4
report("ENS(LGB+XGB+CAT+ET)", ens_oof, y)

# GBDT-only ensemble (usually better)
gbdt_oof = (oof_l + oof_x + oof_c) / 3
gbdt_pte = (pte_l + pte_x + pte_c) / 3
report("ENS_GBDT(LGB+XGB+CAT)", gbdt_oof, y)

print(classification_report(y, gbdt_pte.argmax(0) if False else gbdt_oof.argmax(1),
                             target_names=classes, digits=4))

# Final submission from GBDT ensemble
sub = pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in gbdt_pte.argmax(1)]})
sub.to_csv(DATA / "sub_ens_gbdt.csv", index=False)
print(sub["class"].value_counts())

sub2 = pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in ens_pte.argmax(1)]})
sub2.to_csv(DATA / "sub_ens_all.csv", index=False)