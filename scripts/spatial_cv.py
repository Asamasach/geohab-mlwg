"""Spatial cross-validation: cluster points by location and use GroupKFold.
This gives an honest estimate of how the model generalizes to spatially-unseen regions,
which is much closer to the actual test-set scenario than stratified-random CV."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb

DATA = Path(__file__).parent
train = pd.read_csv(DATA / "train_features.csv")
test = pd.read_csv(DATA / "test_features.csv")

DROP = ["class", "ID", "x", "y"]
FEATS = [c for c in train.columns if c not in DROP]
classes = sorted(train["class"].unique())
c2i = {c: i for i, c in enumerate(classes)}
y = np.array([c2i[c] for c in train["class"].values])
X = train[FEATS].values.astype(np.float32)
Xte = test[FEATS].values.astype(np.float32)
NC = len(classes)

xy = train[["x", "y"]].values

# Spatial clustering into 20 groups (balanced by distance)
for N_CLUSTERS in [10, 20, 40]:
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    groups = km.fit_predict(xy)

    # Within each cluster, how many classes are represented?
    cluster_class = pd.DataFrame({"cluster": groups, "class": train["class"]})
    cluster_counts = cluster_class.groupby("cluster")["class"].nunique()
    print(f"\n{N_CLUSTERS} clusters: class counts per cluster min={cluster_counts.min()} max={cluster_counts.max()} mean={cluster_counts.mean():.1f}")

    # GroupKFold 5-fold
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(X), NC))
    pte = np.zeros((len(Xte), NC))

    params = dict(
        objective="multiclass", num_class=NC, learning_rate=0.05,
        num_leaves=63, min_data_in_leaf=20, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, lambda_l2=1.0,
        metric="multi_logloss", verbose=-1, class_weight="balanced", seed=42,
    )
    for tr_i, va_i in gkf.split(X, y, groups):
        dtr = lgb.Dataset(X[tr_i], y[tr_i])
        dva = lgb.Dataset(X[va_i], y[va_i], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof[va_i] = m.predict(X[va_i], num_iteration=m.best_iteration)
        pte += m.predict(Xte, num_iteration=m.best_iteration) / 5
    f1m = f1_score(y, oof.argmax(1), average="macro")
    f1w = f1_score(y, oof.argmax(1), average="weighted")
    print(f"  Spatial-CV OOF: macroF1={f1m:.4f}  weightedF1={f1w:.4f}")
    # Per-class F1
    for c in classes:
        ci = c2i[c]
        mask = y == ci
        if mask.any():
            f = f1_score(y == ci, oof.argmax(1) == ci, average="binary")
            print(f"    {c}: {f:.4f}  (n={mask.sum()})")