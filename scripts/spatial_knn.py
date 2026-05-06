"""Spatial kNN predictor — since test points are within the same site and habitat is
spatially contiguous, a local majority vote is a strong baseline.
Also compute for each test point the NN-majority and see where it disagrees with GBDT."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from collections import Counter

DATA = Path(__file__).parent
tr = pd.read_csv(DATA / "train.csv")
te = pd.read_csv(DATA / "test.csv")

classes = sorted(tr["class"].unique())
c2i = {c: i for i, c in enumerate(classes)}; i2c = {i: c for c, i in c2i.items()}
NC = len(classes)
y_tr = np.array([c2i[c] for c in tr["class"].values])
xy_tr = tr[["x", "y"]].values
xy_te = te[["x", "y"]].values


def spatial_knn_predict(xy_train, y_train, xy_query, k, weight="uniform"):
    tree = cKDTree(xy_train)
    d, idx = tree.query(xy_query, k=k)
    if k == 1:
        idx = idx[:, None]; d = d[:, None]
    labels = y_train[idx]
    probs = np.zeros((len(xy_query), NC))
    if weight == "uniform":
        for i in range(len(xy_query)):
            for l in labels[i]:
                probs[i, l] += 1
        probs /= k
    else:  # inverse distance
        w = 1.0 / (d + 1e-3)
        for i in range(len(xy_query)):
            for j, l in enumerate(labels[i]):
                probs[i, l] += w[i, j]
            probs[i] /= probs[i].sum()
    return probs


# CV to find best k
print("Spatial kNN cross-validated macro-F1:")
for k in [1, 3, 5, 7, 10, 15, 25, 50]:
    oof = np.zeros((len(tr), NC))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tri, vai in skf.split(xy_tr, y_tr):
        p_u = spatial_knn_predict(xy_tr[tri], y_tr[tri], xy_tr[vai], k, "uniform")
        p_w = spatial_knn_predict(xy_tr[tri], y_tr[tri], xy_tr[vai], k, "inv")
        oof[vai] = (p_u + p_w) / 2
    pred = oof.argmax(1)
    f1 = f1_score(y_tr, pred, average="macro")
    print(f"  k={k:3d}: macroF1={f1:.4f}")

# Best k by CV — then predict for test
best_k = 10
p_u = spatial_knn_predict(xy_tr, y_tr, xy_te, best_k, "uniform")
p_w = spatial_knn_predict(xy_tr, y_tr, xy_te, best_k, "inv")
p_spatial = (p_u + p_w) / 2
pte_pred = p_spatial.argmax(1)

# Load GBDT ensemble test probs
pl = np.load(DATA / "pte_lgb_v1.npy")
px = np.load(DATA / "pte_xgb_v1.npy")
pc = np.load(DATA / "pte_cat_v1.npy")
p_gbdt = (pl + px + pc) / 3
p_gbdt_pred = p_gbdt.argmax(1)

# Diff between spatial kNN and GBDT
disagree = np.where(p_gbdt_pred != pte_pred)[0]
print(f"\nSpatial kNN vs GBDT disagree on {len(disagree)} test points:")
for i in disagree:
    ord_ = np.argsort(p_gbdt[i])[::-1]
    print(f"  ID {te['ID'].iloc[i]:3d} GBDT={classes[p_gbdt_pred[i]]} ({p_gbdt[i, p_gbdt_pred[i]]:.2f})  "
          f"  Spatial={classes[pte_pred[i]]} ({p_spatial[i, pte_pred[i]]:.2f})")

# Combine: average probabilities (50/50)
p_combo = 0.5 * p_gbdt + 0.5 * p_spatial
p_combo_pred = p_combo.argmax(1)
print(f"\nGBDT + Spatial 50/50: pred dist {Counter([classes[i] for i in p_combo_pred])}")
diff_combo = np.where(p_combo_pred != p_gbdt_pred)[0]
print(f"Combo differs from GBDT on {len(diff_combo)} points: IDs {list(te['ID'].iloc[diff_combo])}")

# Save all variants
pd.DataFrame({"ID": te["ID"].values, "class": [classes[i] for i in pte_pred]}).to_csv(DATA / "sub_spatial_knn.csv", index=False)
pd.DataFrame({"ID": te["ID"].values, "class": [classes[i] for i in p_combo_pred]}).to_csv(DATA / "sub_combo_50_50.csv", index=False)

# Also 70/30 and 30/70
for w in [0.3, 0.7]:
    pc_ = w * p_gbdt + (1 - w) * p_spatial
    pp = pc_.argmax(1)
    diff = (pp != p_gbdt_pred).sum()
    pd.DataFrame({"ID": te["ID"].values, "class": [classes[i] for i in pp]}).to_csv(
        DATA / f"sub_combo_{int(w*100)}_{int((1-w)*100)}.csv", index=False
    )
    print(f"w_gbdt={w}: differs from GBDT on {diff} points")

np.save(DATA / "pte_spatial.npy", p_spatial)