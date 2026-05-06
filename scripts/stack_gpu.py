"""Stack GBDT ensemble + GPU CNN. Grid-search weights, analyze flips vs current leader."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from scipy.spatial import cKDTree

DATA = Path(__file__).parent
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
classes = sorted(train["class"].unique())
c2i = {c: i for i, c in enumerate(classes)}
i2c = {i: c for c, i in c2i.items()}
y = np.array([c2i[c] for c in train["class"].values])

# Load all OOFs
oof_l = np.load(DATA / "oof_lgb_v1.npy")
oof_x = np.load(DATA / "oof_xgb_v1.npy")
oof_c = np.load(DATA / "oof_cat_v1.npy")
oof_gbdt = (oof_l + oof_x + oof_c) / 3
oof_cnn_cpu = np.load(DATA / "oof_cnn.npy")
oof_cnn_gpu = np.load(DATA / "oof_cnn_gpu.npy")

pte_l = np.load(DATA / "pte_lgb_v1.npy")
pte_x = np.load(DATA / "pte_xgb_v1.npy")
pte_c = np.load(DATA / "pte_cat_v1.npy")
pte_gbdt = (pte_l + pte_x + pte_c) / 3
pte_cnn_cpu = np.load(DATA / "pte_cnn.npy")
pte_cnn_gpu = np.load(DATA / "pte_cnn_gpu.npy")

print("Component OOF macro-F1:")
for n, o in [("LGB", oof_l), ("XGB", oof_x), ("CAT", oof_c),
             ("GBDT", oof_gbdt), ("CNN_cpu", oof_cnn_cpu), ("CNN_gpu", oof_cnn_gpu)]:
    print(f"  {n:10s}: {f1_score(y, o.argmax(1), average='macro'):.4f}")

print("\nGrid search GBDT vs CNN_gpu:")
best = (0, 0)
for w in np.linspace(0, 1, 21):
    o = w * oof_gbdt + (1 - w) * oof_cnn_gpu
    f = f1_score(y, o.argmax(1), average="macro")
    if f > best[0]:
        best = (f, w)
    print(f"  w_gbdt={w:.2f}  oof_macroF1={f:.4f}")

print(f"\nBest: w_gbdt={best[1]:.2f} oof={best[0]:.4f}")

# Also triple ensemble
print("\nTriple grid search GBDT + CNN_gpu + CNN_cpu:")
best3 = (0, 0, 0)
for wg in np.linspace(0.3, 1.0, 8):
    for wgpu in np.linspace(0, 1 - wg, 6):
        wcpu = 1 - wg - wgpu
        if wcpu < 0:
            continue
        o = wg * oof_gbdt + wgpu * oof_cnn_gpu + wcpu * oof_cnn_cpu
        f = f1_score(y, o.argmax(1), average="macro")
        if f > best3[0]:
            best3 = (f, wg, wgpu)
print(f"Best triple: w_gbdt={best3[1]:.2f} w_cnngpu={best3[2]:.2f} w_cnncpu={1-best3[1]-best3[2]:.2f} oof={best3[0]:.4f}")

# Build several candidate subs
baseline = pd.read_csv(DATA / "sub_lgb_baseline.csv").sort_values("ID").reset_index(drop=True)
flip3 = pd.read_csv(DATA / "sub_3flips_safe.csv").sort_values("ID").reset_index(drop=True)


def save_sub(name, pte_arr):
    pred = pte_arr.argmax(1)
    sub = pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in pred]})
    sub = sub.sort_values("ID").reset_index(drop=True)
    dbase = (sub["class"] != baseline["class"]).sum()
    dflip3 = (sub["class"] != flip3["class"]).sum()
    sub.to_csv(DATA / f"sub_{name}.csv", index=False)
    dist = sub["class"].value_counts().to_dict()
    print(f"  sub_{name}.csv  diffs_vs_base={dbase}  diffs_vs_flip3={dflip3}  dist={dist}")
    return sub


print("\n=== Candidate test submissions ===")
save_sub("gpu_alone", pte_cnn_gpu)
save_sub("gbdt_cnngpu_50", 0.5 * pte_gbdt + 0.5 * pte_cnn_gpu)
save_sub("gbdt_cnngpu_70", 0.7 * pte_gbdt + 0.3 * pte_cnn_gpu)
save_sub("gbdt_cnngpu_80", 0.8 * pte_gbdt + 0.2 * pte_cnn_gpu)
save_sub("gbdt_cnngpu_best", best[1] * pte_gbdt + (1 - best[1]) * pte_cnn_gpu)
# Triple best
wg, wgpu = best3[1], best3[2]
wcpu = 1 - wg - wgpu
save_sub("triple_best", wg * pte_gbdt + wgpu * pte_cnn_gpu + wcpu * pte_cnn_cpu)

# Also: high-confidence voting — use CNN_gpu prediction only when CNN confidence is high
# AND GBDT is uncertain (margin<0.3). This is targeted flipping.
print("\n=== Targeted flip analysis: CNN overrides GBDT where CNN is confident AND GBDT weak ===")
gbdt_sort = np.sort(pte_gbdt, axis=1)
gbdt_margin = gbdt_sort[:, -1] - gbdt_sort[:, -2]
cnn_max = pte_cnn_gpu.max(axis=1)
gbdt_pred = pte_gbdt.argmax(1)
cnn_pred = pte_cnn_gpu.argmax(1)

# Spatial context for sanity
tr = pd.read_csv(DATA / "train.csv")
tree = cKDTree(tr[["x", "y"]].values)
tr_labels = tr["class"].values

candidates = []
for i in range(len(test)):
    if gbdt_pred[i] == cnn_pred[i]:
        continue
    if gbdt_margin[i] < 0.35 and cnn_max[i] > 0.55:
        x, yy = test["x"].iloc[i], test["y"].iloc[i]
        d, idx = tree.query([x, yy], k=10)
        nn = tr_labels[idx]
        nn_cnt = {c: int((nn == c).sum()) for c in classes if (nn == c).sum() > 0}
        candidates.append({
            "ID": int(test["ID"].iloc[i]),
            "gbdt": classes[gbdt_pred[i]],
            "gbdt_p": float(pte_gbdt[i, gbdt_pred[i]]),
            "cnn": classes[cnn_pred[i]],
            "cnn_p": float(cnn_max[i]),
            "nn10": nn_cnt,
            "d5m": float(d[4]),
        })
for c in sorted(candidates, key=lambda x: (-x["cnn_p"], x["gbdt_p"])):
    print(f"  ID {c['ID']:3d}: GBDT={c['gbdt']}({c['gbdt_p']:.2f}) CNN={c['cnn']}({c['cnn_p']:.2f}) NN={c['nn10']} d5={c['d5m']:.1f}m")

# Build targeted-flip sub: only flip IDs where CNN strongly disagrees AND spatial NN ALSO agrees with CNN
print("\n=== Spatial-confirmed CNN flip candidates ===")
spatial_confirmed = []
for c in candidates:
    if c["cnn"] in c["nn10"] and c["nn10"].get(c["cnn"], 0) >= 5:
        spatial_confirmed.append(c)
        print(f"  ID {c['ID']}: CNN={c['cnn']} with NN10 {c['nn10']}")

sub_targeted = baseline.copy()
for c in spatial_confirmed:
    sub_targeted.loc[sub_targeted["ID"] == c["ID"], "class"] = c["cnn"]
dbase = (sub_targeted["class"] != baseline["class"]).sum()
dflip3 = (sub_targeted["class"] != flip3["class"]).sum()
sub_targeted.to_csv(DATA / "sub_cnn_targeted.csv", index=False)
print(f"sub_cnn_targeted.csv: {dbase} flips from baseline, {dflip3} from current leader")