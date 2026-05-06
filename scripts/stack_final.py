"""Stack GBDT ensemble + CNN predictions. Grid search weights on OOF macro-F1."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

DATA = Path(__file__).parent
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
classes = sorted(train["class"].unique())
c2i = {c: i for i, c in enumerate(classes)}; i2c = {i: c for c, i in c2i.items()}
y = np.array([c2i[c] for c in train["class"].values])

oof_l = np.load(DATA / "oof_lgb_v1.npy")
oof_x = np.load(DATA / "oof_xgb_v1.npy")
oof_c = np.load(DATA / "oof_cat_v1.npy")
oof_e = np.load(DATA / "oof_et_v1.npy")
pte_l = np.load(DATA / "pte_lgb_v1.npy")
pte_x = np.load(DATA / "pte_xgb_v1.npy")
pte_c = np.load(DATA / "pte_cat_v1.npy")
pte_e = np.load(DATA / "pte_et_v1.npy")

oof_cnn = np.load(DATA / "oof_cnn.npy")
pte_cnn = np.load(DATA / "pte_cnn.npy")

oof_gbdt = (oof_l + oof_x + oof_c) / 3
pte_gbdt = (pte_l + pte_x + pte_c) / 3

print("Component OOF macro-F1:")
for n, o in [("LGB", oof_l), ("XGB", oof_x), ("CAT", oof_c), ("ET", oof_e),
             ("GBDT", oof_gbdt), ("CNN", oof_cnn)]:
    print(f"  {n:6s}: {f1_score(y, o.argmax(1), average='macro'):.4f}")

best = (0, None)
print("\nWeight grid search GBDT+CNN:")
for w in np.linspace(0, 1, 21):
    o = w * oof_gbdt + (1 - w) * oof_cnn
    f = f1_score(y, o.argmax(1), average="macro")
    if f > best[0]:
        best = (f, w)
    print(f"  w_gbdt={w:.2f}  macroF1={f:.4f}")
print(f"\nBest: w_gbdt={best[1]:.2f} macroF1={best[0]:.4f}")

w = best[1]
final_oof = w * oof_gbdt + (1 - w) * oof_cnn
final_pte = w * pte_gbdt + (1 - w) * pte_cnn

pred = final_pte.argmax(1)
baseline = pd.read_csv(DATA / "sub_lgb_baseline.csv")
sub = pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in pred]})
diffs = (sub["class"].values != baseline.sort_values("ID")["class"].values).sum()
print(f"\nFinal sub differs from baseline on {diffs} points")
print(sub["class"].value_counts().to_dict())
sub.to_csv(DATA / "sub_stack_final.csv", index=False)

# Also save LGB+CNN 50/50 and LGB alone for comparison
for tag, p in [("lgb_cnn_50", 0.5 * pte_l + 0.5 * pte_cnn),
               ("gbdt_cnn_50", 0.5 * pte_gbdt + 0.5 * pte_cnn),
               ("gbdt_cnn_70", 0.7 * pte_gbdt + 0.3 * pte_cnn),
               ("cnn_only", pte_cnn)]:
    pp = p.argmax(1)
    s = pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in pp]})
    d = (s["class"].values != baseline.sort_values("ID")["class"].values).sum()
    s.to_csv(DATA / f"sub_{tag}.csv", index=False)
    print(f"  sub_{tag}: {d} diffs  dist={s['class'].value_counts().to_dict()}")