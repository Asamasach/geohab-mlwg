"""Dual-scale CNN: process 64x64 AND 128x128 patches simultaneously.
The idea: 64x64 captures local texture, 128x128 captures habitat-patch context.
Merge features before classification. Should find different errors than single-scale."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import time

DATA = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}", flush=True)
if DEVICE.type == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

BATCH = 48
EPOCHS = 50
LR = 1.5e-3
WD = 1e-4
N_FOLDS = 5
SEEDS = [42, 7, 123]
MIXUP_ALPHA = 0.3

X64 = np.load(DATA / "patches_train.npy")
X128 = np.load(DATA / "patches_train_128.npy")
Xte64 = np.load(DATA / "patches_test.npy")
Xte128 = np.load(DATA / "patches_test_128.npy")
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
classes = sorted(train["class"].unique())
c2i = {c: i for i, c in enumerate(classes)}
i2c = {i: c for c, i in c2i.items()}
NC = len(classes)
y = np.array([c2i[c] for c in train["class"].values])
print(f"train 64:{X64.shape} 128:{X128.shape}  test 64:{Xte64.shape} 128:{Xte128.shape}")


class DualDataset(Dataset):
    def __init__(self, x64, x128, y=None, augment=False):
        self.x64 = x64; self.x128 = x128; self.y = y; self.augment = augment

    def __len__(self): return len(self.x64)

    def __getitem__(self, i):
        a = self.x64[i].copy()
        b = self.x128[i].copy()
        if self.augment:
            k = np.random.randint(4)
            if k:
                a = np.rot90(a, k, axes=(1, 2)).copy()
                b = np.rot90(b, k, axes=(1, 2)).copy()
            if np.random.rand() < 0.5:
                a = a[:, :, ::-1].copy(); b = b[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                a = a[:, ::-1, :].copy(); b = b[:, ::-1, :].copy()
            if np.random.rand() < 0.4:
                n = np.random.randn() * 0.06
                a = a + n; b = b + n
        a = torch.from_numpy(np.ascontiguousarray(a))
        b = torch.from_numpy(np.ascontiguousarray(b))
        if self.y is not None:
            return a, b, int(self.y[i])
        return a, b


class Branch(nn.Module):
    def __init__(self, in_ch, dims, input_size):
        super().__init__()
        layers = []
        ci = in_ch
        for co in dims:
            layers += [
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            ci = co
        self.net = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.gap(self.net(x)).flatten(1)


class DualCNN(nn.Module):
    def __init__(self, nc=5):
        super().__init__()
        self.branch64 = Branch(2, [32, 64, 128, 192], 64)   # 64->4
        self.branch128 = Branch(2, [32, 64, 128, 192, 256], 128)  # 128->4
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(192 + 256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, nc),
        )

    def forward(self, x64, x128):
        f64 = self.branch64(x64)
        f128 = self.branch128(x128)
        return self.fc(torch.cat([f64, f128], dim=1))


counts = np.bincount(y)
w_class = torch.tensor(counts.sum() / (NC * counts), dtype=torch.float32).to(DEVICE)


def tta8(model, a, b):
    model.eval()
    out = torch.zeros(a.shape[0], NC, device=DEVICE)
    with torch.no_grad():
        for k in range(4):
            ar = torch.rot90(a, k, dims=(2, 3))
            br = torch.rot90(b, k, dims=(2, 3))
            out += F.softmax(model(ar, br), dim=1)
            out += F.softmax(model(torch.flip(ar, dims=(3,)), torch.flip(br, dims=(3,))), dim=1)
    return (out / 8).cpu().numpy()


oof = np.zeros((len(X64), NC), dtype=np.float32)
pte = np.zeros((len(Xte64), NC), dtype=np.float32)

for seed in SEEDS:
    print(f"\n==== SEED {seed} ====", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_i, va_i) in enumerate(skf.split(X64, y)):
        print(f"--- fold {fold} ---", flush=True)
        tr_ds = DualDataset(X64[tr_i], X128[tr_i], y[tr_i], augment=True)
        va_ds = DualDataset(X64[va_i], X128[va_i], y[va_i], augment=False)
        te_ds = DualDataset(Xte64, Xte128, augment=False)
        tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
        te_dl = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

        model = DualCNN(NC).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.CrossEntropyLoss(weight=w_class, label_smoothing=0.05)

        best_f1 = 0.0; best_state = None; t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            for a, b, yb in tr_dl:
                a = a.to(DEVICE, non_blocking=True)
                b = b.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                if MIXUP_ALPHA > 0 and np.random.rand() < 0.5:
                    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                    idx = torch.randperm(a.size(0), device=DEVICE)
                    a2 = lam * a + (1 - lam) * a[idx]
                    b2 = lam * b + (1 - lam) * b[idx]
                    logits = model(a2, b2)
                    loss = lam * loss_fn(logits, yb) + (1 - lam) * loss_fn(logits, yb[idx])
                else:
                    loss = loss_fn(model(a, b), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
            model.eval()
            all_p = []
            with torch.no_grad():
                for a, b, yb in va_dl:
                    all_p.append(F.softmax(model(a.to(DEVICE), b.to(DEVICE)), dim=1).cpu().numpy())
            all_p = np.vstack(all_p)
            vf1 = f1_score(y[va_i], all_p.argmax(1), average="macro")
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 5 == 0 or ep == EPOCHS - 1:
                print(f"  ep{ep:02d}: valF1={vf1:.4f} best={best_f1:.4f} t={time.time()-t0:.0f}s", flush=True)
        model.load_state_dict(best_state)
        vb = []
        for a, b, _ in va_dl:
            vb.append(tta8(model, a.to(DEVICE), b.to(DEVICE)))
        oof[va_i] += np.vstack(vb) / len(SEEDS)
        tb = []
        for a, b in te_dl:
            tb.append(tta8(model, a.to(DEVICE), b.to(DEVICE)))
        pte += np.vstack(tb) / (N_FOLDS * len(SEEDS))
        print(f"  fold {fold} seed {seed} best={best_f1:.4f} t={time.time()-t0:.0f}s", flush=True)

print(f"\nDual-CNN OOF macroF1: {f1_score(y, oof.argmax(1), average='macro'):.4f}")
print(classification_report(y, oof.argmax(1), target_names=classes, digits=4))
np.save(DATA / "oof_dual.npy", oof)
np.save(DATA / "pte_dual.npy", pte)
sub = pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in pte.argmax(1)]})
sub.to_csv(DATA / "sub_dual.csv", index=False)
print(f"\nsub_dual.csv dist: {sub['class'].value_counts().to_dict()}")