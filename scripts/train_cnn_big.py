"""Bigger GPU CNN v3: ResNet-like + larger patches + more augmentation + 5-seed bag."""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import numpy as np
import pandas as pd
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

BATCH = 96
EPOCHS = 60
LR = 1.5e-3
WD = 1e-4
N_FOLDS = 5
SEEDS = [42, 7, 123, 2024, 31]
MIXUP_ALPHA = 0.3

X = np.load(DATA / "patches_train.npy")
Xte = np.load(DATA / "patches_test.npy")
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
classes = sorted(train["class"].unique())
c2i = {c: i for i, c in enumerate(classes)}
i2c = {i: c for c, i in c2i.items()}
NC = len(classes)
y = np.array([c2i[c] for c in train["class"].values])
print(f"train {X.shape}  test {Xte.shape}")


class PatchDataset(Dataset):
    def __init__(self, X, y=None, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.augment:
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k, axes=(1, 2)).copy()
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :].copy()
            if np.random.rand() < 0.5:
                sh, sw = np.random.randint(-6, 7, size=2)
                x = np.roll(x, (sh, sw), axis=(1, 2))
            if np.random.rand() < 0.4:
                x = x + np.random.randn(*x.shape).astype(np.float32) * 0.08
            # Channel-wise scale (backscatter dB noise)
            if np.random.rand() < 0.3:
                x = x * (0.9 + 0.2 * np.random.rand())
        x = torch.from_numpy(np.ascontiguousarray(x))
        if self.y is not None:
            return x, int(self.y[i])
        return x


class BasicBlock(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(co)
        self.conv2 = nn.Conv2d(co, co, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(co)
        self.short = None
        if stride != 1 or ci != co:
            self.short = nn.Sequential(
                nn.Conv2d(ci, co, 1, stride=stride, bias=False),
                nn.BatchNorm2d(co),
            )

    def forward(self, x):
        r = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.short is not None:
            r = self.short(r)
        return F.relu(out + r, inplace=True)


class SmallResNet(nn.Module):
    def __init__(self, nc=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 384, 2), BasicBlock(384, 384))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(384 * 2, nc)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        a = self.gap(x).flatten(1)
        m = self.gmp(x).flatten(1)
        return self.fc(self.drop(torch.cat([a, m], dim=1)))


counts = np.bincount(y)
w_class = torch.tensor(counts.sum() / (NC * counts), dtype=torch.float32).to(DEVICE)
print(f"class weights: {w_class.cpu().tolist()}")


def tta16(model, x):
    model.eval()
    out = torch.zeros(x.shape[0], NC, device=DEVICE)
    with torch.no_grad():
        for k in range(4):
            xr = torch.rot90(x, k, dims=(2, 3))
            out += F.softmax(model(xr), dim=1)
            out += F.softmax(model(torch.flip(xr, dims=(3,))), dim=1)
            out += F.softmax(model(torch.flip(xr, dims=(2,))), dim=1)
            out += F.softmax(model(torch.flip(torch.flip(xr, dims=(2,)), dims=(3,))), dim=1)
    return (out / 16).cpu().numpy()


oof = np.zeros((len(X), NC), dtype=np.float32)
pte = np.zeros((len(Xte), NC), dtype=np.float32)

for seed in SEEDS:
    print(f"\n==== SEED {seed} ====", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_i, va_i) in enumerate(skf.split(X, y)):
        print(f"--- fold {fold} ---", flush=True)
        tr_ds = PatchDataset(X[tr_i], y[tr_i], augment=True)
        va_ds = PatchDataset(X[va_i], y[va_i], augment=False)
        te_ds = PatchDataset(Xte, None, augment=False)
        tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
        te_dl = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

        model = SmallResNet(NC).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.CrossEntropyLoss(weight=w_class, label_smoothing=0.05)

        best_f1 = 0.0
        best_state = None
        t0 = time.time()
        for ep in range(EPOCHS):
            model.train()
            for xb, yb in tr_dl:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                if MIXUP_ALPHA > 0 and np.random.rand() < 0.5:
                    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                    idx = torch.randperm(xb.size(0), device=DEVICE)
                    xb2 = lam * xb + (1 - lam) * xb[idx]
                    logits = model(xb2)
                    loss = lam * loss_fn(logits, yb) + (1 - lam) * loss_fn(logits, yb[idx])
                else:
                    loss = loss_fn(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sch.step()

            model.eval()
            all_p = []
            with torch.no_grad():
                for xb, yb in va_dl:
                    all_p.append(F.softmax(model(xb.to(DEVICE, non_blocking=True)), dim=1).cpu().numpy())
            all_p = np.vstack(all_p)
            vf1 = f1_score(y[va_i], all_p.argmax(1), average="macro")
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 5 == 0 or ep == EPOCHS - 1:
                print(f"  ep{ep:02d}: valF1={vf1:.4f} best={best_f1:.4f} t={time.time()-t0:.0f}s", flush=True)

        model.load_state_dict(best_state)
        vb = []
        for xb, _ in va_dl:
            vb.append(tta16(model, xb.to(DEVICE, non_blocking=True)))
        oof[va_i] += np.vstack(vb) / len(SEEDS)
        tb = []
        for xb in te_dl:
            tb.append(tta16(model, xb.to(DEVICE, non_blocking=True)))
        pte += np.vstack(tb) / (N_FOLDS * len(SEEDS))
        print(f"  fold {fold} seed {seed} best={best_f1:.4f} t={time.time()-t0:.0f}s", flush=True)

print(f"\nOOF macroF1: {f1_score(y, oof.argmax(1), average='macro'):.4f}")
print(f"OOF weightedF1: {f1_score(y, oof.argmax(1), average='weighted'):.4f}")
print(classification_report(y, oof.argmax(1), target_names=classes, digits=4))

np.save(DATA / "oof_cnn_big.npy", oof)
np.save(DATA / "pte_cnn_big.npy", pte)
sub = pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in pte.argmax(1)]})
sub.to_csv(DATA / "sub_cnn_big.csv", index=False)
print(f"\nsub_cnn_big.csv dist: {sub['class'].value_counts().to_dict()}")