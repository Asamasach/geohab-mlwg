"""Small CNN on 2x64x64 patches, 5-fold stratified CV, class-balanced CE loss."""
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
import sys
sys.stdout.reconfigure(line_buffering=True)
DEVICE = "cpu"
torch.set_num_threads(8)
SEED = 42
BATCH = 128
EPOCHS = 20
LR = 3e-3
WD = 1e-4
N_FOLDS = 5

torch.manual_seed(SEED)
np.random.seed(SEED)

X = np.load(DATA / "patches_train.npy")  # (6256, 2, 64, 64)
Xte = np.load(DATA / "patches_test.npy")  # (98, 2, 64, 64)
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

classes = sorted(train["class"].unique())
c2i = {c: i for i, c in enumerate(classes)}; i2c = {i: c for c, i in c2i.items()}
NC = len(classes)
y = np.array([c2i[c] for c in train["class"].values])
print(f"train {X.shape}  test {Xte.shape}  classes={classes}")


class PatchDataset(Dataset):
    def __init__(self, X, y=None, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            # Random 90° rotations + flips (dihedral-8 symmetry)
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k, axes=(1, 2)).copy()
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :].copy()
        x = torch.from_numpy(np.ascontiguousarray(x))
        if self.y is not None:
            return x, int(self.y[i])
        return x


class SmallCNN(nn.Module):
    def __init__(self, nc=5):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.net = nn.Sequential(
            blk(2, 24),   # 64 -> 32
            blk(24, 48),  # 32 -> 16
            blk(48, 96),  # 16 -> 8
            blk(96, 128), # 8 -> 4
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, nc),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# Class weights (inverse frequency)
counts = np.bincount(y)
w_class = torch.tensor(counts.sum() / (NC * counts), dtype=torch.float32).to(DEVICE)
print("class weights:", w_class.tolist())


def tta_predict(model, x):
    model.eval()
    logits = torch.zeros(x.shape[0], NC, device=DEVICE)
    with torch.no_grad():
        for k in range(4):
            xr = torch.rot90(x, k, dims=(2, 3))
            logits += F.softmax(model(xr), dim=1)
            logits += F.softmax(model(torch.flip(xr, dims=(3,))), dim=1)
    return (logits / 8).cpu().numpy()


oof = np.zeros((len(X), NC), dtype=np.float32)
pte = np.zeros((len(Xte), NC), dtype=np.float32)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (tr_i, va_i) in enumerate(skf.split(X, y)):
    print(f"\n--- fold {fold} ---")
    tr_ds = PatchDataset(X[tr_i], y[tr_i], augment=True)
    va_ds = PatchDataset(X[va_i], y[va_i], augment=False)
    te_ds = PatchDataset(Xte, None, augment=False)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = SmallCNN(NC).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss(weight=w_class, label_smoothing=0.05)

    best_f1 = 0.0
    best_state = None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        sch.step()
        # Validate
        model.eval()
        all_p = []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(DEVICE)
                all_p.append(F.softmax(model(xb), dim=1).cpu().numpy())
        all_p = np.vstack(all_p)
        vf1 = f1_score(y[va_i], all_p.argmax(1), average="macro")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  ep{ep:02d}: valF1={vf1:.4f} best={best_f1:.4f} t={time.time()-t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    # Final OOF and test with TTA
    all_p = []
    for xb, _ in va_dl:
        all_p.append(tta_predict(model, xb.to(DEVICE)))
    oof[va_i] = np.vstack(all_p)
    all_t = []
    for xb in te_dl:
        all_t.append(tta_predict(model, xb.to(DEVICE)))
    pte += np.vstack(all_t) / N_FOLDS
    print(f"  fold {fold} best macroF1={best_f1:.4f}")

oof_pred = oof.argmax(1)
print(f"\nCNN OOF macroF1: {f1_score(y, oof_pred, average='macro'):.4f}")
print(f"CNN OOF weightedF1: {f1_score(y, oof_pred, average='weighted'):.4f}")
print(classification_report(y, oof_pred, target_names=classes, digits=4))

np.save(DATA / "oof_cnn.npy", oof)
np.save(DATA / "pte_cnn.npy", pte)
sub = pd.DataFrame({"ID": test["ID"].values, "class": [i2c[i] for i in pte.argmax(1)]})
sub.to_csv(DATA / "sub_cnn.csv", index=False)
print(f"\nsub_cnn.csv dist: {sub['class'].value_counts().to_dict()}")