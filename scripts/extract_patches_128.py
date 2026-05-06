"""Extract 128x128 patches — 32m x 32m context vs current 16m."""
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

DATA = Path(__file__).parent
BATHY = DATA / "MBES" / "bathymetry.tif"
BACK = DATA / "MBES" / "backscatter.tif"
PATCH = 128
HALF = PATCH // 2


def load_raster(p):
    with rasterio.open(p) as ds:
        a = ds.read(1).astype(np.float32)
        nd = ds.nodata
        if nd is not None:
            a = np.where(a == nd, np.nan, a)
        return a, ds.transform


def extract_patches(points_df, bath, back, T):
    xs = points_df["x"].values.astype(np.float64)
    ys = points_df["y"].values.astype(np.float64)
    inv = ~T
    cols_f, rows_f = inv * (xs, ys)
    rows = np.round(rows_f).astype(int)
    cols = np.round(cols_f).astype(int)
    H, W = bath.shape
    N = len(points_df)
    patches = np.zeros((N, 2, PATCH, PATCH), dtype=np.float32)
    for i in range(N):
        r, c = rows[i], cols[i]
        r0, r1 = r - HALF, r + HALF
        c0, c1 = c - HALF, c + HALF
        pr0, pr1 = max(0, r0), min(H, r1)
        pc0, pc1 = max(0, c0), min(W, c1)
        bp = bath[pr0:pr1, pc0:pc1]
        sp = back[pr0:pr1, pc0:pc1]
        dr0, dr1 = pr0 - r0, pr1 - r0
        dc0, dc1 = pc0 - c0, pc1 - c0
        patches[i, 0, dr0:dr1, dc0:dc1] = bp
        patches[i, 1, dr0:dr1, dc0:dc1] = sp
    return patches


bath, T = load_raster(BATHY)
back, _ = load_raster(BACK)
print(f"rasters {bath.shape}")

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
print(f"extracting train {len(train)}...")
pt_tr = extract_patches(train, bath, back, T)
print(f"extracting test {len(test)}...")
pt_te = extract_patches(test, bath, back, T)
print(f"train {pt_tr.shape}, test {pt_te.shape}")


def stats(a, ch):
    v = a[:, ch]
    v = v[~np.isnan(v)]
    return v.mean(), v.std() + 1e-6


mb, sb = stats(pt_tr, 0)
ms, ss = stats(pt_tr, 1)
print(f"bath mean={mb:.3f} std={sb:.3f}  bs mean={ms:.3f} std={ss:.3f}")


def norm(a):
    a = a.copy()
    a[:, 0] = (a[:, 0] - mb) / sb
    a[:, 1] = (a[:, 1] - ms) / ss
    return np.nan_to_num(a, nan=0.0)


pt_tr = norm(pt_tr)
pt_te = norm(pt_te)
np.save(DATA / "patches_train_128.npy", pt_tr)
np.save(DATA / "patches_test_128.npy", pt_te)
print(f"saved: train {pt_tr.nbytes//1024//1024} MB, test {pt_te.nbytes//1024//1024} MB")