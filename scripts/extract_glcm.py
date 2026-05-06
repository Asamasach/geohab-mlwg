"""GLCM (gray-level co-occurrence matrix) texture features for backscatter.
Gold standard for seafloor habitat classification literature."""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import rasterio
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path
import time

DATA = Path(__file__).parent
BACK = DATA / "MBES" / "backscatter.tif"
BATHY = DATA / "MBES" / "bathymetry.tif"

# Use windows 16x16, 32x32 around each point
WINDOWS = [16, 32]
# 4 directions, distance=1
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
LEVELS = 32  # quantization levels


def load_raster(p):
    with rasterio.open(p) as ds:
        a = ds.read(1).astype(np.float32)
        nd = ds.nodata
        if nd is not None:
            a = np.where(a == nd, np.nan, a)
        return a, ds.transform


def quantize(a, lo, hi, levels):
    """Clip and quantize to uint8 in [0, levels-1]."""
    q = (a - lo) / (hi - lo)
    q = np.clip(q, 0, 1)
    q = (q * (levels - 1)).astype(np.uint8)
    return q


def compute_glcm_feats(points_df, raster, T, channel_name):
    xs = points_df["x"].values.astype(np.float64)
    ys = points_df["y"].values.astype(np.float64)
    inv = ~T
    cols_f, rows_f = inv * (xs, ys)
    rows = np.round(rows_f).astype(int)
    cols = np.round(cols_f).astype(int)
    H, W = raster.shape

    valid = raster[~np.isnan(raster)]
    lo = np.percentile(valid, 1)
    hi = np.percentile(valid, 99)
    print(f"  {channel_name} quantize range: [{lo:.2f}, {hi:.2f}]")
    q_full = quantize(np.nan_to_num(raster, nan=lo), lo, hi, LEVELS)

    N = len(points_df)
    feats = {}
    t0 = time.time()
    for win in WINDOWS:
        half = win // 2
        for prop in PROPS:
            feats[f"{channel_name}_glcm_{prop}_w{win}"] = np.full(N, np.nan, dtype=np.float32)
    for i in range(N):
        r, c = rows[i], cols[i]
        for win in WINDOWS:
            half = win // 2
            r0, r1 = max(0, r - half), min(H, r + half)
            c0, c1 = max(0, c - half), min(W, c + half)
            patch = q_full[r0:r1, c0:c1]
            if patch.size < 16:
                continue
            try:
                g = graycomatrix(patch, distances=[1], angles=ANGLES,
                                 levels=LEVELS, symmetric=True, normed=True)
                for prop in PROPS:
                    v = graycoprops(g, prop).mean()
                    feats[f"{channel_name}_glcm_{prop}_w{win}"][i] = v
            except Exception:
                pass
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N}  t={time.time()-t0:.0f}s", flush=True)
    print(f"  {channel_name} done in {time.time()-t0:.0f}s")
    return pd.DataFrame(feats)


if __name__ == "__main__":
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    print("Loading backscatter...")
    back, T = load_raster(BACK)
    print("Loading bathymetry...")
    bath, _ = load_raster(BATHY)

    print("Computing backscatter GLCM for train...")
    tr_bs = compute_glcm_feats(train, back, T, "bs")
    print("Computing backscatter GLCM for test...")
    te_bs = compute_glcm_feats(test, back, T, "bs")

    print("Computing bathymetry GLCM for train...")
    tr_bath = compute_glcm_feats(train, bath, T, "bath")
    print("Computing bathymetry GLCM for test...")
    te_bath = compute_glcm_feats(test, bath, T, "bath")

    tr = pd.concat([tr_bs, tr_bath], axis=1)
    te = pd.concat([te_bs, te_bath], axis=1)
    tr.to_csv(DATA / "train_glcm.csv", index=False)
    te.to_csv(DATA / "test_glcm.csv", index=False)
    print(f"Saved train_glcm.csv ({tr.shape}) and test_glcm.csv ({te.shape})")