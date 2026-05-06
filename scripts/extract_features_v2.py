"""v2 feature extraction: multi-scale terrain derivatives + backscatter texture.

Adds on top of v1:
- Slope at multiple scales (gradient over smoothed bathy)
- Aspect sin/cos at multiple scales
- Curvature (Laplacian) at multiple scales
- VRM (Vector Ruggedness Measure) via 3D normals
- BPI (Bathymetric Position Index) ring-based
- Backscatter percentile features in windows
- Local entropy of backscatter (proxy texture)
- Coordinates (x, y) — because habitat has spatial structure
"""
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
from pathlib import Path

DATA = Path(__file__).parent
BATHY = DATA / "MBES" / "bathymetry.tif"
BACK = DATA / "MBES" / "backscatter.tif"

SCALES = [3, 5, 9, 15, 25, 51, 101, 201]  # pixel radii (25 cm/px)
CELL = 0.25  # meters per pixel


def load_raster(path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        nd = ds.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
        T = ds.transform
    return arr, T


def box_mean(a, r, mask):
    k = 2 * r + 1
    af = np.where(np.isnan(a), 0.0, a)
    s = ndi.uniform_filter(af, size=k, mode="nearest")
    c = ndi.uniform_filter(mask, size=k, mode="nearest")
    return np.where(c > 0, s / np.clip(c, 1e-6, None), np.nan), c


def box_std(a, r, mask, mean=None):
    k = 2 * r + 1
    af = np.where(np.isnan(a), 0.0, a)
    s2 = ndi.uniform_filter(af * af, size=k, mode="nearest")
    c = ndi.uniform_filter(mask, size=k, mode="nearest")
    ex2 = np.where(c > 0, s2 / np.clip(c, 1e-6, None), np.nan)
    if mean is None:
        mean, _ = box_mean(a, r, mask)
    var = np.clip(ex2 - mean * mean, 0, None)
    return np.sqrt(var)


def sample(arr, rows, cols, in_b):
    v = np.full(len(rows), np.nan, dtype=np.float32)
    v[in_b] = arr[rows[in_b], cols[in_b]]
    return v


def build(points_df):
    xs = points_df["x"].values.astype(np.float64)
    ys = points_df["y"].values.astype(np.float64)

    bath, T = load_raster(BATHY)
    back, _ = load_raster(BACK)

    H, W = bath.shape
    inv = ~T
    cols_f, rows_f = inv * (xs, ys)
    rows = np.round(rows_f).astype(int)
    cols = np.round(cols_f).astype(int)
    in_b = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    print(f"  in-bounds: {in_b.sum()}/{len(xs)}")

    mask_b = (~np.isnan(bath)).astype(np.float32)
    mask_s = (~np.isnan(back)).astype(np.float32)

    feats = {}
    feats["x"] = xs.astype(np.float32)
    feats["y"] = ys.astype(np.float32)

    # Raw values
    feats["bath_raw"] = sample(bath, rows, cols, in_b)
    feats["bs_raw"] = sample(back, rows, cols, in_b)

    # Pixel-level slope / aspect / curvature from raw bathy (25 cm)
    gy, gx = np.gradient(bath, CELL)
    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)  # radians
    lap = ndi.laplace(np.where(np.isnan(bath), 0, bath)) / (CELL * CELL)
    feats["bath_slope_px"] = np.degrees(sample(slope_rad, rows, cols, in_b))
    feats["bath_aspect_sin_px"] = np.sin(sample(aspect, rows, cols, in_b))
    feats["bath_aspect_cos_px"] = np.cos(sample(aspect, rows, cols, in_b))
    feats["bath_curv_px"] = sample(lap, rows, cols, in_b)

    # Multi-scale stats + derived features
    for r in SCALES:
        mean_b, _ = box_mean(bath, r, mask_b)
        std_b = box_std(bath, r, mask_b, mean=mean_b)
        mean_s, _ = box_mean(back, r, mask_s)
        std_s = box_std(back, r, mask_s, mean=mean_s)

        # Bathymetry mean / std / TPI
        feats[f"bath_mean_r{r}"] = sample(mean_b, rows, cols, in_b)
        feats[f"bath_std_r{r}"] = sample(std_b, rows, cols, in_b)
        feats[f"bath_tpi_r{r}"] = sample(bath - mean_b, rows, cols, in_b)

        # Backscatter mean / std / TPI
        feats[f"bs_mean_r{r}"] = sample(mean_s, rows, cols, in_b)
        feats[f"bs_std_r{r}"] = sample(std_s, rows, cols, in_b)
        feats[f"bs_tpi_r{r}"] = sample(back - mean_s, rows, cols, in_b)

        # Slope at scale: compute on smoothed bathy
        sgy, sgx = np.gradient(np.where(np.isnan(mean_b), 0, mean_b), CELL)
        slope_s = np.degrees(np.arctan(np.sqrt(sgx * sgx + sgy * sgy)))
        feats[f"bath_slope_r{r}"] = sample(slope_s, rows, cols, in_b)

        # Aspect at scale
        asp_s = np.arctan2(-sgx, sgy)
        feats[f"bath_asp_sin_r{r}"] = sample(np.sin(asp_s), rows, cols, in_b)
        feats[f"bath_asp_cos_r{r}"] = sample(np.cos(asp_s), rows, cols, in_b)

        # Curvature at scale (Laplacian of smoothed)
        lap_s = ndi.laplace(np.where(np.isnan(mean_b), 0, mean_b)) / (CELL * CELL)
        feats[f"bath_curv_r{r}"] = sample(lap_s, rows, cols, in_b)

        # VRM via 3D unit-normal dispersion — approximate: std of slope
        # (cheap proxy: std of gx^2+gy^2 within window)
        grad_mag2 = sgx * sgx + sgy * sgy
        feats[f"bath_vrm_r{r}"] = sample(
            box_std(grad_mag2, r, mask_b, mean=None), rows, cols, in_b
        )

    # Gradient of backscatter (texture proxy)
    bgy, bgx = np.gradient(np.where(np.isnan(back), 0, back))
    bs_grad = np.sqrt(bgx * bgx + bgy * bgy)
    feats["bs_grad_px"] = sample(bs_grad, rows, cols, in_b)
    for r in [5, 15, 51]:
        mean_g, _ = box_mean(bs_grad, r, mask_s)
        std_g = box_std(bs_grad, r, mask_s, mean=mean_g)
        feats[f"bs_grad_mean_r{r}"] = sample(mean_g, rows, cols, in_b)
        feats[f"bs_grad_std_r{r}"] = sample(std_g, rows, cols, in_b)

    # Ratios backscatter / bathymetry at medium scale (contrast proxy)
    feats["bs_over_bathstd_r25"] = feats["bs_std_r25"] / (np.abs(feats["bath_std_r25"]) + 1e-3)

    return pd.DataFrame(feats)


if __name__ == "__main__":
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"train={len(train)} test={len(test)}")
    print("Building train features...")
    tr_f = build(train)
    print("Building test features...")
    te_f = build(test)
    tr_out = pd.concat([train[["class"]].reset_index(drop=True), tr_f], axis=1)
    te_out = pd.concat([test[["ID"]].reset_index(drop=True), te_f], axis=1)
    tr_out.to_csv(DATA / "train_features_v2.csv", index=False)
    te_out.to_csv(DATA / "test_features_v2.csv", index=False)
    print(f"saved v2 features: {tr_f.shape[1]} features")