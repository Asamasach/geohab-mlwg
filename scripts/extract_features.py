"""Multi-scale feature extraction from bathymetry + backscatter rasters at point locations."""
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi
from pathlib import Path

DATA = Path(__file__).parent
BATHY = DATA / "MBES" / "bathymetry.tif"
BACK = DATA / "MBES" / "backscatter.tif"

SCALES = [3, 5, 9, 15, 25, 51, 101]  # pixel radii for multiscale stats (1 px = 25 cm)


def sample_point_features(raster_path, xs, ys, scales, band_prefix):
    feats = {}
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1).astype(np.float32)
        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        H, W = arr.shape
        T = ds.transform
        # Convert (x,y) -> (row,col)
        inv = ~T
        cols_f, rows_f = inv * (xs, ys)
        rows = np.round(rows_f).astype(int)
        cols = np.round(cols_f).astype(int)

        in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        print(f"  {band_prefix}: {in_bounds.sum()}/{len(xs)} points in bounds")

        # Raw value at point
        raw = np.full(len(xs), np.nan, dtype=np.float32)
        raw[in_bounds] = arr[rows[in_bounds], cols[in_bounds]]
        feats[f"{band_prefix}_raw"] = raw

        # Gradient magnitude (slope proxy) at pixel resolution
        gy, gx = np.gradient(arr)
        slope = np.sqrt(gx * gx + gy * gy)
        sl = np.full(len(xs), np.nan, dtype=np.float32)
        sl[in_bounds] = slope[rows[in_bounds], cols[in_bounds]]
        feats[f"{band_prefix}_slope"] = sl

        # Multiscale window stats (uniform mean + std via box filter)
        a_nan = np.isnan(arr)
        a_filled = np.where(a_nan, 0.0, arr)
        mask = (~a_nan).astype(np.float32)

        for r in scales:
            k = 2 * r + 1
            # mean
            s = ndi.uniform_filter(a_filled, size=k, mode="nearest")
            c = ndi.uniform_filter(mask, size=k, mode="nearest")
            mean = np.where(c > 0, s / np.clip(c, 1e-6, None), np.nan)
            # E[x^2]
            s2 = ndi.uniform_filter(a_filled * a_filled, size=k, mode="nearest")
            ex2 = np.where(c > 0, s2 / np.clip(c, 1e-6, None), np.nan)
            var = np.clip(ex2 - mean * mean, 0, None)
            std = np.sqrt(var)

            m_at = np.full(len(xs), np.nan, dtype=np.float32)
            s_at = np.full(len(xs), np.nan, dtype=np.float32)
            m_at[in_bounds] = mean[rows[in_bounds], cols[in_bounds]]
            s_at[in_bounds] = std[rows[in_bounds], cols[in_bounds]]
            feats[f"{band_prefix}_mean_r{r}"] = m_at
            feats[f"{band_prefix}_std_r{r}"] = s_at

            # TPI: raw - mean (for bathymetry this is topographic position)
            tpi_full = arr - mean
            tpi_at = np.full(len(xs), np.nan, dtype=np.float32)
            tpi_at[in_bounds] = tpi_full[rows[in_bounds], cols[in_bounds]]
            feats[f"{band_prefix}_tpi_r{r}"] = tpi_at

    return feats, in_bounds


def build_features(points_df):
    xs = points_df["x"].values.astype(np.float64)
    ys = points_df["y"].values.astype(np.float64)
    print(f"Sampling bathymetry for {len(points_df)} points...")
    fb, ib_b = sample_point_features(BATHY, xs, ys, SCALES, "bath")
    print(f"Sampling backscatter for {len(points_df)} points...")
    fbs, ib_bs = sample_point_features(BACK, xs, ys, SCALES, "bs")
    feats = {**fb, **fbs}
    return pd.DataFrame(feats)


if __name__ == "__main__":
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    print(f"train={len(train)} test={len(test)}")

    tr_f = build_features(train)
    te_f = build_features(test)

    tr_out = pd.concat([train.reset_index(drop=True), tr_f], axis=1)
    te_out = pd.concat([test.reset_index(drop=True), te_f], axis=1)

    tr_out.to_csv(DATA / "train_features.csv", index=False)
    te_out.to_csv(DATA / "test_features.csv", index=False)
    print("saved train_features.csv, test_features.csv")
    print("features:", list(tr_f.columns))