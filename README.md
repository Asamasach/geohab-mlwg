# GeoHab 2026 MLWG — 1st place solution

Multibeam-sonar seafloor habitat classification on the Refuge Cove dataset (Wilsons Promontory, Australia). Five classes (ALG, FMAT, NVB, SGAM, SGZ); evaluation is macro F1.

- **Final placement:** 1st (private LB 0.87866)
- **Full writeup:** [`discussion_post.md`](./discussion_post.md)
- **Comp page:** https://www.kaggle.com/competitions/geohab-mlwg-competition-2026

## Repository contents

```
.
├── discussion_post.md   # Full methodology writeup
├── scripts/
│   ├── extract_features.py     # multi-scale terrain stats (v1)
│   ├── extract_features_v2.py  # richer terrain derivatives (v2 — overfit on LB)
│   ├── extract_glcm.py         # GLCM texture features on bathy + backscatter
│   ├── extract_patches.py      # 64×64 raster patches around each point
│   ├── extract_patches_128.py  # 128×128 patches (wider context)
│   ├── train_baseline.py       # LightGBM baseline
│   ├── train_v1_ensemble.py    # LGB + XGB + CatBoost + ExtraTrees on v1 features
│   ├── train_v1_glcm.py        # LGB on v1 + GLCM
│   ├── train_v2.py             # LGB on v2 features
│   ├── train_ensemble.py       # multi-model v2 ensemble
│   ├── train_robust.py         # spatial-CV-regularised models
│   ├── train_cnn.py            # 2-channel CNN on 64×64 patches (CPU)
│   ├── train_cnn_v2.py         # wider CNN variant (CPU)
│   ├── train_cnn_gpu.py        # GPU CNN with mixup + 16-way TTA
│   ├── train_cnn_big.py        # ResNet-style CNN (GPU)
│   ├── train_cnn_128.py        # CNN on 128×128 patches (GPU)
│   ├── train_dual_scale.py     # dual-scale (64 + 128) CNN
│   ├── spatial_cv.py           # spatial KMeans → GroupKFold validation
│   ├── spatial_knn.py          # spatial k-NN consensus voter
│   ├── stack_final.py          # GBDT + CNN blend (CPU CNNs)
│   └── stack_gpu.py            # GBDT + CNN blend (GPU CNNs)
└── README.md
```

## Reproducing

The competition data, derived feature CSVs, OOF/PTE arrays, and submission CSVs are **not** redistributed in this repo (Kaggle TOS). To reproduce:

1. Accept the comp rules at https://www.kaggle.com/competitions/geohab-mlwg-competition-2026 and download the data:

   ```
   kaggle competitions download -c geohab-mlwg-competition-2026 -p data/raw/
   ```

2. Place the MBES rasters at `MBES/bathymetry.tif` and `MBES/backscatter.tif`.
3. Run the feature extractors:

   ```
   python scripts/extract_features.py
   python scripts/extract_glcm.py
   python scripts/extract_patches.py
   ```

4. Train and stack:

   ```
   python scripts/train_v1_ensemble.py    # GBDT bag
   python scripts/train_cnn_gpu.py        # GPU CNN (3 seeds × 5 folds)
   python scripts/stack_gpu.py            # blend + targeted-flip refinement
   ```

## Hardware

A single 8 GB consumer GPU (RTX 2070 SUPER) was sufficient. The full pipeline trains in well under an hour.

## Acknowledgements

Dataset by Schimel, Gaylard, Ierodiaconou et al. (Deakin Marine Mapping Group), released CC-BY-4.0. Competition organised by Benjamin Misiuk (Memorial University of Newfoundland) and the GeoHab MLWG.
