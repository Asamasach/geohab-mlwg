# 1st place — multi-scale MBES features, GBDT/CNN ensemble, targeted flips

First, a big thank-you to Benjamin Misiuk, the GeoHab MLWG 2026 organisers, and to the Deakin Marine Mapping Group (Schimel, Gaylard, Ierodiaconou et al.) who collected and shared the Refuge Cove multibeam dataset under CC-BY-4.0. The 25 cm bathymetry + backscatter rasters are an outstanding teaching dataset for seafloor habitat classification.

The task — assigning each ground-truth point to one of five seabed classes (ALG, FMAT, NVB, SGAM, SGZ) from MBES bathymetry + backscatter — is small (~98 test points) but information-rich, and it punishes anything that overfits to the public LB.

## TL;DR

The win was **not** the heavy stacking machinery. It was a **strong LGB baseline (~0.878 private F1) plus a couple of targeted single-point flips that 4–7 of my models agreed on.**

Concretely:

- A solid LGB on multi-scale terrain + texture features was already extremely competitive on its own.
- The full GBDT (LGB + XGB + CAT) and 2-channel CNN ensemble's main job ended up being **a consensus signal** — telling me which test points the baseline got wrong with high confidence.
- Where ≥4 of 7 models disagreed with the baseline AND the spatial nearest neighbours agreed with the consensus, I flipped the prediction. Two single-point flips (e.g., ID 7 → SGAM, ID 91 NVB → ALG) added ≈+0.015 private F1 each over the bare LGB.

Stacking didn't beat the baseline by much. Targeted, ensemble-justified flips did.

## Validation

Random stratified k-fold is optimistic on a single survey site like Refuge Cove because nearby points are highly spatially correlated. I used **two CV setups in parallel**:

1. 5-fold stratified random folds — for hyperparameter selection and the OOF stack.
2. 5-fold spatial folds (KMeans on (x, y) at 10 / 20 / 40 clusters → GroupKFold) — as a sanity check that gains weren't just spatial leakage.

Decisions had to look good on both. When they disagreed, I trusted the spatial-CV signal more.

The other essential validation tool was **counting model agreement on the test set itself**. With seven independent learners (LGB, XGB, CAT, ET, CNN-cpu, CNN-gpu, CNN-big), a flip the baseline disagreed with that ≥4 of the others endorsed was a much higher-quality signal than any single-model OOF.

## Feature engineering — multi-scale + texture

For each point I sampled the **bathymetry** and **backscatter** rasters at multiple window radii — 3, 5, 9, 15, 25, 51, 101 px (≈ 75 cm to 25 m) — and computed:

- raw value at the point
- pixel-level slope (gradient magnitude)
- multi-scale **mean** and **std** in each window
- **TPI** (raw − local mean) at each scale — the relief-from-mean signature

On top of those I computed a **GLCM texture pack** on backscatter and bathymetry (16×16 and 32×32 windows, 4 directions, 32 quantisation levels, all six standard properties: contrast, dissimilarity, homogeneity, energy, correlation, ASM).

I also prototyped a richer **v2** feature set (multi-scale slope/aspect/curvature/VRM + backscatter percentiles), but on the public LB v2 generalised noticeably worse than v1 — almost certainly fine-scale-noise overfit. The final pipeline used v1 + GLCM only. (See "things that didn't help" below.)

## Models

### GBDT bag

All three boosters bagged across 5 random seeds × 5 stratified folds:

- **LightGBM** — multiclass log-loss, 63 leaves, lr 0.05, balanced class weight, 3 000 round budget with early-stopping on the held fold.
- **XGBoost** — depth 7, lr 0.05, sample-weighted by 1 / class-frequency.
- **CatBoost** — depth 7, lr 0.05, `auto_class_weights="Balanced"`.

ENS3 = mean of the three. ExtraTrees was kept on hand as another OOF input but didn't make the final blend.

### 2-channel CNN on raster patches

- 64 × 64 px patches around each point (≈ 16 m × 16 m), 2 channels (bathy, backscatter), normalised per-channel.
- 4 conv blocks (32 → 64 → 128 → 192 ch) with BN+ReLU, GAP+GMP fusion → FC head.
- AdamW, cosine LR, 40 epochs, label smoothing 0.05, class-weighted CE, **mixup α=0.2** on half the batches.
- Augmentations: rotations, h/v flips, ±4 px shifts, additive Gaussian noise.
- **16-way TTA** (4 rotations × 4 flip combinations) at inference.
- Bagged across 3 seeds × 5 folds. A "big" wider variant trained for diversity.

### Spatial kNN (low weight)

A distance-weighted kNN on (x, y) gave a high spatial-CV F1 but generalised poorly on its own — kept only as one of the seven consensus voters.

## Stacking — necessary but not sufficient

Final blend was a grid search over weights of (GBDT, CNN-gpu, CNN-cpu) on OOF macro-F1. The optimum landed somewhere around 0.5 / 0.3 / 0.2 but the surface was very flat — anything in that neighbourhood scored within noise. **The blended argmax barely outperformed the LGB baseline on private F1.**

That was the moment I realised the right framing was different.

## The actual winning step — targeted, consensus-justified flips

With only 98 test points, every flip is worth ≈ 0.005 macro-F1. A flip from a wrong label to a right label is a +0.005-or-so gain; the wrong direction costs the same. So I treated the test set as a discrete decision problem.

For each of the 98 points I asked:

1. What does the LGB baseline say?
2. What does each of the other six models say (XGB, CAT, ET, CNN-cpu, CNN-gpu, CNN-big)?
3. What do the 10 nearest training neighbours by (x, y) say?

I only flipped a baseline prediction when:

- **≥ 4 of the 7 models** agreed on the new class, **and**
- The new class also dominated among the 10 spatial NNs (≥ 5 of 10), **and**
- The CNN-gpu max-prob was > 0.55 (high confidence), **and**
- The GBDT margin (top1 − top2) was < 0.35 (genuinely uncertain).

A few high-conviction flips (ID 7 → SGAM, ID 91 NVB → ALG, ID 23 → ALG, ID 48 → SGAM) each contributed independently to private F1. Some flips that looked attractive in isolation cancelled out when combined — e.g., a "5-flip super" submission scored *worse* than the bare baseline (0.863 vs 0.878 private) because the second-tier flips were noisier than I'd thought. The lesson: a flip should be locked in only if the consensus and spatial evidence are both strong; do not pile up marginal flips.

## Things that didn't help

- **v2 features** (richer terrain derivatives) — overfit the public LB.
- **Heavier CNNs** — same OOF, more variance.
- **Test-time pseudo-labelling** — held-out folds got worse.
- **Stack via meta-learner** (logistic/xgb on OOFs) — flat-out worse than a grid-searched convex blend.
- **Piling up flips beyond the 2–3 highest-conviction ones** — added noise.

## Hardware

Single RTX 2070 SUPER (8 GB). Full pipeline (features + GBDTs + CNNs + stack) trains in well under an hour. Only the CNN bagging benefited from GPU; everything else is CPU-bound.

## Take-aways for small-test-set seafloor comps

1. **Build a strong baseline first**, then validate every "improvement" against it on private-equivalent spatial CV.
2. **Use ensembles as a consensus signal**, not just to take an argmax. With 7 model voters and 98 test points, model agreement is a more robust per-point signal than any single OOF.
3. **Spatial nearest-neighbour evidence** is essential as a sanity check — a flip should be plausible given what's labelled within ~10 metres.
4. **Resist piling up marginal flips.** On a 98-point test set, every wrong flip costs ~0.005 F1.

Happy to answer questions, and looking forward to seeing the rest of the writeups — particularly anything that managed to make richer terrain derivatives (VRM, aspect, curvature) generalise without overfitting.

— Seyed Mehdi Sadat Hosseini ([@asamasach](https://www.kaggle.com/asamasach))
