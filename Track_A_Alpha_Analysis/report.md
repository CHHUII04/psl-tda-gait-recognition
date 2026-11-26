# Track A — Alpha complex and geometric topology of the raw acceleration clouds

All numbers below are produced by `scripts/01–07` (log: `results/run_log.txt`, tables: `results/tables/`).
Scale convention: GUDHI's Alpha filtration value is the **squared radius** α²; diagrams are stored raw (α²) and
analysed in the **radius scale r = √α²** (unit: g). A Rips edge length ε corresponds to roughly 2r, so Alpha and
Rips values are not numerically comparable without this factor of 2 (and the square root).

---

## 1. Dataset

| | value |
|---|---|
| training windows | 240 × 200 × 3 (80 per person A, B, C) |
| evaluation windows | 22 × 200 × 3 |
| NaN / inf | 0 / 0 (train and eval) |
| duplicate points inside a window | 0 (relevant: the Alpha complex would merge them) |
| ‖a‖ mean ± std (g) | A 1.09 ± 0.43, B 1.17 ± 0.55, C 1.32 ± 0.59 |
| ‖a‖ range (g) | 0.05 – 3.47 |
| window mean (≈ gravity in sensor frame) | A (0.18, 0.99, 0.08), B (−0.43, 0.91, −0.05), C (0.39, 0.95, −0.01) |
| RMS radius of the centred cloud (g) | A 0.594 ± 0.025, B 0.808 ± 0.023, C 1.016 ± 0.036 |

Windows were cut from one long recording per person with a sliding window: 9 968 of the 10 426 distinct training
points occur in more than one window, and 4 381 / 4 400 evaluation points also occur in training windows
(`window_overlap.csv`). Training windows are therefore **not independent**; this inflates any similarity between
windows of the same person (see limitations).

The zip archive `Exam - Data project.zip` contains byte-identical copies of both pickles (MD5 checked).

## 2. Why the Alpha complex, why Euclidean distance

* Each window is treated as a **point cloud** X ⊂ ℝ³ (time order is ignored in this track).
* The Alpha complex is the nerve of the union of balls B(x, r) ∩ Vor(x); it is **homotopy-equivalent to the union of
  balls** (same homotopy type as the Čech complex) but is a subcomplex of the Delaunay triangulation, hence small:
  on average 200 vertices, 1 376 edges, 2 327 triangles, 1 151 tetrahedra per window, computed in **0.027 s** per
  window (6.5 s for all 240). A Rips complex on the same 200 points up to dimension 2 has C(200,3) ≈ 1.3 M triangles.
* The ambient metric is **Euclidean**, the natural metric of acceleration vectors. Distances between acceleration
  vectors are physical (g) and invariant under rotations of the sensor frame. No per-axis standardisation or scaling
  was applied — it would destroy exactly this invariance.
* Homology with Z₂ coefficients, full filtration (tetrahedra included), so every H1 class has a correct death.

## 3. Rotation and translation invariance (scripts 03, 07)

Phones were carried with different orientations, so the three persons' series live in different, unknown sensor
frames. A pipeline that is not rotation invariant can learn *the phone orientation* instead of *the gait*. The Alpha
filtration depends only on pairwise Euclidean distances (Delaunay/Voronoi structure is preserved by isometries), so it
should be exactly invariant under X ↦ XQ (Q orthogonal) and X ↦ X + v.

| transformation (6 windows; 5 rotations each) | max \|ΔD\| distance matrix | max d_B H0 | max d_B H1 |
|---|---|---|---|
| random rotations Q ∈ SO(3) | 1.8e-15 | 3.3e-16 | 1.0e-15 |
| reflection (det Q = −1) | 1.3e-15 | 3.3e-16 | 1.4e-15 |
| translation by v ~ N(0, 5²) | 2.7e-15 | 1.2e-15 | 3.3e-15 |
| centring (remove gravity offset) | 8.9e-16 | 1.1e-16 | 3.9e-16 |
| **one random rotation for each of the 240 windows** | — | **3.9e-16** (mean 2.0e-16) | **5.0e-14** (mean 1.2e-15) |

Numbers of diagram points are unchanged in all cases. All differences are at floating-point precision: **the Alpha
diagrams are exactly invariant to the phone orientation and to the constant gravity offset**. Consequently the
"orientation shortcut" (window mean, which alone identifies the persons) is invisible to this track: whatever
separation we observe comes from the *shape and size* of the cloud.

## 4. What H0 and H1 mean here (script 04)

**H0.** All vertices enter at r = 0, so every H0 point has birth 0. Deaths are merge radii of connected components;
numerically, the H0 death radii equal **half the edge lengths of the Euclidean minimum spanning tree** (max error
6.9e-17 over 10 windows). H0 therefore encodes the **distribution of gaps** between acceleration samples: small deaths
= dense regions (slowly varying acceleration), large deaths = isolated points (sharp peaks, e.g. heel strikes,
reached only briefly). Exactly one essential class (0, ∞) per window; it is identical for all windows and contributes 0
to the bottleneck distances (checked: d_B with and without it are identical).

**H1.** About 330 H1 points per window, the vast majority hugging the diagonal (mean lifetime 0.005–0.010 g):
short-lived cycles created by local Delaunay triangulation of a noisy, tangled trajectory — **noise**. The few points
far from the diagonal are holes in the cloud formed by repeated excursions of the trajectory. The most persistent
class is not dramatically larger than the next one (top-1/top-2 ratio 1.27–1.41, not different between persons,
Kruskal–Wallis p = 0.11): there is **no single clean "gait loop"** in the raw 3D cloud. Its birth edge / killing
triangle lie in sparse outer regions of the cloud (figure `04_dominant_loop_geometry.png`).

**Class-wise statistics** (radius scale, mean ± std over 80 windows; all Kruskal–Wallis p < 1e-12 except where noted;
p-values are optimistic because overlapping windows are not independent samples):

| statistic | A | B | C |
|---|---|---|---|
| H0 total persistence | 11.81 ± 0.63 | 14.93 ± 0.85 | 20.09 ± 0.97 |
| H0 max finite death | 0.315 ± 0.068 | 0.327 ± 0.047 | 0.472 ± 0.083 |
| #H1 points | 345 ± 21 | 330 ± 17 | 316 ± 28 |
| H1 total persistence | 1.62 ± 0.17 | 2.23 ± 0.22 | 3.19 ± 0.30 |
| H1 max persistence | 0.080 ± 0.018 | 0.146 ± 0.034 | 0.188 ± 0.041 |
| H1 top1/top2 ratio | 1.33 ± 0.31 | 1.41 ± 0.42 | 1.27 ± 0.27 (p = 0.11) |

**Noise band** (script 07). Using c = 95 % quantile of d_B(full, 80 % subsample) as perturbation scale
(c_H1 = 0.076 g, c_H0 = 0.146 g), points with persistence > 2c are called significant (heuristic, in the spirit of
subsampling confidence sets):

| | A | B | C |
|---|---|---|---|
| windows with ≥ 1 significant H1 loop | **0 %** | 40 % | 79 % |
| mean # significant H1 loops | 0.00 | 0.49 | 1.54 |
| windows with ≥ 1 significant H0 gap | 65 % | 84 % | 100 % |

Person A's raw cloud has no loop that survives this noise level; C's cloud is larger and contains robust holes.

**Scale dependence** (`scale_dependence.csv`, figure `06_scale_dependence.png`). The diagram statistics are strongly
driven by the overall size of the cloud: correlation with the RMS radius over all windows is 0.96 (H0 total
persistence), 0.93 (H1 total persistence), 0.82 (H1 max persistence), but only 0.2–0.4 within a class. The RMS radius
alone already orders the persons A < B < C without overlap. After dividing by the RMS radius, H1 max persistence is
0.135 ± 0.031 (A) vs 0.180 ± 0.041 (B) vs 0.184 ± 0.040 (C): A keeps a relatively smaller loop, B and C become
indistinguishable. **A large part of the topological signal of the raw cloud is a size (movement-amplitude) signal.**

## 5. Bottleneck distances (script 05)

Exact bottleneck distance (no approximation parameter), radius scale, parallel computation with resumable cache.
Both 240 × 240 matrices are symmetric, finite, zero on the diagonal and have no zero off-diagonal entry.

| | min | median | max (off-diagonal) |
|---|---|---|---|
| d_B H0 | 0.0019 | 0.111 | 0.367 |
| d_B H1 | 0.0035 | 0.070 | 0.143 |

Files: `results/distance_matrices/bottleneck_H0.npy|csv`, `bottleneck_H1.npy|csv`, row order in `sample_order.csv`
(rows already sorted A, B, C).

## 6. Class-wise distance analysis (script 06, no classifier)

Mean (median) bottleneck distance:

| Homology | A-A | B-B | C-C | A-B | A-C | B-C |
|---|---|---|---|---|---|---|
| H0 mean | 0.088 | 0.067 | 0.114 | 0.082 | 0.186 | 0.169 |
| H0 median | 0.077 | 0.060 | 0.099 | 0.072 | 0.178 | 0.159 |
| H1 mean | 0.038 | 0.062 | 0.078 | 0.066 | 0.090 | 0.081 |
| H1 median | 0.037 | 0.061 | 0.073 | 0.066 | 0.089 | 0.079 |

| | within mean | between mean | between/within | silhouette (all) | silhouette A | B | C |
|---|---|---|---|---|---|---|---|
| H0 | 0.090 | 0.146 | 1.63 | 0.130 | −0.071 | 0.178 | 0.283 |
| H1 | 0.059 | 0.079 | 1.34 | 0.158 | **0.425** | 0.024 | 0.026 |

Interpretation:
* **H0 separates C from A and B** (A-C, B-C ≈ 0.17–0.19 vs C-C 0.11) but **not A from B**: the mean A-B distance
  (0.082) is even smaller than A-A (0.088), and A has a negative silhouette. The bottleneck distance only sees the
  largest discrepancy, which for H0 is governed by a few isolated points (largest gaps); H0 total persistence differs
  strongly between A and B (11.8 vs 14.9) but d_B is blind to such a sum.
* **H1 isolates A** (A-A 0.038, a compact cluster, silhouette 0.43) but **B and C overlap** (B-B 0.062, C-C 0.078,
  B-C 0.081; silhouettes ≈ 0.02).
* H0 and H1 are therefore **complementary**: H0 identifies C, H1 identifies A. Neither alone separates all three.
* Overall silhouettes (0.13 / 0.16) are low: the classes are distinguishable on average, not well separated clusters.
* Descriptively, the nearest diagram of a window has the same label in 91.7 % (H0) / 95.4 % (H1) of the cases; this
  is inflated by overlapping windows of the same recording and is **not** a classification accuracy.

## 7. MDS (figure `06_mds_H0_H1.png`)

Metric MDS (4 initialisations, seed 2025). Kruskal stress-1: **0.12 for H0** (faithful 2D picture), **0.26 for H1**
(poor fit — the H1 picture must be read with caution).
* H0: C forms its own region; A and B are intermixed.
* H1: A forms a tight arc; B and C occupy a common diffuse region (C more spread, with a few far outliers).
This matches the distance tables. MDS is a visualisation of the distance matrix, not evidence of separability by
itself.

## 8. Subsampling robustness (script 07)

12 windows (4 per class) × 5 random subsamples of 80 % of the points (160 / 200):

| | mean d_B(full, 80 %) | median | max | median within-class d_B | median between-class d_B |
|---|---|---|---|---|---|
| H0 | 0.053 | 0.036 | 0.274 | 0.076 | 0.135 |
| H1 | 0.043 | 0.042 | 0.085 | 0.058 | 0.078 |

* The perturbation caused by removing 20 % of the points is **~70–75 % of the typical within-class distance**.
  For H1 it is of the same order as the B-C between-class distance (0.081), which explains why B and C cannot be told
  apart with d_B(H1): the differences between B and C windows are at the sampling-noise level.
* H0 is sensitive to *which* points are removed when the cloud has isolated extreme points (C: up to 0.27): removing a
  peak sample changes the largest MST gap. Removing points can only make gaps larger, so H0 deaths increase
  systematically (mean max H0 persistence 0.389 → 0.408).
* Sensitivity grows with cloud size: mean d_B(H1) 0.029 (A), 0.048 (B), 0.053 (C).

## 9. Evaluation windows

Alpha diagrams (H0, H1, H2; raw α² and radius scale) for the 22 evaluation windows are stored in
`results/evaluation_diagrams/`, with per-window summaries in `eval_diagram_summary.csv`. No labels were predicted in
this track.

## 10. Limitations

* **Dependent samples**: overlapping windows make within-class distances smaller than for truly independent
  recordings; nearest-neighbour agreement and silhouettes are optimistic.
* **Time is ignored**: the raw cloud forgets the order of the samples, hence the periodicity of walking. The gait cycle
  does not appear as a robust H1 class in the raw 3D cloud.
* **Size confound**: most of the class differences are explained by the RMS size of the cloud (movement amplitude);
  topology adds a smaller, scale-free contribution (relative loop size of A).
* **Bottleneck = max**: blind to differences spread over many points (e.g. H0 total persistence A vs B).
* Noise band: heuristic (80 % subsampling without replacement, 12 windows), not a calibrated confidence set.
* One recording session per person: no evidence of generalisation to new sessions.

## 11. Conclusions and comparison points for the temporal (Rips / time-delay) pipeline

1. The Alpha filtration of the raw cloud is cheap (0.03 s/window), exactly invariant to phone orientation and
   gravity offset (verified to 1e-14), and contains genuine person-specific information.
2. H0 (gap structure ↔ isolated peaks) separates **C**; H1 (holes) isolates **A**; **B vs C** is the hard pair, at the
   level of sampling noise for H1.
3. The dominant signal is the **size** of the acceleration cloud; the raw cloud has **no significant gait loop for A**
   and only weak loops for B.

Questions to compare with the partner's experiments:
* Does the time-delay embedding create a significant H1 class (the gait cycle) for all three persons, where the raw
  cloud has none for A? Compare max H1 persistence in units of 2c.
* Does the temporal pipeline separate **B from C** better than d_B(H1) here (B-C 0.081 vs within B 0.062)?
* Does the Wasserstein distance (sums over all points) separate A from B in H0, where the bottleneck fails?
* After normalising the scale (e.g. dividing by the RMS radius), how much accuracy does each pipeline keep? This
  measures how much of each pipeline is "topology" vs "amplitude".
* Unit check before comparing numbers: Alpha r vs Rips ε ≈ 2r.

## Figures for the presentation

| figure | message |
|---|---|
| `02_3d_trajectories.png` | what a window looks like as a point cloud |
| `03_rotation_invariance.png` | same diagram after a random rotation (d_B ~ 1e-16) |
| `04_alpha_diagrams_representatives.png` | H0 (births = 0) and H1 (mostly noise near the diagonal) |
| `04_summary_statistics_by_class.png` | class-wise differences of diagram statistics |
| `06_bottleneck_heatmaps.png` + `06_mds_H0_H1.png` | H0 isolates C, H1 isolates A, B/C overlap |
| `06_scale_dependence.png` | the size confound |
| `07_subsampling_robustness.png` | sampling noise ≈ within-class distances |
