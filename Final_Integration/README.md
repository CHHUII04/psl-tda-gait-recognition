# Final integration (P0 / P1)

Experiments that put the three routes under one protocol, for the presentation.
All results are also shown in Section 7 of `../TDA_project_solution.ipynb`.

| route | representation | filtration |
|---|---|---|
| A | raw cloud {a(t)} ⊂ R³ | Alpha (radius scale) |
| B | delay vectors (a(t), a(t+ℓ), a(t+2ℓ)) ∈ R⁹, ℓ = 2 (Question 4) and ℓ = 8 (best lag of the sweep) | Vietoris–Rips |
| C | ‖a(t)‖ | sub/superlevel filtration on the path graph (H0 valleys / peaks) |

Protocol: blocked & purged 5-fold CV (windows overlap in time), random forest (300 trees).

## Run (from this folder, Anaconda Python with gudhi)
```bash
python p0_1_unified_comparison.py              # route x {H0, H1, both} x {landscape, silhouette}
python p0_2_rotation_demo.py                   # rotate the test windows: baselines vs TDA routes
python p0_3_predictions_and_p1_4_annotated.py  # 22 labels (file order) + annotated (b, d, d-b) figure
python p0_4_confidence_stability.py           # probabilities + 20-seed stability of the 22 predictions, window #14
python p0_5_orientation_drift.py              # rotation that changes DURING the window (phone turning)
```
Diagrams are cached in `cache/` (delete to recompute). `final_utils.py` holds the shared code.

## Results

**P0-1 route × homology** (`tables/p0_1_route_x_homology.csv`, `figures/p0_1_route_x_homology.png`)

| route | first diagram | second diagram | both |
|---|---|---|---|
| A Alpha, landscape | H0 0.671 | H1 0.974 | 0.941 |
| A Alpha, silhouette | H0 0.967 | H1 0.961 | 0.967 |
| B delay-Rips, landscape | H0 0.693 | H1 0.953 | 0.959 |
| B delay-Rips, silhouette | H0 0.959 | H1 0.945 | 0.969 |
| B delay-Rips **lag 8**, landscape | H0 0.937 | H1 0.927 | 0.993 |
| B delay-Rips **lag 8**, silhouette | H0 0.969 | H1 0.982 | **1.000** |
| C ‖a‖, landscape | valleys 0.993 | peaks 0.996 | 0.996 |
| C ‖a‖, silhouette | valleys 0.985 | peaks **1.000** | **1.000** |

**P0-2 rotation demo** (`tables/p0_2_rotation_demo.csv`, `figures/p0_2_rotation_demo.png`)

| method | CV acc. (as recorded) | CV acc. (test rotated) | eval predictions unchanged |
|---|---|---|---|
| window mean (orientation) | 1.000 | **0.310** | 7/22 |
| xyz mean/std/min/max | 1.000 | **0.554** | 10/22 |
| route A | 0.967 | 0.967 | 22/22 |
| route B | 0.969 | 0.969 | 22/22 |
| route C | 1.000 | 1.000 | 22/22 |

**P0-3 final predictions** (`tables/final_predictions_22.csv`, `figures/p0_3_final_predictions.png`)

Windows 1–22: `B C B B A A B C B B B B A B B A B A B A A B` → **A = 7, B = 13, C = 2** (sum 22).
Routes A and B disagree only on window #14 (they say C).

**P1-4** `figures/p1_4_annotated_diagrams.png`: Alpha H1 point (b, d) = (0.283, 0.423), persistence 0.139;
peak of ‖a‖ 2.236 g merging at 0.371 g, persistence 1.865 g.

**P0-4 confidence and seed stability** (`tables/p0_4_confidence_22.csv`, `figures/p0_4_confidence_22.png`, `figures/p0_4_window14.png`)

All 22 predictions of the final model: probability ≥ 0.98, margin ≥ 0.96, identical for 20 random-forest seeds (every pipeline:
100 % seed agreement). Window #14: route C → B (0.998), routes A / B (lag 2 and 8) → C (0.97–0.998). Its peak total
persistence (42.1) lies inside the B training range (37.0–46.8), C starts at 47.8.

**P0-5 orientation drift** (`tables/p0_5_orientation_drift.csv`, `figures/p0_5_orientation_drift.png`)

Test windows rotated by an angle growing from 0 to θ during the window (blocked CV accuracy):

| drift θ | A Alpha | B delay-Rips lag 8 | C ‖a‖ |
|---|---|---|---|
| 0° | 0.967 | 1.000 | 1.000 |
| 30° | 0.766 | 0.983 | 1.000 |
| 45° | 0.531 | 0.941 | 1.000 |
| 90° | 0.343 | 0.519 | 1.000 |

Window #14 has 47° of drift (angle between the gravity directions of its first 25 and last 100 samples); training windows:
median 1–4°, 95 % ≤ 9°. Drifting windows are pushed towards C by routes A/B → the #14 disagreement is explained; final label B.

**P0-6 amplitude control** (`p0_6_amplitude_control.py`, `tables/p0_6_amplitude_control.csv`)

Amplitude alone already classifies perfectly: mean + std of |a(t)| → 1.000, RMS radius of the cloud (one number) → 1.000.
After removing the amplitude of every window (|a| z-scored, cloud scaled to RMS radius 1): TDA C 0.901, B 0.823,
A 0.690 (chance 0.33) — topology keeps shape information, but the autocorrelation of |a| (rhythm) stays at 1.000.

**P0-7 information dissection** (`p0_7_information_dissection.py`, `tables/p0_7_information_dissection.csv`,
`figures/p0_7_information_dissection.png`) — blocked CV accuracy

| representation | raw | amplitude removed | time shuffled | amplitude removed + shuffled |
|---|---|---|---|---|
| amplitude: mean, std of ‖a‖ | 1.000 | 0.279 | 1.000 | 0.279 |
| quantiles of ‖a‖ | 1.000 | 0.907 | 1.000 | 0.907 |
| autocorrelation of ‖a‖ | 1.000 | 1.000 | 0.299 | 0.299 |
| TDA A: Alpha (cloud) | 0.967 | 0.690 | 0.967 | 0.690 |
| TDA B: delay Rips, lag 8 | 1.000 | 0.823 | 1.000 | 0.679 |
| TDA C: sublevel of ‖a‖ | 1.000 | 0.901 | 1.000 | 0.886 |

Reading: Alpha ignores time (unchanged by shuffling); delay Rips is the only topological route that uses temporal
structure (0.823 → 0.679 when shuffled, down to the Alpha level); route C uses amplitude + value distribution
(≈ quantiles), almost no temporal shape; rhythm is captured best by the (non-topological) autocorrelation.
