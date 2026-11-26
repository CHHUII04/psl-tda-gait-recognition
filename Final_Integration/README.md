# Final integration (P0 / P1)

Experiments that put the three routes under one protocol, for the presentation.
All results are also shown in Section 7 of `../TDA_project_solution.ipynb`.

| route | representation | filtration |
|---|---|---|
| A | raw cloud {a(t)} ⊂ R³ | Alpha (radius scale) |
| B | delay vectors (a(t), a(t+2), a(t+4)) ∈ R⁹ | Vietoris–Rips |
| C | ‖a(t)‖ | sub/superlevel filtration on the path graph (H0 valleys / peaks) |

Protocol: blocked & purged 5-fold CV (windows overlap in time), random forest (300 trees).

## Run (from this folder, Anaconda Python with gudhi)
```bash
python p0_1_unified_comparison.py              # route x {H0, H1, both} x {landscape, silhouette}
python p0_2_rotation_demo.py                   # rotate the test windows: baselines vs TDA routes
python p0_3_predictions_and_p1_4_annotated.py  # 22 labels (file order) + annotated (b, d, d-b) figure
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
