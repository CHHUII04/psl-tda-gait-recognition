# PSL Week – TDA exam: gait recognition from smartphone accelerometer data

Identify walkers A / B / C from 200 × 3 accelerometer windows with persistent homology + machine learning,
and predict the 22 unlabeled windows.

**Final answer (windows 1–22, file order):** `B C B B A A B C B B B B A B B A B A B A A B` → **A = 7, B = 13, C = 2**

## Three topological routes (all rotation invariant)

| route | representation | filtration | blocked-CV accuracy |
|---|---|---|---|
| A (subject, Method A) | raw cloud {a(t)} ⊂ R³ | Alpha | ≈ 0.96–0.97 |
| B (subject, Method B) | delay vectors (a(t), a(t+ℓ), a(t+2ℓ)) ∈ R⁹ | Vietoris–Rips | ≈ 0.95–0.97 |
| C (final model) | ‖a(t)‖ | sub/superlevel filtration on the path graph, H0 | 1.00 |

## Where to look

| path | content |
|---|---|
| `TDA_project_solution.ipynb` | **main notebook**: exploration, filtrations, distances (bottleneck / Wasserstein), vectorizations × classifiers, delay/lag sweep, final model, predictions; Section 7 = synthesis |
| `Track_A_Alpha_Analysis/` | detailed geometric analysis of route A (Alpha): invariance tests, bottleneck matrices, MDS, scale effect, robustness — see `report.md` |
| `Final_Integration/` | common-protocol comparison of the 3 routes, rotation demo, final prediction figure — see `README.md` |
| `Exam - Data project.ipynb` | original starter notebook (unchanged) |
| `tda_utils.py` | helpers for parallel computations (needed on Windows) |

## Run
Python 3.11 with `gudhi` (3.13), `numpy` < 2, `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `plotly`.
Open `TDA_project_solution.ipynb` and run all cells (first run ≈ 40 min: diagrams and distance matrices are then cached as
`cache_*.pkl`, which are not versioned). Track A: `python Track_A_Alpha_Analysis/scripts/run_all.py`.
