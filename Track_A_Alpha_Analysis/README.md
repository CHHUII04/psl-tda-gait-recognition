# Track A — Alpha complex / geometric topology

Self-contained branch of the TDA gait-recognition project: Alpha-complex persistence of the raw 3D acceleration
clouds, bottleneck distances, class-wise geometry, MDS, rotation and subsampling robustness. No classifier is trained
here (vectorizations, classifiers, Wasserstein, time-delay and Rips experiments belong to the partner track).

## Requirements
Python 3.11 with `gudhi` (tested 3.13.0), `numpy` 1.26, `scipy`, `scikit-learn`, `pandas`, `matplotlib`.
The data files `psl_week_acc_train_data.pkl` and `psl_week_acc_eval_data.pkl` must be in the parent folder.

## Run
```bash
cd Track_A_Alpha_Analysis/scripts
python run_all.py            # all steps, ~6 min on 16 cores (bottleneck matrices dominate)
python run_all.py --force    # recompute cached diagrams / distance matrices
```
Single steps can be run individually (`python 05_bottleneck_distances.py`); later steps reuse saved results.

| script | content |
|---|---|
| `common.py` | paths, data loading, `alpha_persistence()`, conventions (α² vs radius), colours, seed |
| `01_data_audit.py` | shapes, labels, NaN/inf, ranges, duplicates, window overlap |
| `02_visualize_data.py` | XYZ / norm time series, 3D trajectories |
| `03_rotation_invariance.py` | rotations, reflection, translations: distance matrices + diagrams |
| `04_alpha_persistence.py` | diagrams for train + eval (cached), lifetime statistics, diagram figures |
| `05_bottleneck_distances.py` | 240×240 exact bottleneck matrices H0 / H1 (parallel, resumable) |
| `06_class_analysis.py` | heatmaps, within/between distances, silhouettes, MDS, scale dependence |
| `07_robustness.py` | rotation of all windows, 80 % subsampling, noise band |

## Outputs
* `results/persistence_diagrams/` — `train_diagrams_sqradius.pkl` (list of {0: H0, 1: H1, 2: H2}, raw GUDHI α² values,
  essential intervals kept), per-dimension pickles, radius-scale `.npz`, essential intervals, persistence pairs.
* `results/evaluation_diagrams/` — same for the 22 evaluation windows.
* `results/distance_matrices/` — `bottleneck_H0.npy`, `bottleneck_H1.npy` (+ labelled `.csv`), `sample_order.csv`, MDS coordinates.
* `results/tables/` — all numerical results (CSV). `results/run_log.txt` — console log of the full run.
* `figures/` — PNG figures (200 dpi).
* `report.md` — results and interpretation.

Loading example:
```python
import numpy as np, pickle
D = pickle.load(open("results/persistence_diagrams/train_diagrams_sqradius.pkl", "rb"))  # alpha^2 scale
H1_radius = [np.sqrt(d[1]) for d in D]
B1 = np.load("results/distance_matrices/bottleneck_H1.npy")                             # radius scale
```

Random seed: 2025 (`common.SEED`).
