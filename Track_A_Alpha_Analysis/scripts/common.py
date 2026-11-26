"""Shared utilities for Track A (Alpha complex / geometric topology).

Conventions used everywhere in Track A
--------------------------------------
* GUDHI's AlphaComplex stores the SQUARED radius alpha^2 as filtration value
  (an edge of a Gabriel pair {p,q} enters at (|p-q|/2)^2).  Raw diagrams are
  saved in this convention ("sqradius").
* For plots and distances we use the RADIUS scale r = sqrt(alpha^2), which has
  the unit of the data (g) and is comparable to a Cech radius.  A Rips edge
  length epsilon corresponds roughly to 2r, so Alpha and Rips values must not
  be compared numerically without this factor.
* Homology is computed with coefficients in Z/2Z (GUDHI default field Z2).
"""
from __future__ import annotations

import pickle
import random
import time
from pathlib import Path

import numpy as np

try:
    import gudhi as gd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("GUDHI is required (pip install gudhi)") from exc

# ----------------------------------------------------------------------------- paths
TRACK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TRACK_DIR.parent                      # folder that holds the .pkl files
RESULTS = TRACK_DIR / "results"
PD_DIR = RESULTS / "persistence_diagrams"
DM_DIR = RESULTS / "distance_matrices"
TAB_DIR = RESULTS / "tables"
EVAL_DIR = RESULTS / "evaluation_diagrams"
CACHE_DIR = RESULTS / "cache"
FIG_DIR = TRACK_DIR / "figures"
for _d in (PD_DIR, DM_DIR, TAB_DIR, EVAL_DIR, CACHE_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "psl_week_acc_train_data.pkl"
EVAL_FILE = DATA_DIR / "psl_week_acc_eval_data.pkl"

CLASSES = ["A", "B", "C"]
COLORS = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}   # same colours in every figure
SEED = 2025


def set_seed(seed: int = SEED) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------------- data
def load_train():
    """Return X (240,200,3) float64 and labels y (240,) as str."""
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"training file not found: {TRAIN_FILE}")
    with open(TRAIN_FILE, "rb") as f:
        X, y_raw = pickle.load(f)
    y = np.array([l.decode() if isinstance(l, bytes) else str(l) for l in y_raw])
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[2] != 3 or len(y) != len(X):
        raise ValueError(f"unexpected training shapes {X.shape}, {len(y)}")
    return X, y


def load_eval():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"evaluation file not found: {EVAL_FILE}")
    with open(EVAL_FILE, "rb") as f:
        E = np.asarray(pickle.load(f), dtype=float)
    if E.ndim != 3 or E.shape[2] != 3:
        raise ValueError(f"unexpected evaluation shape {E.shape}")
    return E


def representative_indices(y, per_class: int = 2):
    """Deterministic representatives: evenly spaced inside each class."""
    out = {}
    for c in CLASSES:
        idx = np.where(y == c)[0]
        out[c] = list(idx[np.linspace(0, len(idx) - 1, per_class + 2, dtype=int)[1:-1]])
    return out


# ----------------------------------------------------------------------------- persistence
def alpha_persistence(points: np.ndarray, max_dim: int = 2) -> dict:
    """Alpha complex persistence of a point cloud in R^3.

    The full Delaunay-based alpha filtration is built (vertices, edges,
    triangles, tetrahedra), so H1 classes can die (they are killed by
    triangles) and H2 is available as well.  Coefficients: Z2.

    Returns raw intervals in the SQUARED-radius convention, including
    essential (infinite) intervals, plus simplex counts and timing.
    """
    points = np.asarray(points, dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError("point cloud contains NaN/inf")
    t0 = time.perf_counter()
    ac = gd.AlphaComplex(points=points)
    st = ac.create_simplex_tree()            # default: no max_alpha_square -> full filtration
    st.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    elapsed = time.perf_counter() - t0
    counts = np.zeros(st.dimension() + 1, dtype=int)
    for simplex, _ in st.get_simplices():
        counts[len(simplex) - 1] += 1
    diag = {k: np.asarray(st.persistence_intervals_in_dimension(k), float).reshape(-1, 2)
            for k in range(max_dim + 1)}
    return {"sqradius": diag, "n_vertices": st.num_vertices(), "simplex_counts": counts,
            "seconds": elapsed, "pairs": st.persistence_pairs()}


def to_radius(d: np.ndarray) -> np.ndarray:
    """squared radius -> radius (inf stays inf)."""
    return np.sqrt(np.asarray(d, float))


def finite(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, float).reshape(-1, 2)
    return d[np.isfinite(d[:, 1])]


def lifetimes(d: np.ndarray) -> np.ndarray:
    f = finite(d)
    return f[:, 1] - f[:, 0]


def save_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)                         # atomic write: no half-written cache files


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_radius_diagrams(split: str = "train"):
    """List over samples of {0: H0, 1: H1} in radius scale (essential intervals kept)."""
    folder = PD_DIR if split == "train" else EVAL_DIR
    f = folder / f"{split}_diagrams_sqradius.pkl"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing - run 04_alpha_persistence.py first")
    raw = load_pickle(f)
    return [{k: to_radius(d[k]) for k in (0, 1)} for d in raw]


def savefig(fig, name: str):
    path = FIG_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("  figure ->", path.relative_to(TRACK_DIR))
