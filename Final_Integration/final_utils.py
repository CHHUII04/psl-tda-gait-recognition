"""Shared code for the final integration experiments (P0/P1).

Three topological routes, all rotation invariant:
  A  Alpha complex on the raw cloud {a(t)} in R^3                (radius scale r = sqrt(alpha^2))
  B  Vietoris-Rips on the delay embedding (a(t), a(t+l), a(t+2l)) in R^9, l = 2  (B8: l = 8)
  C  sublevel / superlevel filtration of |a(t)| on the path graph (H0 of valleys / peaks)
Evaluation: blocked & purged 5-fold CV (windows overlap in time, see window_offsets).
"""
import multiprocessing as mp
import pickle
from pathlib import Path

import gudhi as gd
import gudhi.representations as gdr
import numpy as np
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, make_union

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
TAB = HERE / "tables"; TAB.mkdir(exist_ok=True)
CLASSES = ["A", "B", "C"]
COLORS = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}
LAG = 2

ROUTES = {
    "A: Alpha on R^3": ["H0", "H1"],
    "B: Rips on delay R^9": ["H0", "H1"],                    # lag 2 (default of Question 4)
    "B8: Rips on delay R^9, lag 8": ["H0", "H1"],            # lag 8 (best window in the lag sweep, Section 5d)
    "C: sublevel |a(t)|": ["H0 valleys", "H0 peaks"],
}


def key(route):
    """short cache key: 'A', 'B', 'B8', 'C'"""
    return route.split(":")[0]


def load():
    with open(DATA / "psl_week_acc_train_data.pkl", "rb") as f:
        X, y = pickle.load(f)
    with open(DATA / "psl_week_acc_eval_data.pkl", "rb") as f:
        E = pickle.load(f)
    return np.asarray(X, float), np.array([l.decode() for l in y]), np.asarray(E, float)


# ------------------------------------------------------------------ diagrams
def _finite(d):
    d = np.asarray(d, float).reshape(-1, 2)
    return d[np.isfinite(d[:, 1])]


def diag_alpha(w):
    st = gd.AlphaComplex(points=w).create_simplex_tree(); st.compute_persistence()
    return [np.sqrt(_finite(st.persistence_intervals_in_dimension(k))) for k in (0, 1)]


def diag_rips_delay(w, lag=LAG):
    pc = TimeDelayEmbedding(dim=9, delay=lag, skip=1)(w)        # gudhi: dim = total dimension = 3 x 3
    st = gd.RipsComplex(points=pc).create_simplex_tree(max_dimension=2); st.compute_persistence()
    return [_finite(st.persistence_intervals_in_dimension(k)) for k in (0, 1)]


def diag_rips_delay8(w):
    return diag_rips_delay(w, lag=8)


def _sublevel_h0(f):
    st = gd.SimplexTree()
    for i, v in enumerate(f): st.insert([i], v)
    for i in range(len(f) - 1): st.insert([i, i + 1], max(f[i], f[i + 1]))
    st.compute_persistence(); return _finite(st.persistence_intervals_in_dimension(0))


def diag_norm_sublevel(w):
    s = np.linalg.norm(w, axis=1)
    return [_sublevel_h0(s), _sublevel_h0(-s)]                 # [valleys, peaks]


ROUTE_FN = {"A: Alpha on R^3": diag_alpha, "B: Rips on delay R^9": diag_rips_delay,
            "B8: Rips on delay R^9, lag 8": diag_rips_delay8, "C: sublevel |a(t)|": diag_norm_sublevel}


def _job(args):
    name, w = args
    return ROUTE_FN[name](w)


def compute_diagrams(name, windows, cache_name):
    f = CACHE / f"{cache_name}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    with mp.Pool() as pool:
        D = pool.map(_job, [(name, w) for w in windows], chunksize=4)
    pickle.dump(D, open(f, "wb"))
    return D


# ------------------------------------------------------------------ blocked & purged CV
def window_offsets(W):
    """start time of each window in its recording, recovered from samples shared by overlapping windows."""
    loc = {}
    for i in range(len(W)):
        for t in range(W.shape[1]):
            loc.setdefault(tuple(W[i, t]), []).append((i, t))
    pos, stack = {0: 0}, [0]
    while stack:
        i = stack.pop()
        for t in range(W.shape[1]):
            for j, u in loc[tuple(W[i, t])]:
                if j not in pos:
                    pos[j] = pos[i] + t - u; stack.append(j)
    o = np.array([pos[i] for i in range(len(W))]); return o - o.min()


def blocked_purged_splits(X, y, n_blocks=5):
    off = np.zeros(len(y), int); fold = np.zeros(len(y), int)
    for c in CLASSES:
        m = y == c; off[m] = window_offsets(X[m])
        fold[m] = np.minimum(off[m] * n_blocks // (off[m].max() + 1), n_blocks - 1)
    splits = []
    for g in range(n_blocks):
        te = fold == g; tr = ~te
        for c in CLASSES:
            m = y == c; lo, hi = off[te & m].min(), off[te & m].max() + 200
            tr &= ~(m & (off + 200 > lo) & (off < hi))        # purge training windows overlapping the test block
        splits.append((np.where(tr)[0], np.where(te)[0]))
    return splits


# ------------------------------------------------------------------ features / models
class DimSelect(BaseEstimator, TransformerMixin):
    def __init__(self, k=0): self.k = k
    def fit(self, D, y=None): return self
    def transform(self, D): return [d[self.k] for d in D]


def pers_weight(x): return x[1] - x[0]


def vectorizer(kind):
    if kind == "landscape": return gdr.Landscape(num_landscapes=5, resolution=100)
    if kind == "silhouette": return gdr.Silhouette(weight=pers_weight, resolution=100)
    raise ValueError(kind)


def model(dims, kind="landscape", seed=0):
    feats = make_union(*[make_pipeline(DimSelect(k), gdr.DiagramSelector(use=True), vectorizer(kind)) for k in dims])
    return make_pipeline(feats, RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))


def cv_scores(D, y, splits, dims, kind="landscape"):
    accs = []
    for tr, te in splits:
        m = model(dims, kind).fit([D[i] for i in tr], y[tr])
        accs.append((m.predict([D[i] for i in te]) == y[te]).mean())
    return np.array(accs)


def random_rotations(n, seed):
    from scipy.spatial.transform import Rotation
    return Rotation.random(n, random_state=seed).as_matrix()
