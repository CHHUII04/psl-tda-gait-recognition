"""Helpers that must live in a module so that multiprocessing works on Windows
(functions defined inside a notebook cannot be sent to worker processes)."""
import numpy as np
import multiprocessing as mp
import gudhi as gd
from gudhi.hera import wasserstein_distance
from gudhi.point_cloud.timedelay import TimeDelayEmbedding


# ---------- pairwise distances between persistence diagrams ----------
def _pair(args):
    i, j, d1, d2, metric = args
    if metric == "bottleneck":
        return i, j, gd.bottleneck_distance(d1, d2, 1e-3)
    if metric == "W1":
        return i, j, wasserstein_distance(d1, d2, order=1, internal_p=2)
    if metric == "W2":
        return i, j, wasserstein_distance(d1, d2, order=2, internal_p=2)
    raise ValueError(metric)


def distance_matrix(diags, metric="bottleneck", processes=None):
    """Symmetric matrix of pairwise distances ('bottleneck', 'W1' or 'W2')."""
    n = len(diags)
    M = np.zeros((n, n))
    jobs = [(i, j, diags[i], diags[j], metric) for i in range(n) for j in range(i)]
    with mp.Pool(processes) as pool:
        for i, j, v in pool.imap_unordered(_pair, jobs, chunksize=200):
            M[i, j] = M[j, i] = v
    return M


# ---------- time-delay embeddings ----------
def _finite(d):
    d = np.asarray(d, float).reshape(-1, 2)
    return d[np.isfinite(d[:, 1])]


def rips_td_diagrams(w, m=3, delay=2, skip=1):
    """Rips persistence (H0, H1) of the multivariate delay embedding
    (a(t), a(t+delay), ..., a(t+(m-1)delay)) in R^(3m).
    NB: in gudhi, `dim` is the TOTAL output dimension, i.e. 3*m for a 3D series."""
    pc = TimeDelayEmbedding(dim=w.shape[1] * m, delay=delay, skip=skip)(w)
    st = gd.RipsComplex(points=pc).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    return [_finite(st.persistence_intervals_in_dimension(k)) for k in (0, 1)]


def norm_takens_diagrams(w, dim=3, delay=5):
    """Alpha persistence (H0, H1) of the Takens embedding of |a(t)| in R^dim."""
    s = np.linalg.norm(w, axis=1)
    pc = TimeDelayEmbedding(dim=dim, delay=delay, skip=1)(s)
    st = gd.AlphaComplex(points=pc).create_simplex_tree()
    st.compute_persistence()
    return [np.sqrt(_finite(st.persistence_intervals_in_dimension(k))) for k in (0, 1)]


def _rips_job(args):
    return rips_td_diagrams(*args)


def _norm_job(args):
    return norm_takens_diagrams(*args)


def parallel_rips_td(windows, m, delay, skip=1, processes=None):
    with mp.Pool(processes) as pool:
        return pool.map(_rips_job, [(w, m, delay, skip) for w in windows], chunksize=8)


def parallel_norm_takens(windows, dim, delay, processes=None):
    with mp.Pool(processes) as pool:
        return pool.map(_norm_job, [(w, dim, delay) for w in windows], chunksize=8)
