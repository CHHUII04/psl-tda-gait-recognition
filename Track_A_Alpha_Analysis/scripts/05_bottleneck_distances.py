"""05 - Pairwise bottleneck distances between the 240 training Alpha diagrams, separately for H0 and H1.

* radius scale r = sqrt(alpha^2)
* exact bottleneck distance (gudhi.bottleneck_distance without approximation parameter)
* essential intervals are KEPT: every H0 diagram has exactly one essential point (0, inf) and GUDHI matches
  essential points separately, so they contribute 0 here; this is checked explicitly below.
* parallel computation by rows, partial results cached in results/cache/ (resumable), progress printed.
Outputs: results/distance_matrices/bottleneck_H0.npy, bottleneck_H1.npy (+ .csv with labels), sample_order.csv
"""
import multiprocessing as mp
import sys
import time

import gudhi as gd
import numpy as np
import pandas as pd

from common import CACHE_DIR, DM_DIR, finite, load_radius_diagrams, load_train


def _row(args):
    """distances d(i, j) for all j < i (lower triangle row i)."""
    i, diags = args
    return i, np.array([gd.bottleneck_distance(diags[i], diags[j]) for j in range(i)])


def bottleneck_matrix(diags, name, processes=None):
    n = len(diags)
    cache = CACHE_DIR / f"{name}_partial.npz"
    M = np.full((n, n), np.nan); np.fill_diagonal(M, 0.0)
    done = np.zeros(n, bool); done[0] = True
    if cache.exists():                                   # resume from a previous interrupted run
        z = np.load(cache); M, done = z["M"], z["done"]
        print(f"  resuming {name}: {done.sum()}/{n} rows already computed")
    todo = [i for i in range(n) if not done[i]]
    t0 = time.time(); total = sum(todo); finished = 0; next_report = 0.1
    with mp.Pool(processes) as pool:
        for k, (i, row) in enumerate(pool.imap_unordered(_row, [(i, diags) for i in todo[::-1]], chunksize=1)):
            M[i, :i] = row; M[:i, i] = row; done[i] = True
            finished += i
            if total and finished / total >= next_report:
                print(f"  {name}: {100 * finished / total:5.1f}% of pairs  ({time.time() - t0:.0f}s)")
                next_report += 0.1
            if k % 40 == 39:
                np.savez(cache, M=M, done=done)          # intermediate cache
    np.savez(cache, M=M, done=done)
    return M


def validate(M, name):
    assert M.shape == (240, 240), name
    assert np.isfinite(M).all(), f"{name}: non-finite entries"
    assert np.allclose(M, M.T), f"{name}: not symmetric"
    assert np.all(np.diag(M) == 0), f"{name}: non-zero diagonal"
    off = M[~np.eye(len(M), dtype=bool)]
    print(f"  {name}: symmetric, zero diagonal, finite; off-diagonal min={off.min():.4f} "
          f"median={np.median(off):.4f} max={off.max():.4f}; #zero off-diag={int((off == 0).sum())}")


def main():
    force = "--force" in sys.argv
    X, y = load_train()
    diags = load_radius_diagrams("train")
    assert len(diags) == len(y) == 240

    # essential-interval check: same number (1) and same birth (0) in every H0 diagram -> contribute 0
    ess = [d[0][np.isinf(d[0][:, 1])] for d in diags]
    assert all(len(e) == 1 and e[0, 0] == 0 for e in ess)
    assert all(not np.isinf(d[1][:, 1]).any() for d in diags)
    rng = np.random.default_rng(0)
    for i, j in rng.integers(0, 240, size=(5, 2)):
        a = gd.bottleneck_distance(diags[i][0], diags[j][0]); b = gd.bottleneck_distance(finite(diags[i][0]), finite(diags[j][0]))
        assert abs(a - b) < 1e-12
    print("essential intervals: one (0, inf) per H0 diagram; bottleneck with/without it identical (checked)")

    pd.DataFrame({"row": range(240), "label": y}).to_csv(DM_DIR / "sample_order.csv", index=False)
    for k in (0, 1):
        out = DM_DIR / f"bottleneck_H{k}.npy"
        if out.exists() and not force:
            print(f"H{k}: cached -> {out.name}"); M = np.load(out)
        else:
            print(f"H{k}: computing 240x240 bottleneck matrix ...")
            M = bottleneck_matrix([d[k] for d in diags], f"bottleneck_H{k}")
            np.save(out, M)
        validate(M, f"bottleneck_H{k}")
        pd.DataFrame(M, index=[f"{l}_{i}" for i, l in enumerate(y)], columns=[f"{l}_{i}" for i, l in enumerate(y)]) \
          .to_csv(DM_DIR / f"bottleneck_H{k}.csv", float_format="%.6f")
    print("saved -> results/distance_matrices/")


if __name__ == "__main__":
    main()
