"""03 - Rigid-motion invariance of the Alpha filtration.

For a few representative windows we apply
  * random proper rotations  Q in SO(3)  (X_rot = X @ Q)
  * one improper orthogonal map (reflection, det Q = -1)
  * translations X + v
and compare (i) pairwise Euclidean distance matrices and (ii) Alpha persistence
diagrams (bottleneck distance, radius scale).  Output: results/tables/rotation_invariance.csv
"""
import gudhi as gd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from common import (CLASSES, COLORS, SEED, TAB_DIR, alpha_persistence, finite, load_train,
                    representative_indices, savefig, set_seed, to_radius)

N_ROT = 5


def compare(P, P2, ref_diag):
    """distance-matrix difference + bottleneck distance between alpha diagrams (radius scale)."""
    ddist = np.abs(pdist(P) - pdist(P2)).max()          # pairwise Euclidean distances must be preserved
    d2 = alpha_persistence(P2)["sqradius"]
    out = {"max_abs_diff_distance_matrix": ddist}
    for k in (0, 1):
        a, b = to_radius(ref_diag[k]), to_radius(d2[k])
        out[f"bottleneck_H{k}"] = gd.bottleneck_distance(a, b)          # essential points included
        out[f"n_points_H{k}_orig"], out[f"n_points_H{k}_new"] = len(a), len(b)
        # direct comparison of the sorted finite lifetimes (stronger than bottleneck = 0)
        la, lb = np.sort(np.diff(finite(a), axis=1).ravel()), np.sort(np.diff(finite(b), axis=1).ravel())
        out[f"max_abs_diff_sorted_lifetimes_H{k}"] = np.abs(la - lb).max() if len(la) == len(lb) else np.nan
    return out


def main():
    rng = set_seed(SEED)
    X, y = load_train()
    reps = representative_indices(y, per_class=2)
    rows = []
    for c in CLASSES:
        for i in reps[c]:
            P = X[i]
            ref = alpha_persistence(P)["sqradius"]
            # proper rotations, uniformly distributed on SO(3)
            for r in range(N_ROT):
                Q = Rotation.random(random_state=int(rng.integers(1e9))).as_matrix()
                rows.append({"sample": i, "label": c, "transform": f"rotation_{r}", "det": np.linalg.det(Q),
                             **compare(P, P @ Q, ref)})
            # improper orthogonal map: rotation followed by a reflection
            Qr = Rotation.random(random_state=int(rng.integers(1e9))).as_matrix() @ np.diag([1, 1, -1])
            rows.append({"sample": i, "label": c, "transform": "reflection", "det": np.linalg.det(Qr),
                         **compare(P, P @ Qr, ref)})
            # translations: random vector, and centring (removes the gravity offset)
            v = rng.normal(0, 5, size=3)
            rows.append({"sample": i, "label": c, "transform": "translation", "det": 1.0, **compare(P, P + v, ref)})
            rows.append({"sample": i, "label": c, "transform": "centring", "det": 1.0, **compare(P, P - P.mean(0), ref)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "rotation_invariance.csv", index=False, float_format="%.3e")
    cols = ["max_abs_diff_distance_matrix", "bottleneck_H0", "bottleneck_H1",
            "max_abs_diff_sorted_lifetimes_H0", "max_abs_diff_sorted_lifetimes_H1"]
    summ = df.assign(kind=df["transform"].str.replace(r"_\d+", "", regex=True)).groupby("kind")[cols].agg(["max", "mean"])
    summ.to_csv(TAB_DIR / "rotation_invariance_summary.csv", float_format="%.3e")
    print(summ.to_string(float_format=lambda v: f"{v:.2e}"))
    print("all point counts preserved:", bool((df.n_points_H0_orig == df.n_points_H0_new).all()
                                             and (df.n_points_H1_orig == df.n_points_H1_new).all()))

    # figure: one window, original vs rotated cloud, and overlaid diagrams
    i = reps["B"][0]; P = X[i]
    Q = Rotation.random(random_state=SEED).as_matrix(); P2 = P @ Q
    d1, d2 = alpha_persistence(P)["sqradius"], alpha_persistence(P2)["sqradius"]
    fig = plt.figure(figsize=(16, 4.6))
    for j, (pts, ttl) in enumerate([(P, "original (sensor frame)"), (P2, "randomly rotated  X @ Q")]):
        ax = fig.add_subplot(1, 4, j + 1, projection="3d"); ax.plot(*pts.T, color=COLORS["B"], lw=.6)
        ax.set_title(f"window #{i}: {ttl}", fontsize=9)
    for k in (0, 1):
        ax = fig.add_subplot(1, 4, 3 + k)
        a, b = finite(to_radius(d1[k])), finite(to_radius(d2[k]))
        ax.scatter(a[:, 0], a[:, 1], s=30, facecolors="none", edgecolors="k", label="original")
        ax.scatter(b[:, 0], b[:, 1], s=6, c="tab:orange", label="rotated")
        m = max(a.max(), b.max()) * 1.05; ax.plot([0, m], [0, m], "k--", lw=.7)
        ax.set_xlabel("birth radius r = sqrt(alpha^2)  [g]"); ax.set_ylabel("death radius [g]")
        ax.set_title(f"H{k}: bottleneck = {gd.bottleneck_distance(to_radius(d1[k]), to_radius(d2[k])):.1e}", fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout(); savefig(fig, "03_rotation_invariance.png"); plt.close(fig)


if __name__ == "__main__":
    main()
