"""07 - Robustness of the Alpha diagrams.

9.1  random rotations: one uniformly random rotation Q in SO(3) for EVERY training window (240 windows),
     bottleneck(original, rotated) for H0 and H1; combined with the detailed test of script 03.
9.2  point-cloud subsampling: 4 windows per class, 5 random subsamples of 80% of the points each,
     bottleneck(full, subsample) for H0 and H1, compared with the typical within/between-class distances.
     NB: this removes points from the cloud; it is NOT a change of time window / delay.
"""
import multiprocessing as mp

import gudhi as gd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from common import (CLASSES, COLORS, DM_DIR, SEED, TAB_DIR, alpha_persistence, finite, lifetimes, load_radius_diagrams,
                    load_train, savefig, set_seed, to_radius)

N_PER_CLASS, N_REP, FRAC = 4, 5, 0.8


def _rotation_job(args):
    i, P, seed, ref = args
    Q = Rotation.random(random_state=seed).as_matrix()
    d = alpha_persistence(P @ Q)["sqradius"]
    return {"sample": i, **{f"bottleneck_H{k}": gd.bottleneck_distance(ref[k], to_radius(d[k])) for k in (0, 1)}}


def _subsample_job(args):
    i, r, P, idx, ref = args
    d = alpha_persistence(P[idx])["sqradius"]
    out = {"sample": i, "repeat": r, "n_points": len(idx)}
    for k in (0, 1):
        dk = to_radius(d[k])
        out[f"bottleneck_H{k}"] = gd.bottleneck_distance(ref[k], dk)
        out[f"max_pers_H{k}_full"] = lifetimes(ref[k]).max()
        out[f"max_pers_H{k}_sub"] = lifetimes(dk).max()
        out[f"n_H{k}_sub"] = len(finite(dk))
    return out


def main():
    rng = set_seed(SEED)
    X, y = load_train()
    diags = load_radius_diagrams("train")
    M = {k: np.load(DM_DIR / f"bottleneck_H{k}.npy") for k in (0, 1)}

    # ---------------- 9.1 rotations on all 240 windows ----------------
    seeds = rng.integers(0, 2**31 - 1, size=len(X))
    with mp.Pool() as pool:
        rot = pd.DataFrame(pool.map(_rotation_job, [(i, X[i], int(seeds[i]), diags[i]) for i in range(len(X))]))
    rot["label"] = y
    rot.to_csv(TAB_DIR / "robustness_rotation_all_windows.csv", index=False, float_format="%.3e")
    rs = rot[["bottleneck_H0", "bottleneck_H1"]].agg(["max", "mean"])
    rs.to_csv(TAB_DIR / "robustness_rotation_summary.csv", float_format="%.3e")
    print("9.1 random rotation of all 240 windows - bottleneck(original, rotated):")
    print(rs.to_string(float_format=lambda v: f"{v:.2e}"))

    # ---------------- 9.2 subsampling ----------------
    subset = np.concatenate([rng.choice(np.where(y == c)[0], N_PER_CLASS, replace=False) for c in CLASSES])
    jobs = []
    for i in subset:
        for r in range(N_REP):
            idx = np.sort(rng.choice(X.shape[1], int(FRAC * X.shape[1]), replace=False))
            jobs.append((int(i), r, X[i], idx, diags[i]))
    with mp.Pool() as pool:
        sub = pd.DataFrame(pool.map(_subsample_job, jobs))
    sub["label"] = y[sub["sample"].values]
    sub.to_csv(TAB_DIR / "robustness_subsampling.csv", index=False, float_format="%.5f")

    # reference scales from the distance matrices
    ref = {}
    for k in (0, 1):
        same = y[:, None] == y[None, :]; off = ~np.eye(len(y), dtype=bool)
        ref[f"H{k}"] = {"median within-class d_B": np.median(M[k][same & off]),
                        "median between-class d_B": np.median(M[k][~same])}
    summ = []
    for k in (0, 1):
        v = sub[f"bottleneck_H{k}"]
        summ.append({"homology": f"H{k}", "mean d_B(full, 80%)": v.mean(), "median": v.median(), "max": v.max(),
                     **ref[f"H{k}"],
                     "mean ratio to within-class median": v.mean() / ref[f"H{k}"]["median within-class d_B"],
                     "mean max-persistence full": sub[f"max_pers_H{k}_full"].mean(),
                     "mean max-persistence 80%": sub[f"max_pers_H{k}_sub"].mean()})
    summ = pd.DataFrame(summ).set_index("homology")
    summ.to_csv(TAB_DIR / "robustness_subsampling_summary.csv", float_format="%.5f")
    print("\n9.2 subsampling (80% of points, 5 repeats, 12 windows):")
    print(summ.round(4).T.to_string())
    print("\nper class mean d_B(full, 80%):")
    print(sub.groupby("label")[["bottleneck_H0", "bottleneck_H1"]].mean().round(4).to_string())

    # ---------------- noise band (heuristic, in the spirit of subsampling confidence sets, Fasy et al. 2014) ----------------
    # c = 95% quantile of d_B(full, subsample).  A point is called "significant" if persistence > 2c, i.e. it cannot be
    # matched to the diagonal by a perturbation of size c.  NB: 80% subsamples without replacement, 12 windows only ->
    # this is an indicative threshold, not a calibrated confidence band.
    sig_rows = []
    for k in (0, 1):
        c_k = sub[f"bottleneck_H{k}"].quantile(0.95)
        for c in CLASSES:
            idx = np.where(y == c)[0]
            n_sig = np.array([(lifetimes(diags[i][k]) > 2 * c_k).sum() for i in idx])
            sig_rows.append({"homology": f"H{k}", "c_95": c_k, "threshold_2c": 2 * c_k, "class": c,
                             "mean_n_significant_points": n_sig.mean(), "frac_windows_with_>=1": (n_sig >= 1).mean()})
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(TAB_DIR / "significant_features_noise_band.csv", index=False, float_format="%.4f")
    print("\nnoise band from subsampling (points with persistence > 2c):")
    print(sig.round(4).to_string(index=False))

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.5))
    for ax, k in zip(axes[:2], (0, 1)):
        data = [sub.loc[sub.label == c, f"bottleneck_H{k}"] for c in CLASSES]
        bp = ax.boxplot(data, labels=CLASSES, patch_artist=True)
        for patch, c in zip(bp["boxes"], CLASSES): patch.set_facecolor(COLORS[c]); patch.set_alpha(.5)
        ax.axhline(ref[f"H{k}"]["median within-class d_B"], c="k", ls="--", lw=1, label="median within-class d_B")
        ax.axhline(ref[f"H{k}"]["median between-class d_B"], c="k", ls=":", lw=1, label="median between-class d_B")
        ax.set_title(f"H{k}: d_B(full diagram, 80% subsample)"); ax.set_ylabel("d_B [g]"); ax.legend(fontsize=8)
    # overlay of diagrams for one window
    i = int(subset[0]); P = X[i]
    ax = axes[2]
    f = finite(diags[i][1]); ax.scatter(f[:, 0], f[:, 1], s=30, facecolors="none", edgecolors="k", label="full (200 pts)")
    for r in range(3):
        idx = np.sort(np.random.default_rng(r).choice(200, 160, replace=False))
        g = finite(to_radius(alpha_persistence(P[idx])["sqradius"][1]))
        ax.scatter(g[:, 0], g[:, 1], s=6, alpha=.6, label=f"80% subsample {r}")
    m = f[:, 1].max() * 1.1; ax.plot([0, m], [0, m], "k--", lw=.7)
    ax.set_xlabel("birth radius [g]"); ax.set_ylabel("death radius [g]"); ax.legend(fontsize=8)
    ax.set_title(f"H1 of window #{i} ({y[i]}): full vs subsampled")
    fig.tight_layout(); savefig(fig, "07_subsampling_robustness.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rot.bottleneck_H0, bins=30, alpha=.6, label="H0"); ax.hist(rot.bottleneck_H1, bins=30, alpha=.6, label="H1")
    ax.set_xlabel("d_B(original, rotated)"); ax.set_title("random rotation of all 240 windows"); ax.legend()
    fig.tight_layout(); savefig(fig, "07_rotation_all_windows.png"); plt.close(fig)


if __name__ == "__main__":
    main()
