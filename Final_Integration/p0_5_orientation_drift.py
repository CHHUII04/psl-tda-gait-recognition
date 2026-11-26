"""P0-5  Orientation DRIFT inside a window (phone turning in the pocket).

Routes A (Alpha) and B (delay Rips) are invariant to a CONSTANT rotation of the window (they only use distances),
route C is invariant to ANY time-dependent rotation Q(t), since |Q(t) a(t)| = |a(t)|.
Test: models trained on the data as recorded; test windows (blocked & purged CV) receive a rotation whose angle grows
linearly from 0 to theta_max during the window, around a random axis:  a(t) -> R(axis, theta_max * t/199) a(t).
Also: estimate the drift present in evaluation window #14.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from final_utils import COLORS, FIG, TAB, blocked_purged_splits, compute_diagrams, key, load, model

ROUTES_T = {"A: Alpha on R^3": "A (Alpha, R^3)", "B8: Rips on delay R^9, lag 8": "B (delay Rips, lag 8)",
            "C: sublevel |a(t)|": "C (|a(t)| sublevel)"}
ANGLES = [0, 15, 30, 45, 60, 90]


def drift(W, theta_deg, seed):
    rng = np.random.default_rng(seed)
    out = np.empty_like(W)
    for n, w in enumerate(W):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        ang = np.deg2rad(theta_deg) * np.arange(len(w)) / (len(w) - 1)
        R = Rotation.from_rotvec(ang[:, None] * axis).as_matrix()           # one rotation per time step
        out[n] = np.einsum("tij,tj->ti", R, w)
    return out


def main():
    X, y, E = load()
    splits = blocked_purged_splits(X, y)
    rows = []
    for theta in ANGLES:
        Xd = X if theta == 0 else drift(X, theta, seed=theta)
        assert np.allclose(np.linalg.norm(Xd, axis=2), np.linalg.norm(X, axis=2))
        for r, short in ROUTES_T.items():
            D0 = compute_diagrams(r, X, "orig_" + key(r) + "_train")
            Dd = D0 if theta == 0 else compute_diagrams(r, Xd, f"drift{theta}_" + key(r))
            acc, predC = [], []
            for tr, te in splits:
                m = model((0, 1), "silhouette").fit([D0[i] for i in tr], y[tr])
                p = m.predict([Dd[i] for i in te])
                acc.append((p == y[te]).mean()); predC.append((p == "C").mean())
            rows.append({"drift angle (deg)": theta, "route": short, "accuracy": np.mean(acc),
                         "fraction predicted C": np.mean(predC)})
            print(f"theta={theta:3d}  {short:24s} acc={np.mean(acc):.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p0_5_orientation_drift.csv", index=False, float_format="%.3f")
    print(df.pivot(index="drift angle (deg)", columns="route", values="accuracy").round(3).to_string())

    # drift present in window #14: angle between the mean (gravity) directions of its first 25 and last 100 samples
    w = E[13]
    g1, g2 = w[:25].mean(0), w[100:].mean(0)
    ang14 = np.degrees(np.arccos(g1 @ g2 / np.linalg.norm(g1) / np.linalg.norm(g2)))
    ref = []
    for c in "ABC":                                                         # same statistic on training windows
        W = X[y == c]
        a = [np.degrees(np.arccos(np.clip(u[:25].mean(0) @ u[100:].mean(0) / np.linalg.norm(u[:25].mean(0)) /
                                          np.linalg.norm(u[100:].mean(0)), -1, 1))) for u in W]
        ref.append({"class": c, "median": np.median(a), "95%": np.percentile(a, 95), "max": np.max(a)})
    ref = pd.DataFrame(ref); ref.to_csv(TAB / "p0_5_gravity_angle_reference.csv", index=False, float_format="%.2f")
    print(f"\nwindow #14: angle between gravity directions (first 25 vs last 100 samples) = {ang14:.1f} deg")
    print("same statistic on training windows:\n", ref.round(1).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for r, short in ROUTES_T.items():
        sub = df[df.route == short]
        c = {"A": "#9467bd", "B": "#ff7f0e", "C": "#17becf"}[short[0]]
        axes[0].plot(sub["drift angle (deg)"], sub.accuracy, "o-", c=c, label=short)
        axes[1].plot(sub["drift angle (deg)"], sub["fraction predicted C"], "o-", c=c, label=short)
    axes[0].axhline(1 / 3, ls=":", c="k"); axes[0].set_ylabel("blocked CV accuracy"); axes[0].set_ylim(0, 1.05)
    axes[1].axhline(1 / 3, ls=":", c="k"); axes[1].set_ylabel("fraction of test windows predicted 'C'")
    for a in axes:
        a.set_xlabel("orientation drift during the window (deg)"); a.axvline(ang14, c="k", lw=1.5, ls="--"); a.legend(fontsize=8)
    axes[0].text(ang14 + 1, .05, f"#14: {ang14:.0f} deg", fontsize=9)
    axes[0].set_title("phone turning in the pocket: only |a(t)| is invariant to a time-dependent rotation", fontsize=10)
    axes[1].set_title("drifting windows look like the largest cloud (person C) for routes A/B", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "p0_5_orientation_drift.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
