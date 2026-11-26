"""02 - Raw data visualisation: XYZ series, |a(t)| series, 3D trajectories (2 representatives per class)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CLASSES, COLORS, load_train, representative_indices, savefig


def main():
    X, y = load_train()
    reps = representative_indices(y, per_class=2)
    print("representatives:", reps)
    t = np.arange(X.shape[1])

    # 1) XYZ components
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex=True, sharey=True)
    for r, c in enumerate(CLASSES):
        for j, i in enumerate(reps[c]):
            for k, (ax_name, col) in enumerate(zip("xyz", ["#444444", "#ff7f0e", "#9467bd"])):
                axes[r, j].plot(t, X[i, :, k], lw=1, color=col, label=f"$a_{ax_name}$")
            axes[r, j].set_title(f"person {c} - window #{i}", fontsize=10)
    axes[0, 0].legend(ncol=3, fontsize=8)
    for a in axes[:, 0]: a.set_ylabel("acceleration (g)")
    for a in axes[-1]: a.set_xlabel("time index")
    fig.suptitle("Accelerometer components in the sensor frame (not comparable across persons)")
    fig.tight_layout(); savefig(fig, "02_xyz_timeseries.png"); plt.close(fig)

    # 2) magnitude |a(t)| = sqrt(ax^2+ay^2+az^2): rotation invariant
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, sharey=True)
    for r, c in enumerate(CLASSES):
        for i, ls in zip(reps[c], ["-", "--"]):
            axes[r].plot(t, np.linalg.norm(X[i], axis=1), ls, color=COLORS[c], lw=1, label=f"#{i}")
        axes[r].set_ylabel(f"{c}: |a| (g)"); axes[r].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time index")
    fig.suptitle("Acceleration magnitude (rotation invariant)")
    fig.tight_layout(); savefig(fig, "02_norm_timeseries.png"); plt.close(fig)

    # 3) 3D trajectories, consecutive points connected; centred so that shapes are comparable
    fig = plt.figure(figsize=(15, 9))
    lim = max(np.abs(X[i] - X[i].mean(0)).max() for c in CLASSES for i in reps[c])
    for r, c in enumerate(CLASSES):
        for j, i in enumerate(reps[c]):
            ax = fig.add_subplot(2, 3, j * 3 + r + 1, projection="3d")
            P = X[i] - X[i].mean(0)
            ax.plot(*P.T, color=COLORS[c], lw=.7)
            ax.scatter(*P.T, color=COLORS[c], s=3)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_title(f"person {c} - window #{i}", fontsize=10)
    fig.suptitle("3D acceleration trajectories (centred, common scale)")
    fig.tight_layout(); savefig(fig, "02_3d_trajectories.png"); plt.close(fig)

    # 4) side-by-side comparison: one window per class overlaid, three 2D projections
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for c in CLASSES:
        P = X[reps[c][0]] - X[reps[c][0]].mean(0)
        for a, (u, v) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
            a.plot(P[:, u], P[:, v], color=COLORS[c], lw=.8, alpha=.8, label=c)
    for a, (u, v) in zip(axes, ["xy", "xz", "yz"]):
        a.set_xlabel(f"$a_{u}$ - mean"); a.set_ylabel(f"$a_{v}$ - mean"); a.set_aspect("equal"); a.legend()
    fig.suptitle("Centred point clouds of A, B, C side by side (sensor frames differ: compare shape/size, not direction)")
    fig.tight_layout(); savefig(fig, "02_classes_side_by_side.png"); plt.close(fig)


if __name__ == "__main__":
    main()
