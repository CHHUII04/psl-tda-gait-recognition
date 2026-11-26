"""P0-1  Same protocol for the 3 routes x {H0, H1, H0+H1}: landscape (+ silhouette) -> random forest, blocked & purged CV."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_utils import (FIG, ROUTES, TAB, blocked_purged_splits, compute_diagrams, cv_scores, load)


def main():
    X, y, E = load()
    XE = np.concatenate([X, E])
    splits = blocked_purged_splits(X, y)
    rows = []
    for route, dnames in ROUTES.items():
        t0 = time.time()
        D = compute_diagrams(route, XE, "orig_" + route[0])[:240]
        print(f"{route}: diagrams ready ({time.time() - t0:.0f}s)", flush=True)
        for dims, lab in [((0,), dnames[0]), ((1,), dnames[1]), ((0, 1), "both")]:
            for kind in ("landscape", "silhouette"):
                acc = cv_scores(D, y, splits, dims, kind)
                rows.append({"route": route, "homology": lab, "vectorization": kind,
                             "accuracy": acc.mean(), "fold_std": acc.std(), "fold_min": acc.min()})
                print(f"   {lab:11s} {kind:10s} acc={acc.mean():.3f} ± {acc.std():.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p0_1_route_x_homology.csv", index=False, float_format="%.4f")

    # figure: heatmap per vectorization, rows = route, cols = H0 / H1 / both
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, kind in zip(axes, ("landscape", "silhouette")):
        sub = df[df.vectorization == kind]
        Z = np.array([[sub[(sub.route == r)].iloc[j].accuracy for j in range(3)] for r in ROUTES])
        S = np.array([[sub[(sub.route == r)].iloc[j].fold_std for j in range(3)] for r in ROUTES])
        im = ax.imshow(Z, cmap="RdYlGn", vmin=0.7, vmax=1.0)
        for i, r in enumerate(ROUTES):
            labs = list(ROUTES[r]) + ["both"]
            for j in range(3):
                ax.text(j, i, f"{labs[j]}\n{Z[i, j]:.3f} ± {S[i, j]:.3f}", ha="center", va="center", fontsize=8.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(["first diagram", "second diagram", "both"])
        ax.set_yticks(range(3)); ax.set_yticklabels(list(ROUTES))
        ax.set_title(f"{kind} + random forest  (blocked & purged 5-fold CV)", fontsize=10)
    plt.colorbar(im, ax=axes, fraction=.02)
    fig.savefig(FIG / "p0_1_route_x_homology.png", dpi=200, bbox_inches="tight")
    print(df.pivot_table(index="route", columns=["vectorization", "homology"], values="accuracy").round(3).T.to_string())


if __name__ == "__main__":
    main()
