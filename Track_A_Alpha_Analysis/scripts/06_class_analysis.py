"""06 - Class-wise analysis of the bottleneck matrices (no classifier): heatmaps, within/between-class
distances, silhouette scores, 2D metric MDS.  Outputs in results/tables and figures/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_samples, silhouette_score

from common import CLASSES, COLORS, DM_DIR, SEED, TAB_DIR, load_train, savefig

PAIRS = [("A", "A"), ("B", "B"), ("C", "C"), ("A", "B"), ("A", "C"), ("B", "C")]


def block_values(M, y, c1, c2):
    """all distances between class c1 and class c2 (upper triangle only when c1 == c2, no diagonal)."""
    i1, i2 = np.where(y == c1)[0], np.where(y == c2)[0]
    B = M[np.ix_(i1, i2)]
    return B[np.triu_indices(len(i1), 1)] if c1 == c2 else B.ravel()


def main():
    X, y = load_train()
    order = np.argsort(y, kind="stable")               # sort by label A, B, C (keeps the original order inside a class)
    y_s = y[order]
    mats = {k: np.load(DM_DIR / f"bottleneck_H{k}.npy")[np.ix_(order, order)] for k in (0, 1)}
    bounds = [np.searchsorted(y_s, c) for c in CLASSES[1:]]

    # ---------------- heatmaps ----------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.3))
    for ax, (k, M) in zip(axes, mats.items()):
        im = ax.imshow(M, cmap="viridis", interpolation="nearest")
        for b in bounds:
            ax.axhline(b - .5, c="w", lw=1.2); ax.axvline(b - .5, c="w", lw=1.2)
        centres = [np.mean(np.where(y_s == c)[0]) for c in CLASSES]
        ax.set_xticks(centres); ax.set_xticklabels(CLASSES); ax.set_yticks(centres); ax.set_yticklabels(CLASSES)
        ax.set_title(f"Bottleneck distance between Alpha diagrams - H{k}  (radius scale, g)")
        plt.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout(); savefig(fig, "06_bottleneck_heatmaps.png"); plt.close(fig)

    # ---------------- within / between class statistics ----------------
    rows, long = [], []
    for k, M in mats.items():
        r = {"homology": f"H{k}"}
        for c1, c2 in PAIRS:
            v = block_values(M, y_s, c1, c2)
            r[f"{c1}-{c2} mean"], r[f"{c1}-{c2} median"], r[f"{c1}-{c2} std"] = v.mean(), np.median(v), v.std()
            long += [{"homology": f"H{k}", "pair": f"{c1}-{c2}", "kind": "within" if c1 == c2 else "between", "d": x} for x in v]
        w = np.concatenate([block_values(M, y_s, c, c) for c in CLASSES])
        b = np.concatenate([block_values(M, y_s, c1, c2) for c1, c2 in PAIRS if c1 != c2])
        r.update({"within mean": w.mean(), "between mean": b.mean(), "between/within ratio": b.mean() / w.mean(),
                  "silhouette (all)": silhouette_score(M, y_s, metric="precomputed")})
        s = silhouette_samples(M, y_s, metric="precomputed")
        for c in CLASSES:
            r[f"silhouette {c}"] = s[y_s == c].mean()
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("homology")
    tab.to_csv(TAB_DIR / "classwise_distances.csv", float_format="%.5f")
    mean_tab = tab[[f"{a}-{b} mean" for a, b in PAIRS]].rename(columns=lambda s: s.replace(" mean", ""))
    med_tab = tab[[f"{a}-{b} median" for a, b in PAIRS]].rename(columns=lambda s: s.replace(" median", ""))
    mean_tab.to_csv(TAB_DIR / "classwise_mean_distance.csv", float_format="%.4f")
    med_tab.to_csv(TAB_DIR / "classwise_median_distance.csv", float_format="%.4f")
    print("MEAN bottleneck distance\n", mean_tab.round(4).to_string())
    print("\nMEDIAN bottleneck distance\n", med_tab.round(4).to_string())
    print("\n", tab[["within mean", "between mean", "between/within ratio", "silhouette (all)",
                     "silhouette A", "silhouette B", "silhouette C"]].round(3).to_string())

    # nearest-neighbour label agreement: purely descriptive (fraction of samples whose closest other sample has the same label)
    nn = {}
    for k, M in mats.items():
        D = M + np.diag(np.full(len(M), np.inf))
        nn[f"H{k}"] = float((y_s[D.argmin(1)] == y_s).mean())
    pd.Series(nn, name="nearest_neighbour_same_label").to_csv(TAB_DIR / "nearest_neighbour_agreement.csv")
    print("\nfraction of windows whose nearest diagram has the same label:", nn,
          "\n(descriptive only; overlapping windows inflate this number)")

    # violin plot of within/between distances
    L = pd.DataFrame(long)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.3))
    for ax, k in zip(axes, (0, 1)):
        data = [L[(L.homology == f"H{k}") & (L.pair == f"{a}-{b}")].d.values for a, b in PAIRS]
        vp = ax.violinplot(data, showmedians=True)
        for body, (a, b) in zip(vp["bodies"], PAIRS):
            body.set_facecolor(COLORS[a] if a == b else "grey"); body.set_alpha(.55)
        ax.set_xticks(range(1, 7)); ax.set_xticklabels([f"{a}-{b}" for a, b in PAIRS])
        ax.set_title(f"H{k}: within-class (coloured) vs between-class (grey) bottleneck distances"); ax.set_ylabel("d_B [g]")
    fig.tight_layout(); savefig(fig, "06_within_between_distances.png"); plt.close(fig)

    # ---------------- MDS ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    stress = {}
    for ax, (k, M) in zip(axes, mats.items()):
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, n_init=4, max_iter=1000, normalized_stress="auto")
        pos = mds.fit_transform(M)
        # Kruskal stress-1: sqrt(sum (d_ij - ||x_i-x_j||)^2 / sum d_ij^2)  -> quality of the 2D picture
        from scipy.spatial.distance import pdist, squareform
        E = squareform(pdist(pos)); iu = np.triu_indices(len(M), 1)
        stress[f"H{k}"] = float(np.sqrt(((M[iu] - E[iu]) ** 2).sum() / (M[iu] ** 2).sum()))
        np.save(DM_DIR / f"mds2d_bottleneck_H{k}.npy", pos)
        for c in CLASSES:
            ax.scatter(*pos[y_s == c].T, s=14, c=COLORS[c], label=c, alpha=.8)
        ax.set_title(f"2D metric MDS of bottleneck(H{k})   Kruskal stress-1 = {stress[f'H{k}']:.2f}")
        ax.legend(); ax.set_aspect("equal")
    fig.tight_layout(); savefig(fig, "06_mds_H0_H1.png"); plt.close(fig)
    pd.Series(stress, name="kruskal_stress1").to_csv(TAB_DIR / "mds_stress.csv")
    print("\nMDS Kruskal stress-1:", {k: round(v, 3) for k, v in stress.items()})
    scale_dependence(y)


def scale_dependence(y):
    """How much of the diagram differences is explained by the overall SIZE of the cloud?
    rms_radius = sqrt(mean ||a(t) - centroid||^2) is rotation/translation invariant.  Diagrams are NOT rescaled
    (Track A keeps the raw geometry); we only measure correlations and scale-free ratios."""
    s = pd.read_csv(TAB_DIR / "diagram_summary_per_window.csv")
    rms = pd.read_csv(TAB_DIR / "per_window_stats.csv")["rms_radius"].values
    feats = ["H0_total_pers", "H0_max_pers", "H1_total_pers", "H1_max_pers", "H1_max_death", "H1_n"]
    rows = []
    for f in feats:
        v = s[f].values
        rows.append({"feature": f, "corr_with_rms_all": np.corrcoef(v, rms)[0, 1],
                     "corr_with_rms_within_class_mean": np.mean([np.corrcoef(v[y == c], rms[y == c])[0, 1] for c in CLASSES]),
                     **{f"{c}_mean_of_feature/rms": (v[y == c] / rms[y == c]).mean() for c in CLASSES},
                     **{f"{c}_std_of_feature/rms": (v[y == c] / rms[y == c]).std() for c in CLASSES}})
    rows.append({"feature": "rms_radius itself", **{f"{c}_mean_of_feature/rms": rms[y == c].mean() for c in CLASSES},
                 **{f"{c}_std_of_feature/rms": rms[y == c].std() for c in CLASSES}})
    df = pd.DataFrame(rows); df.to_csv(TAB_DIR / "scale_dependence.csv", index=False, float_format="%.4f")
    print("\nscale dependence of diagram statistics (rms radius of the cloud):")
    print(df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.3))
    for c in CLASSES:
        m = y == c
        axes[0].scatter(rms[m], s.H1_total_pers[m], s=10, c=COLORS[c], label=c)
        axes[1].scatter(rms[m], s.H1_max_pers[m], s=10, c=COLORS[c], label=c)
        axes[2].hist(s.H1_max_pers[m] / rms[m], bins=20, alpha=.6, color=COLORS[c], label=c)
    axes[0].set_xlabel("rms radius of the cloud [g]"); axes[0].set_ylabel("H1 total persistence")
    axes[1].set_xlabel("rms radius of the cloud [g]"); axes[1].set_ylabel("H1 max persistence")
    axes[2].set_xlabel("H1 max persistence / rms radius (scale-free)")
    axes[0].set_title("size drives total persistence"); axes[1].set_title("dominant loop vs size")
    axes[2].set_title("after removing scale: A still has a relatively smaller loop")
    for a in axes: a.legend()
    fig.tight_layout(); savefig(fig, "06_scale_dependence.png"); plt.close(fig)


if __name__ == "__main__":
    main()
