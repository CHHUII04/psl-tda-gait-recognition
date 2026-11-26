"""04 - Alpha complex persistence (H0, H1, H2; Z2 coefficients) of every window + diagram figures.

Outputs
  results/persistence_diagrams/train_diagrams_sqradius.pkl   list[{0:H0,1:H1,2:H2}] raw GUDHI values (alpha^2), essentials kept
  results/persistence_diagrams/train_H0_sqradius.pkl / train_H1_sqradius.pkl   per-dimension lists
  results/persistence_diagrams/train_H0_radius.npz / train_H1_radius.npz       radius scale, one array per sample
  results/persistence_diagrams/train_essential_intervals.csv
  results/tables/alpha_complex_stats.csv          timing, simplex counts, #intervals
  results/tables/diagram_summary_per_window.csv   lifetime statistics per window
  results/tables/diagram_summary_by_class.csv     class-wise means + Kruskal-Wallis tests
  results/evaluation_diagrams/eval_*              same for the 22 evaluation windows
Use --force to recompute the cached diagrams.
"""
import sys

import gudhi as gd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal

from common import (CLASSES, COLORS, EVAL_DIR, PD_DIR, TAB_DIR, alpha_persistence, finite, lifetimes,
                    load_eval, load_pickle, load_train, representative_indices, save_pickle, savefig, to_radius)


def compute_split(W, split, folder, force=False):
    raw_file = folder / f"{split}_diagrams_sqradius.pkl"
    stats_file = TAB_DIR / f"alpha_complex_stats_{split}.csv"
    if raw_file.exists() and stats_file.exists() and not force:
        print(f"[{split}] cached diagrams found -> {raw_file.name}")
        return load_pickle(raw_file), pd.read_csv(stats_file), None
    diags, rows, pairs = [], [], []
    for i, w in enumerate(W):
        try:
            res = alpha_persistence(w)
        except Exception as exc:                      # report and stop: a missing diagram would shift indices
            raise RuntimeError(f"alpha persistence failed for {split} window {i}") from exc
        diags.append(res["sqradius"]); pairs.append(res["pairs"])
        sc = res["simplex_counts"]
        rows.append({"sample": i, "seconds": res["seconds"], "n_vertices": res["n_vertices"],
                     **{f"n_{k}simplices": int(sc[k]) if k < len(sc) else 0 for k in range(4)},
                     **{f"n_intervals_H{k}": len(res["sqradius"][k]) for k in range(3)},
                     **{f"n_essential_H{k}": int(np.isinf(res["sqradius"][k][:, 1]).sum()) for k in range(3)}})
        if (i + 1) % 40 == 0 or i + 1 == len(W):
            print(f"[{split}] {i + 1}/{len(W)} diagrams")
    stats = pd.DataFrame(rows)
    # --- save (raw squared-radius convention, essentials kept) ---
    save_pickle(diags, raw_file)
    save_pickle(pairs, folder / f"{split}_persistence_pairs.pkl")
    for k in (0, 1, 2):
        save_pickle([d[k] for d in diags], folder / f"{split}_H{k}_sqradius.pkl")
    for k in (0, 1):
        np.savez_compressed(folder / f"{split}_H{k}_radius.npz",
                            **{f"sample_{i:03d}": to_radius(d[k]) for i, d in enumerate(diags)})
    ess = [{"sample": i, "dim": k, "birth_sqradius": b} for i, d in enumerate(diags)
           for k in (0, 1, 2) for b, de in d[k] if np.isinf(de)]
    pd.DataFrame(ess).to_csv(folder / f"{split}_essential_intervals.csv", index=False)
    stats.to_csv(stats_file, index=False, float_format="%.5f")
    return diags, stats, pairs


def summary_row(d):
    """Lifetime statistics of one window (radius scale, finite part)."""
    out = {}
    for k in (0, 1):
        f = finite(to_radius(d[k])); l = f[:, 1] - f[:, 0]
        p = l / l.sum() if l.sum() > 0 else l
        out.update({f"H{k}_n": len(l), f"H{k}_total_pers": l.sum(), f"H{k}_max_pers": l.max(initial=0),
                    f"H{k}_mean_pers": l.mean() if len(l) else 0, f"H{k}_entropy": -(p * np.log(p + 1e-15)).sum()})
    l1 = np.sort(lifetimes(to_radius(d[1])))[::-1]
    out["H1_ratio_top1_top2"] = l1[0] / l1[1] if len(l1) > 1 else np.nan
    out["H1_max_death"] = finite(to_radius(d[1]))[:, 1].max()
    return out


def check_h0_equals_mst(X, diags, n=10):
    """Alpha H0 deaths (radius) = half the edge lengths of the Euclidean minimum spanning tree."""
    errs = []
    for i in range(n):
        mst = minimum_spanning_tree(squareform(pdist(X[i]))).data
        h0 = finite(to_radius(diags[i][0]))[:, 1]
        errs.append(np.abs(np.sort(mst / 2) - np.sort(h0)).max())
    return max(errs)


def plot_diagram(ax, d, k, color, title):
    f = finite(to_radius(d[k])); l = f[:, 1] - f[:, 0]
    ax.scatter(f[:, 0], f[:, 1], s=10, c=color, alpha=.7, edgecolors="none")
    top = np.argsort(l)[::-1][:3]
    ax.scatter(f[top, 0], f[top, 1], s=60, facecolors="none", edgecolors="k", lw=1)
    ess = to_radius(d[k])[np.isinf(d[k][:, 1])]
    m = f[:, 1].max() * 1.08 if len(f) else 1
    for b in ess[:, 0]:
        ax.scatter([b], [m], marker="^", c="k", s=30); ax.text(b, m, "  ess. (inf)", fontsize=7, va="center")
    ax.plot([0, m], [0, m], "k--", lw=.7)
    ax.set_xlim(-0.01 * m, m * 1.02); ax.set_ylim(-0.01 * m, m * 1.08)
    ax.set_xlabel("birth radius  r = sqrt(alpha^2)  [g]"); ax.set_ylabel("death radius [g]")
    ax.set_title(title, fontsize=9)


def main():
    force = "--force" in sys.argv
    X, y = load_train(); E = load_eval()
    diags, stats, pairs = compute_split(X, "train", PD_DIR, force)
    ev_diags, ev_stats, _ = compute_split(E, "eval", EVAL_DIR, force)
    if pairs is None:
        pairs = load_pickle(PD_DIR / "train_persistence_pairs.pkl")

    stats["label"] = y
    stats.to_csv(TAB_DIR / "alpha_complex_stats.csv", index=False, float_format="%.5f")
    print("\nAlpha complex statistics (mean over training windows):")
    print(stats.drop(columns=["sample", "label"]).mean().round(4).to_string())
    print("total time train (s):", round(stats.seconds.sum(), 2))
    assert (stats.n_essential_H0 == 1).all() and (stats.n_essential_H1 == 0).all() and (stats.n_essential_H2 == 0).all()
    print("essential intervals: exactly one in H0 (birth 0) per window, none in H1/H2 -> OK")
    print("max |H0 death - MST edge/2| over 10 windows:", f"{check_h0_equals_mst(X, diags):.2e}")

    # ---------- lifetime summaries ----------
    summ = pd.DataFrame([summary_row(d) for d in diags]); summ.insert(0, "label", y); summ.insert(0, "sample", range(len(y)))
    summ.to_csv(TAB_DIR / "diagram_summary_per_window.csv", index=False, float_format="%.5f")
    feats = [c for c in summ.columns if c not in ("sample", "label")]
    by = summ.groupby("label")[feats].agg(["mean", "std"])
    # Kruskal-Wallis test of "same distribution in A, B, C"; undefined for a constant feature (e.g. H0_n = 199 always)
    kw = {f: (kruskal(*[summ.loc[summ.label == c, f].dropna() for c in CLASSES]).pvalue
              if summ[f].nunique() > 1 else np.nan) for f in feats}
    by_t = by.T.unstack(); by_t.columns = [f"{c}_{s}" for c, s in by_t.columns]
    by_t["kruskal_wallis_p"] = pd.Series(kw)
    by_t.to_csv(TAB_DIR / "diagram_summary_by_class.csv", float_format="%.4g")
    print("\nclass-wise lifetime statistics (radius scale):")
    print(by_t.round(4).to_string())

    reps = representative_indices(y, per_class=2)
    # ---------- figure: H0/H1 diagrams of representatives ----------
    fig, axes = plt.subplots(2, 6, figsize=(26, 8.5))
    for col, (c, i) in enumerate([(c, i) for c in CLASSES for i in reps[c]]):
        for k in (0, 1):
            plot_diagram(axes[k, col], diags[i], k, COLORS[c], f"person {c}, window #{i} - H{k}")
    fig.suptitle("Alpha persistence diagrams (radius scale; circled: 3 most persistent points; triangle: essential class)")
    fig.tight_layout(); savefig(fig, "04_alpha_diagrams_representatives.png"); plt.close(fig)

    # ---------- figure: lifetime distributions ----------
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.3))
    for c in CLASSES:
        idx = np.where(y == c)[0]
        l0 = np.concatenate([lifetimes(to_radius(diags[i][0])) for i in idx])
        l1 = np.concatenate([lifetimes(to_radius(diags[i][1])) for i in idx])
        axes[0].hist(l0, bins=60, alpha=.5, color=COLORS[c], density=True, label=c)
        axes[1].hist(l1, bins=np.linspace(0, .5, 60), alpha=.5, color=COLORS[c], density=True, label=c, log=True)
        axes[2].hist(summ.loc[summ.label == c, "H1_max_pers"], bins=20, alpha=.6, color=COLORS[c], label=c)
        axes[3].hist(summ.loc[summ.label == c, "H0_max_pers"], bins=20, alpha=.6, color=COLORS[c], label=c)
    axes[0].set_title("H0 lifetimes (= half MST edge lengths), all windows")
    axes[1].set_title("H1 lifetimes, all windows (log density)")
    axes[2].set_title("most persistent H1 loop per window"); axes[3].set_title("largest finite H0 death per window")
    for a in axes: a.set_xlabel("persistence [g]"); a.legend()
    fig.tight_layout(); savefig(fig, "04_lifetime_distributions.png"); plt.close(fig)

    # ---------- figure: class comparison of summary statistics ----------
    show = ["H0_total_pers", "H0_max_pers", "H1_n", "H1_total_pers", "H1_max_pers", "H1_ratio_top1_top2"]
    fig, axes = plt.subplots(1, len(show), figsize=(24, 4))
    for a, f in zip(axes, show):
        bp = a.boxplot([summ.loc[summ.label == c, f].dropna() for c in CLASSES], labels=CLASSES, patch_artist=True)
        for patch, c in zip(bp["boxes"], CLASSES): patch.set_facecolor(COLORS[c]); patch.set_alpha(.5)
        a.set_title(f"{f}\nKruskal-Wallis p = {kw[f]:.1e}", fontsize=9)
    fig.tight_layout(); savefig(fig, "04_summary_statistics_by_class.png"); plt.close(fig)

    # ---------- figure: where is the most persistent loop? (birth edge / death triangle on the cloud) ----------
    fig = plt.figure(figsize=(18, 5.5))
    for col, c in enumerate(CLASSES):
        i = reps[c][0]; P = X[i]
        d1 = diags[i][1]; l = d1[:, 1] - d1[:, 0]
        # persistence pairs of dimension 1: (birth edge, killing triangle); match by filtration values
        p1 = [(b, dth) for b, dth in pairs[i] if len(b) == 2 and len(dth) == 3]
        st = gd.AlphaComplex(points=P).create_simplex_tree()
        filt = {tuple(sorted(s)): f for s, f in st.get_simplices()}
        best = max(p1, key=lambda bd: filt[tuple(sorted(bd[1]))] - filt[tuple(sorted(bd[0]))])
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        ax.plot(*P.T, color=COLORS[c], lw=.5, alpha=.6)
        be = P[list(best[0])]; tr = P[list(best[1]) + [best[1][0]]]
        ax.plot(*be.T, "k-", lw=3, label="birth edge")
        ax.plot(*tr.T, "m-", lw=2, label="killing triangle")
        r_b, r_d = np.sqrt(filt[tuple(sorted(best[0]))]), np.sqrt(filt[tuple(sorted(best[1]))])
        ax.set_title(f"{c} #{i}: most persistent H1 loop  r_birth={r_b:.3f}, r_death={r_d:.3f}", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Most persistent H1 class of one window per person: born when its birth edge closes a cycle (radius r_birth),\n"
                 "killed when the triangle enters (radius r_death). It lies in a sparse outer region of the cloud, not on one clean gait cycle")
    fig.tight_layout(); savefig(fig, "04_dominant_loop_geometry.png"); plt.close(fig)

    # evaluation windows: summary only (no labels, no classifier in this track)
    ev_summ = pd.DataFrame([summary_row(d) for d in ev_diags]); ev_summ.insert(0, "sample", range(len(ev_diags)))
    ev_summ.to_csv(EVAL_DIR / "eval_diagram_summary.csv", index=False, float_format="%.5f")
    print(f"\neval diagrams saved: {len(ev_diags)} windows -> {EVAL_DIR.name}/")


if __name__ == "__main__":
    main()
