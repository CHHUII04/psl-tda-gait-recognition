"""P0-4  Confidence and seed stability of the 22 predictions, with a focus on window #14.

For 5 pipelines (final model C + silhouette, C + landscape, A, B lag 2, B lag 8) and 20 random-forest seeds:
  * predicted class probabilities (mean / std over seeds) = fraction of trees voting for each class
  * seed agreement = fraction of seeds whose predicted label equals the majority label
Window #14 is then examined: probabilities in every pipeline, its signal, and its position on the key feature
(total persistence of the peaks of |a(t)|).  Nothing here uses the labels of the evaluation set.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_utils import COLORS, FIG, TAB, compute_diagrams, key, load, model

PIPES = {  # name: (route, vectorization)
    "C + silhouette (final)": ("C: sublevel |a(t)|", "silhouette"),
    "C + landscape": ("C: sublevel |a(t)|", "landscape"),
    "A + silhouette": ("A: Alpha on R^3", "silhouette"),
    "B lag 2 + silhouette": ("B: Rips on delay R^9", "silhouette"),
    "B lag 8 + silhouette": ("B8: Rips on delay R^9, lag 8", "silhouette"),
}
SEEDS = range(20)
W14 = 13          # 0-based index of window #14


def main():
    X, y, E = load()
    XE = np.concatenate([X, E])
    proba, labels = {}, {}
    for name, (route, kind) in PIPES.items():
        D = compute_diagrams(route, XE, "orig_" + key(route))
        P = []
        for s in SEEDS:
            m = model((0, 1), kind, seed=s).fit(D[:240], y)
            assert list(m.classes_) == ["A", "B", "C"]
            P.append(m.predict_proba(D[240:]))
        P = np.array(P)                                    # seeds x 22 x 3
        proba[name] = P
        labels[name] = np.array(list("ABC"))[P.argmax(2)]  # seeds x 22
        print(f"{name:26s} done", flush=True)

    # ---------------- table: final model ----------------
    P = proba["C + silhouette (final)"]; L = labels["C + silhouette (final)"]
    mean, std = P.mean(0), P.std(0)
    final = np.array(list("ABC"))[mean.argmax(1)]
    rows = []
    for j in range(22):
        srt = np.sort(mean[j])[::-1]
        r = {"window": j + 1, "final label": final[j],
             **{f"P({c}) mean": mean[j, k] for k, c in enumerate("ABC")},
             **{f"P({c}) std over seeds": std[j, k] for k, c in enumerate("ABC")},
             "margin (top1 - top2)": srt[0] - srt[1],
             "seed agreement (final model)": (L[:, j] == final[j]).mean()}
        for name in PIPES:
            if name != "C + silhouette (final)":
                lab = labels[name][:, j]; vals, cnt = np.unique(lab, return_counts=True)
                r[f"{name}: label (seed agreement)"] = f"{vals[cnt.argmax()]} ({cnt.max() / len(lab):.0%})"
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "p0_4_confidence_22.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    print(T[["window", "final label", "P(A) mean", "P(B) mean", "P(C) mean", "margin (top1 - top2)",
             "seed agreement (final model)"]].round(3).to_string(index=False))
    print("\nlabels of the other pipelines (majority over 20 seeds, agreement):")
    print(T[["window", "final label"] + [c for c in T.columns if c.endswith("(seed agreement)")]].to_string(index=False))
    counts = {c: int((final == c).sum()) for c in "ABC"}
    print("\nfinal counts:", counts, "| windows with margin < 0.5:", list(T.window[T["margin (top1 - top2)"] < .5]))

    # window #14 in every pipeline
    w14 = pd.DataFrame({name: {**{f"P({c})": proba[name][:, W14, k].mean() for k, c in enumerate("ABC")},
                               "label (majority over seeds)": pd.Series(labels[name][:, W14]).mode()[0],
                               "seed agreement": (labels[name][:, W14] == pd.Series(labels[name][:, W14]).mode()[0]).mean()}
                        for name in PIPES}).T
    w14.to_csv(TAB / "p0_4_window14_all_pipelines.csv", float_format="%.3f")
    print("\nwindow #14:\n", w14.to_string())

    # ---------------- figure 1: confidence of the 22 predictions ----------------
    fig, ax = plt.subplots(figsize=(13, 3.8))
    bottom = np.zeros(22); xs = np.arange(1, 23)
    for k, c in enumerate("ABC"):
        ax.bar(xs, mean[:, k], bottom=bottom, color=COLORS[c], label=f"P({c})", yerr=std[:, k], capsize=2,
               error_kw=dict(lw=.8, ecolor="k"))
        bottom += mean[:, k]
    for j in range(22):
        ax.text(xs[j], 1.03, final[j], ha="center", fontsize=10, weight="bold", color=COLORS[final[j]])
    ax.add_patch(plt.Rectangle((13.5, 0), 1, 1.12, fill=False, ec="k", lw=2, ls="--", clip_on=False))
    ax.set_xticks(xs); ax.set_xticklabels([f"#{i}" for i in xs], fontsize=8); ax.set_ylim(0, 1.15)
    ax.set_ylabel("probability (fraction of trees)")
    ax.set_title("Final model (route C + silhouette + RF): class probabilities of the 22 evaluation windows, "
                 "mean ± std over 20 seeds", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    fig.savefig(FIG / "p0_4_confidence_22.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---------------- figure 2: window #14 ----------------
    Dc = compute_diagrams("C: sublevel |a(t)|", XE, "orig_C")
    tp = np.array([(d[1][:, 1] - d[1][:, 0]).sum() for d in Dc])      # total persistence of the peaks
    w = E[W14]; s = np.linalg.norm(w, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.6))
    ax = axes[0]
    names = list(PIPES); xp = np.arange(len(names))
    bottom = np.zeros(len(names))
    for k, c in enumerate("ABC"):
        v = np.array([proba[n][:, W14, k].mean() for n in names])
        ax.bar(xp, v, bottom=bottom, color=COLORS[c], label=c); bottom += v
    ax.set_xticks(xp); ax.set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=8)
    ax.set_ylabel("probability"); ax.legend(fontsize=8); ax.set_title("window #14: class probabilities in each pipeline", fontsize=10)
    ax = axes[1]
    for c in "ABC":
        ax.hist(tp[:240][y == c], bins=20, alpha=.55, color=COLORS[c], label=f"training {c}")
    ax.axvline(tp[240 + W14], c="k", lw=2.5, label="#14")
    for j in range(22):
        if j != W14: ax.axvline(tp[240 + j], c="grey", lw=.6, alpha=.6)
    ax.set_xlabel("total persistence of the peaks of |a(t)|  [g]"); ax.legend(fontsize=8)
    ax.set_title("key feature: #14 (black) vs training classes (other eval windows: grey)", fontsize=10)
    ax = axes[2]
    roll = pd.Series(w[:, 0]).rolling(25, center=True, min_periods=5).mean()
    ax.plot(s, c="0.6", lw=.8, label="|a(t)|")
    ax2 = ax.twinx()
    ax2.plot(roll, c="k", lw=2, label="running mean of a_x (25 samples)")
    for c in "ABC":
        ax2.axhline(X[y == c][:, :, 0].mean(), c=COLORS[c], ls="--", lw=1.2, label=f"mean a_x of {c}")
    ax.set_xlabel("time index"); ax.set_ylabel("|a(t)| [g]"); ax2.set_ylabel("a_x [g]")
    ax2.legend(fontsize=7, loc="lower right")
    ax.set_title("#14: the sensor orientation drifts during the window\n(transient at the start, then the B value)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "p0_4_window14.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("\n#14 total peak persistence:", round(tp[240 + W14], 2),
          {c: (round(tp[:240][y == c].min(), 1), round(tp[:240][y == c].max(), 1)) for c in "ABC"},
          "| first/second half mean a_x:", w[:100, 0].mean().round(2), w[100:, 0].mean().round(2))


if __name__ == "__main__":
    main()
