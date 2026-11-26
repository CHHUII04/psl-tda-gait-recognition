"""P0-3  Final predictions of the 22 evaluation windows in FILE ORDER (numbered 1..22) + class counts.
P1-4  Annotated persistence diagrams: (b, d, d-b) of one feature, read back on the data."""
import gudhi as gd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_utils import COLORS, FIG, TAB, compute_diagrams, load, model

FINAL = "C: sublevel |a(t)|"


def predictions():
    X, y, E = load()
    XE = np.concatenate([X, E])
    P = {}
    for r in ["C: sublevel |a(t)|", "A: Alpha on R^3", "B: Rips on delay R^9"]:
        D = compute_diagrams(r, XE, "orig_" + r[0])
        P[r] = model((0, 1), "silhouette").fit(D[:240], y).predict(D[240:])
    df = pd.DataFrame({"window": np.arange(1, 23), "final label (route C)": P[FINAL],
                       "route A (Alpha)": P["A: Alpha on R^3"], "route B (delay Rips)": P["B: Rips on delay R^9"]})
    df["all agree"] = (df.iloc[:, 1] == df.iloc[:, 2]) & (df.iloc[:, 1] == df.iloc[:, 3])
    df.to_csv(TAB / "final_predictions_22.csv", index=False)
    labels = list(P[FINAL]); counts = {c: labels.count(c) for c in "ABC"}
    assert sum(counts.values()) == 22
    print("List of labels:", labels)
    for c in "ABC": print(f"Total number of class {c}: {counts[c]}")
    print("disagreements:\n", df[~df["all agree"]].to_string(index=False))

    # slide figure
    fig, ax = plt.subplots(figsize=(13, 3.3)); ax.axis("off")
    for j, lab in enumerate(labels):
        x = j % 11; yy = 1 - j // 11
        ax.add_patch(plt.Rectangle((x, yy), .92, .85, color=COLORS[lab], alpha=.85))
        ax.text(x + .46, yy + .58, lab, ha="center", va="center", fontsize=20, color="w", weight="bold")
        ax.text(x + .46, yy + .18, f"#{j + 1}", ha="center", va="center", fontsize=9, color="w")
        if not df["all agree"][j]:
            ax.add_patch(plt.Rectangle((x, yy), .92, .85, fill=False, ec="k", lw=2.5, ls="--"))
    ax.set_xlim(-.1, 15.5); ax.set_ylim(-.1, 2)
    for k, c in enumerate("ABC"):
        ax.text(11.6, 1.55 - .55 * k, f"Person {c}:  n = {counts[c]}", fontsize=15, color=COLORS[c], weight="bold", va="center")
    ax.text(11.6, -.05, f"total = {sum(counts.values())}   (dashed: routes A/B disagree)", fontsize=9, va="center")
    ax.set_title("Predicted labels of the 22 evaluation windows, in file order", fontsize=13)
    fig.savefig(FIG / "p0_3_final_predictions.png", dpi=200, bbox_inches="tight")
    return X, y


def annotated(X, y):
    i = int(np.where(y == "A")[0][26])                  # one window of walker A (index 26 of the training set)
    w = X[i]
    # ---- Alpha H1 diagram, most persistent loop ----
    st = gd.AlphaComplex(points=w).create_simplex_tree(); st.compute_persistence()
    h1 = np.sqrt(st.persistence_intervals_in_dimension(1)); l = h1[:, 1] - h1[:, 0]; b, d = h1[l.argmax()]
    # ---- peaks of |a(t)|: superlevel filtration = sublevel of -s ----
    s = np.linalg.norm(w, axis=1)
    sp = gd.SimplexTree()
    for t, v in enumerate(-s): sp.insert([t], v)
    for t in range(len(s) - 1): sp.insert([t, t + 1], max(-s[t], -s[t + 1]))
    sp.compute_persistence()
    pairs = [(p[0][0], dth) for p in sp.persistence_pairs() if len(p[0]) == 1 and len(p[1]) == 2
             for dth in [max(-s[p[1][0]], -s[p[1][1]])]]
    pk = np.array([(-s[t], dd) for t, dd in pairs]); tpk = np.array([t for t, _ in pairs])
    k = np.argmax(pk[:, 1] - pk[:, 0]); bp, dp = pk[k]; tk = tpk[k]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    ax = axes[0]
    ax.scatter(h1[:, 0], h1[:, 1], s=10, c=COLORS["A"], alpha=.6)
    m = h1[:, 1].max() * 1.05; ax.plot([0, m], [0, m], "k--", lw=.8)
    ax.scatter([b], [d], s=120, facecolors="none", edgecolors="k", lw=2)
    ax.annotate(f"b = {b:.3f}\nd = {d:.3f}\nd - b = {d - b:.3f}", (b, d), (b - .12, d - .02), fontsize=11,
                arrowprops=dict(arrowstyle="->"), ha="right", va="center")
    ax.plot([b, b], [b, d], "k:", lw=1)
    ax.set_xlabel("birth radius r [g]"); ax.set_ylabel("death radius r [g]")
    ax.set_title(f"Route A - Alpha H1, walker A (training window #{i})\n"
                 f"a hole appears at radius b, is filled at radius d", fontsize=10)

    ax = axes[1]
    ax.scatter(pk[:, 0], pk[:, 1], s=12, c=COLORS["A"], alpha=.7)
    lo = pk.min() * 1.05; ax.plot([lo, 0], [lo, 0], "k--", lw=.8)
    ax.scatter([bp], [dp], s=120, facecolors="none", edgecolors="k", lw=2)
    ax.annotate(f"b = -{-bp:.3f}  (peak value)\nd = -{-dp:.3f}  (saddle value)\nd - b = {dp - bp:.3f}  (peak height)",
                (bp, dp), (-1.25, -2.05), fontsize=11, arrowprops=dict(arrowstyle="->"))
    ax.set_xlabel("birth  (= -|a| at the peak)"); ax.set_ylabel("death  (= -|a| where the peak merges)")
    ax.set_title("Route C - H0 of the superlevel filtration of |a(t)| (peaks)\nsame window", fontsize=10)

    ax = axes[2]
    ax.plot(s, c=COLORS["A"], lw=1.2)
    tv = int(np.argmin(np.abs(s - (-dp))))                      # time of the saddle (valley) value
    ax.scatter([tk], [-bp], s=70, c="k", zorder=5); ax.scatter([tv], [-dp], s=70, marker="s", c="k", zorder=5)
    ax.annotate("", (tk + 4, -bp), (tk + 4, -dp), arrowprops=dict(arrowstyle="<->", lw=1.5))
    ax.hlines(-dp, tv - 5, tk + 8, colors="k", linestyles=":")
    ax.text(tk - 6, -bp + .05, f"peak {-bp:.2f} = birth", ha="right", fontsize=9)
    ax.text(tv - 6, -dp - .12, f"saddle {-dp:.2f} = death", ha="right", fontsize=9)
    ax.text(5, 2.32, f"persistence = {dp - bp:.2f} g = height of this step impact above the valley\n"
                     f"where it merges with a higher peak (elder rule)", fontsize=9)
    ax.set_ylim(top=2.55)
    ax.set_xlabel("time index"); ax.set_ylabel("|a(t)| [g]")
    ax.set_title("the same feature read on the signal", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "p1_4_annotated_diagrams.png", dpi=200, bbox_inches="tight")
    print(f"Alpha H1 point: b={b:.3f} d={d:.3f} pers={d - b:.3f};  peak: value {-bp:.3f}, merge {-dp:.3f}, pers {dp - bp:.3f}")


if __name__ == "__main__":
    X, y = predictions()
    annotated(X, y)
