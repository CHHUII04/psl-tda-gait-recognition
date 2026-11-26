"""P0-6  Is it only amplitude?  Amplitude-only controls and amplitude-normalised TDA.

(1) Amplitude-only baselines (no topology): mean and std of |a(t)|; RMS radius of the centred cloud.
(2) Remove the amplitude from every window, then redo everything:
      norm signal  s~ = (s - mean(s)) / std(s)                 (z-score per window)
      point cloud  X~ = (X - centroid) / rms_radius(X)         (unit-size cloud, still rotation invariant)
    TDA on the normalised data: route C (sublevel of s~), route A (Alpha on X~), route B (delay Rips on X~, lag 8).
    Non-topological SHAPE baselines on the same normalised data (fair control):
      quantiles of s~ (shape of the value distribution), autocorrelation of s (lags 1..40, scale-free),
      both together.
All: blocked & purged 5-fold CV; random forest (300 trees, seed 0); TDA with silhouettes of H0 + H1.
"""
import gudhi as gd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from final_utils import FIG, TAB, _sublevel_h0, blocked_purged_splits, compute_diagrams, load, model


def zscore(s):
    return (s - s.mean()) / s.std()


def unit_cloud(w):
    c = w - w.mean(0)
    return c / np.sqrt((c ** 2).sum(1).mean())


def autocorr(s, L=40):
    s = s - s.mean(); ac = np.correlate(s, s, "full")[len(s) - 1:]
    return ac[1:L + 1] / ac[0]


def cv_features(F, y, splits):
    acc = [(RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(F[tr], y[tr]).predict(F[te]) == y[te]).mean()
           for tr, te in splits]
    return np.mean(acc), np.min(acc)


def cv_diagrams(D, y, splits):
    acc = [(model((0, 1), "silhouette").fit([D[i] for i in tr], y[tr]).predict([D[i] for i in te]) == y[te]).mean()
           for tr, te in splits]
    return np.mean(acc), np.min(acc)


def main():
    X, y, E = load()
    splits = blocked_purged_splits(X, y)
    S = np.linalg.norm(X, axis=2)
    rms = np.sqrt(((X - X.mean(1, keepdims=True)) ** 2).sum(2).mean(1))
    rows = []

    def add(group, name, res, note=""):
        rows.append({"group": group, "method": name, "accuracy": res[0], "worst fold": res[1], "note": note})
        print(f"{group:28s} {name:46s} acc={res[0]:.3f} (worst fold {res[1]:.3f})", flush=True)

    # ---------- (1) amplitude only ----------
    add("1 amplitude only", "mean + std of |a(t)|  (2 numbers)", cv_features(np.c_[S.mean(1), S.std(1)], y, splits))
    add("1 amplitude only", "std of |a(t)|  (1 number)", cv_features(S.std(1)[:, None], y, splits))
    add("1 amplitude only", "RMS radius of the cloud  (1 number)", cv_features(rms[:, None], y, splits))

    # ---------- reference: TDA on raw data ----------
    for r, nm in [("C: sublevel |a(t)|", "C  sublevel |a|"), ("A: Alpha on R^3", "A  Alpha R^3"),
                  ("B8: Rips on delay R^9, lag 8", "B  delay Rips lag 8")]:
        D = compute_diagrams(r, X, "orig_" + r.split(":")[0] + "_train")
        add("2 TDA, raw data", nm, cv_diagrams(D, y, splits))

    # ---------- (2) amplitude removed ----------
    Sz = np.array([zscore(s) for s in S])
    Dc = [[_sublevel_h0(s), _sublevel_h0(-s)] for s in Sz]
    add("3 TDA, amplitude removed", "C  sublevel of z-scored |a|", cv_diagrams(Dc, y, splits))
    Xu = np.array([unit_cloud(w) for w in X])
    assert np.allclose(np.sqrt((Xu ** 2).sum(2).mean(1)), 1)
    add("3 TDA, amplitude removed", "A  Alpha on unit-size cloud",
        cv_diagrams(compute_diagrams("A: Alpha on R^3", Xu, "unitcloud_A"), y, splits))
    add("3 TDA, amplitude removed", "B  delay Rips lag 8 on unit-size cloud",
        cv_diagrams(compute_diagrams("B8: Rips on delay R^9, lag 8", Xu, "unitcloud_B8"), y, splits))

    Q = np.array([np.quantile(s, np.linspace(.05, .95, 19)) for s in Sz])
    AC = np.array([autocorr(s) for s in S])
    add("4 non-TDA shape, amplitude removed", "quantiles of z-scored |a|  (19)", cv_features(Q, y, splits))
    add("4 non-TDA shape, amplitude removed", "autocorrelation of |a|, lags 1-40", cv_features(AC, y, splits))
    add("4 non-TDA shape, amplitude removed", "quantiles + autocorrelation", cv_features(np.c_[Q, AC], y, splits))

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "p0_6_amplitude_control.csv", index=False, float_format="%.3f")

    # amplitude separation figure: std of |a| per class
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    for c, col in zip("ABC", ["#C8322F", "#2E7D32", "#1F6FB2"]):
        axes[0].hist(S.std(1)[y == c], bins=20, alpha=.6, color=col, label=c)
    axes[0].set_xlabel("std of |a(t)| in the window [g]"); axes[0].set_title("one amplitude number per window"); axes[0].legend()
    d = df.copy(); colors = {"1": "#9AA5B1", "2": "#14213D", "3": "#E07A1F", "4": "#6A7179"}
    axes[1].barh(range(len(d))[::-1], d.accuracy, color=[colors[g[0]] for g in d.group])
    axes[1].set_yticks(range(len(d))[::-1]); axes[1].set_yticklabels(d.method, fontsize=9)
    axes[1].set_xlim(0.3, 1.02); axes[1].axvline(1 / 3, ls=":", c="k")
    for i, a in zip(range(len(d))[::-1], d.accuracy): axes[1].text(a + .005, i, f"{a:.2f}", va="center", fontsize=9)
    axes[1].set_xlabel("blocked & purged CV accuracy")
    fig.tight_layout(); fig.savefig(FIG / "p0_6_amplitude_control.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
