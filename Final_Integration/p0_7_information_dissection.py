"""P0-7  Which information does each representation use?  (When does TDA work?)

Four versions of the data, each removing one kind of information:
  raw                 : everything
  amplitude removed   : |a| z-scored per window, cloud centred and scaled to RMS radius 1
  time shuffled       : the 200 samples of a window randomly permuted (same permutation for x, y, z):
                        keeps the cloud and the value distribution, destroys temporal order (rhythm, local shape)
  both removed        : amplitude removed, then time shuffled
Six representations: amplitude statistics, quantiles of |a|, autocorrelation of |a| (non-topological);
TDA route A (Alpha), B (delay Rips, lag 8), C (sub/superlevel of |a|), silhouettes of H0 + H1 + random forest.
Blocked & purged 5-fold CV for everything.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_utils import FIG, TAB, _sublevel_h0, blocked_purged_splits, compute_diagrams, load
from p0_6_amplitude_control import autocorr, cv_diagrams, cv_features, unit_cloud, zscore

VERSIONS = ["raw", "amplitude removed", "time shuffled", "amplitude removed + shuffled"]
REPS = ["amplitude: mean, std of |a|", "quantiles of |a|", "autocorrelation of |a|",
        "TDA A: Alpha (cloud)", "TDA B: delay Rips, lag 8", "TDA C: sublevel of |a|"]


def main():
    X, y, E = load()
    splits = blocked_purged_splits(X, y)
    rng = np.random.default_rng(0)
    perms = np.array([rng.permutation(X.shape[1]) for _ in range(len(X))])
    shuffle = lambda W: np.array([w[p] for w, p in zip(W, perms)])
    Xu = np.array([unit_cloud(w) for w in X])
    data = {"raw": X, "amplitude removed": Xu, "time shuffled": shuffle(X), "amplitude removed + shuffled": shuffle(Xu)}
    keys = {"raw": "rawtrain", "amplitude removed": "unitcloud", "time shuffled": "shuf", "amplitude removed + shuffled": "unitshuf"}

    res = {}
    for v in VERSIONS:
        W = data[v]
        S = np.linalg.norm(W, axis=2)
        if "amplitude" in v:                                    # remove the amplitude of the norm signal too
            S = np.array([zscore(np.linalg.norm(w0, axis=1)) for w0 in X])
            if "shuffled" in v:
                S = np.array([s[p] for s, p in zip(S, perms)])
        r = {}
        r[REPS[0]] = cv_features(np.c_[S.mean(1), S.std(1)], y, splits)[0]
        r[REPS[1]] = cv_features(np.array([np.quantile(s, np.linspace(.05, .95, 19)) for s in S]), y, splits)[0]
        r[REPS[2]] = cv_features(np.array([autocorr(s) for s in S]), y, splits)[0]
        r[REPS[3]] = cv_diagrams(compute_diagrams("A: Alpha on R^3", W, f"{keys[v]}_A"), y, splits)[0]
        r[REPS[4]] = cv_diagrams(compute_diagrams("B8: Rips on delay R^9, lag 8", W, f"{keys[v]}_B8"), y, splits)[0]
        r[REPS[5]] = cv_diagrams([[_sublevel_h0(s), _sublevel_h0(-s)] for s in S], y, splits)[0]
        res[v] = r
        print(v, {k: round(x, 3) for k, x in r.items()}, flush=True)
    T = pd.DataFrame(res).loc[REPS, VERSIONS]
    T.to_csv(TAB / "p0_7_information_dissection.csv", float_format="%.3f")
    print(T.round(3).to_string())

    fig, ax = plt.subplots(figsize=(13, 5.2))
    im = ax.imshow(T.values, cmap="RdYlGn", vmin=1 / 3, vmax=1, aspect="auto")
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            ax.text(j, i, f"{T.values[i, j]:.2f}", ha="center", va="center", fontsize=15,
                    color="#14213D", weight="bold" if T.values[i, j] > .95 else "normal")
    ax.set_xticks(range(len(VERSIONS))); ax.set_xticklabels([v.replace(" + ", "\n+ ") for v in VERSIONS], fontsize=13)
    ax.set_yticks(range(len(REPS))); ax.set_yticklabels(REPS, fontsize=13)
    ax.axhline(2.5, c="k", lw=2)
    plt.colorbar(im, ax=ax, fraction=.03, label="blocked CV accuracy (chance = 0.33)")
    ax.set_title("Which information does each representation use?", fontsize=15)
    fig.tight_layout(); fig.savefig(FIG / "p0_7_information_dissection.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
