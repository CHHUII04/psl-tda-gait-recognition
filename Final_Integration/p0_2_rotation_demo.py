"""P0-2  Rotation demo: what happens if the phone is held in another orientation?

Each TEST window gets its own random rotation Q (a(t) -> Q a(t)); models are trained on the original training data.
Compared:
  * orientation baseline: random forest on the window mean (~ gravity direction) -> NOT invariant
  * xyz statistics baseline: random forest on per-axis mean / std / min / max     -> NOT invariant
  * routes A, B, C (topological, invariant by construction)
(1) blocked & purged CV accuracy with original vs rotated test folds
(2) predictions on the 22 evaluation windows before / after rotation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from final_utils import (FIG, TAB, blocked_purged_splits, compute_diagrams, load, model, random_rotations)

BEST = {"A: Alpha on R^3": ((0, 1), "silhouette"), "B: Rips on delay R^9": ((0, 1), "silhouette"),
        "C: sublevel |a(t)|": ((0, 1), "silhouette")}


def xyz_stats(W):
    return np.hstack([W.mean(1), W.std(1), W.min(1), W.max(1)])


def main():
    X, y, E = load()
    XE = np.concatenate([X, E])
    Q = random_rotations(len(XE), seed=7)
    XE_rot = np.einsum("ntj,nkj->ntk", XE, Q)                     # row-vector convention: a(t) -> Q a(t)
    assert np.allclose(np.linalg.norm(XE_rot, axis=2), np.linalg.norm(XE, axis=2))   # |Q a| = |a|
    X_rot, E_rot = XE_rot[:240], XE_rot[240:]
    splits = blocked_purged_splits(X, y)

    D, Dr = {}, {}
    for r in BEST:
        D[r] = compute_diagrams(r, XE, "orig_" + r[0])
        Dr[r] = compute_diagrams(r, XE_rot, "rot_" + r[0])
        print(r, "diagrams ready", flush=True)

    rows, preds = [], {}
    simple = {"baseline: window mean (orientation)": lambda W: W.mean(1), "baseline: xyz mean/std/min/max": xyz_stats}
    for name, f in simple.items():
        acc_o, acc_r = [], []
        for tr, te in splits:
            m = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(f(X[tr]), y[tr])
            acc_o.append((m.predict(f(X[te])) == y[te]).mean()); acc_r.append((m.predict(f(X_rot[te])) == y[te]).mean())
        m = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(f(X), y)
        preds[name] = (m.predict(f(E)), m.predict(f(E_rot)))
        rows.append({"method": name, "acc original test": np.mean(acc_o), "acc rotated test": np.mean(acc_r)})
    for r, (dims, kind) in BEST.items():
        acc_o, acc_r = [], []
        for tr, te in splits:
            m = model(dims, kind).fit([D[r][i] for i in tr], y[tr])
            acc_o.append((m.predict([D[r][i] for i in te]) == y[te]).mean())
            acc_r.append((m.predict([Dr[r][i] for i in te]) == y[te]).mean())
        m = model(dims, kind).fit(D[r][:240], y)
        preds[r] = (m.predict(D[r][240:]), m.predict(Dr[r][240:]))
        rows.append({"method": f"TDA route {r} ({kind})", "acc original test": np.mean(acc_o), "acc rotated test": np.mean(acc_r)})
    res = pd.DataFrame(rows)
    res["eval: predictions unchanged after rotation"] = [f"{(a == b).sum()}/22" for a, b in preds.values()]
    res["eval counts A/B/C (original)"] = [str(tuple(int((a == c).sum()) for c in "ABC")) for a, _ in preds.values()]
    res["eval counts A/B/C (rotated)"] = [str(tuple(int((b == c).sum()) for c in "ABC")) for _, b in preds.values()]
    res.to_csv(TAB / "p0_2_rotation_demo.csv", index=False, float_format="%.3f")
    print(res.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    xs = np.arange(len(res)); w = .38
    ax.bar(xs - w / 2, res["acc original test"], w, label="test windows as recorded", color="#888")
    ax.bar(xs + w / 2, res["acc rotated test"], w, label="test windows randomly rotated", color="#e377c2")
    for x, a, b in zip(xs, res["acc original test"], res["acc rotated test"]):
        ax.text(x - w / 2, a + .01, f"{a:.2f}", ha="center", fontsize=8); ax.text(x + w / 2, b + .01, f"{b:.2f}", ha="center", fontsize=8)
    ax.axhline(1 / 3, ls=":", c="k", lw=1); ax.text(len(res) - .5, 1 / 3 + .01, "chance", fontsize=8, ha="right")
    ax.set_xticks(xs); ax.set_xticklabels([m.replace("baseline: ", "baseline:\n").replace("TDA route ", "").replace(" (silhouette)", "")
                                           for m in res.method], fontsize=8)
    ax.set_ylim(0, 1.1); ax.set_ylabel("blocked CV accuracy"); ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Rotate the phone: coordinate-based features collapse, topological features are unchanged")
    fig.savefig(FIG / "p0_2_rotation_demo.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
