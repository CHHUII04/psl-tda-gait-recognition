"""01 - Data audit: shapes, labels, missing values, numeric ranges.  Output: results/tables/*.csv"""
import numpy as np
import pandas as pd

from common import CLASSES, TAB_DIR, load_eval, load_train


def describe(arr: np.ndarray, name: str) -> dict:
    """Global statistics of a (n,200,3) array, per axis and for the norm |a(t)|."""
    norm = np.linalg.norm(arr, axis=2)                       # Euclidean magnitude sqrt(ax^2+ay^2+az^2)
    row = {"subset": name, "n_windows": arr.shape[0], "points_per_window": arr.shape[1], "dim": arr.shape[2],
           "n_nan": int(np.isnan(arr).sum()), "n_inf": int(np.isinf(arr).sum())}
    for k, ax in enumerate("xyz"):
        v = arr[:, :, k]
        row.update({f"{ax}_mean": v.mean(), f"{ax}_std": v.std(), f"{ax}_min": v.min(), f"{ax}_max": v.max()})
    row.update({"norm_mean": norm.mean(), "norm_std": norm.std(), "norm_min": norm.min(), "norm_max": norm.max()})
    # duplicate points inside a window would be merged by the Alpha complex -> check
    row["max_duplicate_points_in_window"] = int(max(arr.shape[1] - len(np.unique(w, axis=0)) for w in arr))
    return row


def main():
    X, y = load_train()
    E = load_eval()
    print("train", X.shape, "eval", E.shape)
    labels, counts = np.unique(y, return_counts=True)
    print("label distribution:", dict(zip(labels, counts)))
    assert set(labels) == set(CLASSES), "unexpected labels"

    rows = [describe(X, "train_all")] + [describe(X[y == c], f"train_{c}") for c in CLASSES] + [describe(E, "eval")]
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "data_audit.csv", index=False, float_format="%.5f")
    print(df[["subset", "n_windows", "n_nan", "n_inf", "norm_mean", "norm_std", "norm_min", "norm_max",
              "max_duplicate_points_in_window"]].to_string(index=False))

    # per-window statistics (used later to understand the geometry: spread of the cloud)
    pw = pd.DataFrame({
        "sample": np.arange(len(X)), "label": y,
        "mean_x": X[:, :, 0].mean(1), "mean_y": X[:, :, 1].mean(1), "mean_z": X[:, :, 2].mean(1),
        "norm_mean": np.linalg.norm(X, axis=2).mean(1), "norm_std": np.linalg.norm(X, axis=2).std(1),
        # rotation-invariant spread of the cloud: RMS distance to its centroid
        "rms_radius": np.sqrt(((X - X.mean(1, keepdims=True)) ** 2).sum(2).mean(1)),
    })
    pw.to_csv(TAB_DIR / "per_window_stats.csv", index=False, float_format="%.5f")
    print("\nper-class mean of per-window statistics:")
    print(pw.groupby("label")[["mean_x", "mean_y", "mean_z", "norm_mean", "norm_std", "rms_radius"]].mean().round(3))

    # windows were cut with a sliding window -> count 3D samples shared by several windows
    seen = {}
    for i, w in enumerate(X):
        for p in map(tuple, w):
            seen.setdefault(p, set()).add(i)
    shared = sum(len(v) > 1 for v in seen.values())
    eval_in_train = sum(tuple(p) in seen for w in E for p in w)
    ov = pd.DataFrame([{"distinct_train_points": len(seen), "points_in_more_than_one_window": shared,
                        "eval_points_found_in_train": eval_in_train, "eval_points_total": E.shape[0] * E.shape[1]}])
    ov.to_csv(TAB_DIR / "window_overlap.csv", index=False)
    print("\nwindow overlap:", ov.to_dict("records")[0])


if __name__ == "__main__":
    main()
