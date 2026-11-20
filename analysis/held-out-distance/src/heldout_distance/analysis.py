"""The three distance curves and their summary.

For each study, test items are grouped by how far they sit from the training data, in
that study's own unit, and an error measure is computed per group:

crop      distance from a test field to its nearest training field (metres); error = 1 - macro-F1
views     angle from a test camera to its nearest training camera (degrees); error = image mean squared error
runtime   size of a test workload over the largest training workload (ratio); error = absolute percentage error

Every curve is also expressed relative to its own nearest group, so the three can be
read on one scale: 1.0 means "as good as on the test items closest to the training data".
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import cluster_bootstrap, interval, macro_f1, spearman

DATA, RESULTS = Path("data"), Path("results")
CROP_EDGES = [0, 50, 100, 200, 400, 800, 1600, 3200, 1e9]
CROP_EDGES_MATCHED = [0, 50, 100, 200, 400, 800, 1e9]
VIEW_EDGES = [0, 12, 16, 20, 26, 90]
RUNTIME_EDGES = [0, 1, 3, 10, 30, 1e9]
N_CROP_CLASSES = 7


def _bins(values: np.ndarray, edges: list) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="left") - 1, 0, len(edges) - 2)


def _curve(error_by_bin, x, bin_id, n_bins, clusters, n_boot, seed) -> dict:
    """Point estimates, bootstrap intervals and the trend for one error-against-distance curve."""
    point = error_by_bin(np.arange(len(x)))
    boot = cluster_bootstrap(clusters, error_by_bin, n_boot, seed)
    rel = boot / boot[:, [0]]
    centre = [float(np.median(x[bin_id == b])) for b in range(n_bins)]
    trend = np.array([spearman(np.arange(n_bins), row) for row in boot])
    return {
        "bin_centre": centre, "n": [int((bin_id == b).sum()) for b in range(n_bins)],
        "error": point.tolist(), "error_ci": interval(boot),
        "relative": (point / point[0]).tolist(), "relative_ci": interval(rel),
        "last_over_first": float(point[-1] / point[0]), "last_over_first_ci": interval(rel[:, -1]),
        "trend_spearman": spearman(np.arange(n_bins), point), "trend_spearman_ci": interval(trend),
        "share_of_resamples_with_last_above_first": float(np.mean(rel[:, -1] > 1)),
    }


def crop(n_boot: int = 2000, designs: tuple = ("random", "block")) -> dict:
    """Crop curve. By default only the two designs that train on the same number of fields.

    The buffered and tile designs train on fewer fields, so pooling them would mix the
    effect of distance with the effect of a smaller training set. ``designs=None`` pools all.
    """
    d = pd.read_csv(DATA / "crop_fields.csv.gz")
    if designs is not None:
        d = d[d.design.isin(designs)].reset_index(drop=True)
    y, pred, x = d.y.to_numpy(), d.pred.to_numpy(), d.distance_m.to_numpy()
    edges = CROP_EDGES if designs is None else CROP_EDGES_MATCHED
    b = _bins(x, edges)
    n = len(edges) - 1

    def err(idx):
        return np.array([1 - macro_f1(y[idx][b[idx] == k], pred[idx][b[idx] == k], N_CROP_CLASSES) for k in range(n)])

    out = _curve(err, x, b, n, d.block.to_numpy(), n_boot, seed=1)
    out.update(designs=sorted(d.design.unique()), unit="metres to the nearest training field", error_name="1 - macro-F1", cluster="2 km block",
               n_clusters=int(d.block.nunique()), edges=edges[:-1],
               macro_f1=[1 - e for e in out["error"]],
               accuracy=[float((y[b == k] == pred[b == k]).mean()) for k in range(n)],
               class_share=[np.round(np.bincount(y[b == k], minlength=N_CROP_CLASSES) / (b == k).sum(), 4).tolist()
                            for k in range(n)])
    return out


def views(n_boot: int = 2000) -> dict:
    d = pd.read_csv(DATA / "view_tests.csv.gz")
    x = d.angle_nearest.to_numpy()
    mse, depth = 10 ** (-d.psnr.to_numpy() / 10), d.depth_mae.to_numpy()
    b = _bins(x, VIEW_EDGES)
    n = len(VIEW_EDGES) - 1
    runs = d.run_id.to_numpy()

    def by_bin(values):
        return lambda idx: np.array([values[idx][b[idx] == k].mean() for k in range(n)])

    out = _curve(by_bin(mse), x, b, n, runs, n_boot, seed=2)
    out.update(unit="degrees to the nearest training camera", error_name="image mean squared error",
               cluster="trained model", n_clusters=int(d.run_id.nunique()), edges=VIEW_EDGES[:-1],
               psnr=[float(d.psnr.to_numpy()[b == k].mean()) for k in range(n)],
               depth=_curve(by_bin(depth), x, b, n, runs, n_boot, seed=3))
    return out


def runtime(n_boot: int = 2000) -> dict:
    d = pd.read_csv(DATA / "runtime_cells.csv.gz")
    inside = d[(d.design == "random") & (d.extrapolation <= 1)]
    outside = d[d.design == "size"]
    d = pd.concat([inside, outside], ignore_index=True)
    x = d.extrapolation.to_numpy()
    b = _bins(x, RUNTIME_EDGES)
    n = len(RUNTIME_EDGES) - 1
    out = {"unit": "test workload size over the largest training workload", "error_name": "absolute percentage error",
           "cluster": "workload", "n_clusters": int(d.workload.nunique()), "edges": RUNTIME_EDGES[:-1], "methods": {}}
    for k, method in enumerate(("gbm", "loglinear", "sim")):
        ape = np.abs(d[method].to_numpy() - d.measured.to_numpy()) / d.measured.to_numpy()
        out["methods"][method] = _curve(lambda idx, a=ape: np.array([a[idx][b[idx] == j].mean() for j in range(n)]),
                                        x, b, n, d.workload.to_numpy(), n_boot, seed=4 + k)
        m = out["methods"][method]
        m["last_over_second"] = float(m["error"][-1] / m["error"][1])     # inside the size design only
    return out


def summarise(n_boot: int = 2000) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a resample can leave a small group empty
        s = {"n_boot": n_boot, "crop": crop(n_boot), "crop_all_designs": crop(n_boot, designs=None),
             "views": views(n_boot), "runtime": runtime(n_boot)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=2))
    return s
