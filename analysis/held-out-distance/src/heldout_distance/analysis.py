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
MAJORITY_MACRO_F1 = 0.088          # the crop study's majority-class baseline


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
    d = d[d.model == "tcn"]
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


def structure_test(n_boot: int = 2000) -> dict:
    """The planned test: does a forest on growth-curve parameters degrade less with distance than
    the same forest on the raw series (and than the network), on the same test fields?

    Rules are in the crop study's FOLLOW_UP_PLAN.md. R is the error in the farthest distance
    group over the error in the nearest. Models are kept paired inside every resample.
    """
    d = pd.read_csv(DATA / "crop_fields.csv.gz")
    d = d[d.design.isin(["random", "block"])]
    key = ["design", "seed", "field"]
    wide = d.pivot(index=key, columns="model", values="pred")
    if "pheno" not in wide:
        return {}
    first = d[d.model == "tcn"].set_index(key).loc[wide.index]
    y, x, block = first.y.to_numpy(), first.distance_m.to_numpy(), first.block.to_numpy()
    models = ("rf", "tcn", "pheno")
    pred = {m: wide[m].to_numpy().astype(int) for m in models}
    b = _bins(x, CROP_EDGES_MATCHED)
    n = len(CROP_EDGES_MATCHED) - 1

    def f1_all(idx):
        return np.array([[macro_f1(y[idx][b[idx] == k], pred[m][idx][b[idx] == k], N_CROP_CLASSES) for k in range(n)]
                         for m in models])

    point, boot = f1_all(np.arange(len(y))), cluster_bootstrap(block, f1_all, n_boot, seed=11)   # (B, 3, n)
    ratio = lambda f: (1 - f[..., -1]) / (1 - f[..., 0])                                     # noqa: E731
    r_point, r_boot = ratio(point), ratio(boot)
    out = {"groups": CROP_EDGES_MATCHED[:-1], "n": [int((b == k).sum()) for k in range(n)], "n_clusters": int(len(np.unique(block))),
           "majority_macro_f1": MAJORITY_MACRO_F1, "models": {}}
    for i, m in enumerate(models):
        out["models"][m] = {"macro_f1": point[i].tolist(), "macro_f1_ci": interval(boot[:, i, :]),
                            "relative_error": ((1 - point[i]) / (1 - point[i, 0])).tolist(),
                            "relative_error_ci": interval((1 - boot[:, i, :]) / (1 - boot[:, i, [0]])),
                            "bin_centre": [float(np.median(x[b == k])) for k in range(n)],
                            "R": float(r_point[i]), "R_ci": interval(r_boot[:, i]),
                            "drop_macro_f1": float(point[i, 0] - point[i, -1]),
                            "drop_macro_f1_ci": interval(boot[:, i, 0] - boot[:, i, -1])}
    ph, rf, tcn = models.index("pheno"), models.index("rf"), models.index("tcn")
    guard_bar = (MAJORITY_MACRO_F1 + point[rf, 0]) / 2
    diff = lambda a, c: {"difference": float(r_point[a] - r_point[c]), "ci": interval(r_boot[:, a] - r_boot[:, c])}  # noqa: E731
    p1, p2 = diff(ph, rf), diff(ph, tcn)
    far = {"difference": float(point[ph, -1] - point[rf, -1]), "ci": interval(boot[:, ph, -1] - boot[:, rf, -1])}
    guard = bool(point[ph, 0] >= guard_bar)
    verdict = lambda t: bool(t["difference"] < 0 and t["ci"][1] < 0)                           # noqa: E731
    out["tests"] = {
        "G": {"pheno_near_macro_f1": float(point[ph, 0]), "bar": float(guard_bar), "passed": guard},
        "P1_R_pheno_minus_R_rf": {**p1, "supported": bool(guard and verdict(p1))},
        "P2_R_pheno_minus_R_tcn": {**p2, "supported": bool(guard and verdict(p2))},
        "P3_far_macro_f1_pheno_minus_rf": {**far, "supported": bool(far["ci"][1] >= 0)},
    }
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
             "views": views(n_boot), "runtime": runtime(n_boot), "structure_test": structure_test(n_boot)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(s, indent=2))
    return s
