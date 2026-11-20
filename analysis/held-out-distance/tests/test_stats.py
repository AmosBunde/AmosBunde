import numpy as np
import pandas as pd

from heldout_distance import analysis
from heldout_distance.stats import cluster_bootstrap, interval, macro_f1, spearman


def test_macro_f1_known_value():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    # class 0: tp 1, fp 1, fn 1 -> 0.5; class 1: tp 2, fp 1, fn 0 -> 0.8; class 2: tp 1, fp 0, fn 1 -> 2/3
    assert abs(macro_f1(y, pred, 3) - (0.5 + 0.8 + 2 / 3) / 3) < 1e-12


def test_macro_f1_counts_absent_class_as_zero():
    y = np.array([0, 0, 1, 1])
    assert abs(macro_f1(y, y, 3) - 2 / 3) < 1e-12


def test_macro_f1_perfect_and_worst():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y, y, 3) == 1.0
    assert macro_f1(y, (y + 1) % 3, 3) == 0.0


def test_spearman_monotone_and_ties():
    assert abs(spearman(np.arange(5), np.array([1, 2, 4, 8, 16])) - 1.0) < 1e-12
    assert abs(spearman(np.arange(5), -np.arange(5)) + 1.0) < 1e-12
    assert abs(spearman(np.array([1, 2, 3, 4]), np.array([1, 1, 2, 2])) - 0.894427190999916) < 1e-9
    assert np.isnan(spearman(np.arange(4), np.ones(4)))


def test_cluster_bootstrap_keeps_clusters_whole():
    clusters = np.repeat(np.arange(10), 5)
    seen = []
    cluster_bootstrap(clusters, lambda idx: seen.append(idx) or 0.0, n_boot=20, seed=0)
    for idx in seen:
        assert len(idx) == 50
        counts = np.bincount(clusters[idx], minlength=10)
        assert (counts % 5 == 0).all()


def test_cluster_bootstrap_is_deterministic_and_wider_than_naive():
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 1, 30)
    values = np.repeat(centres, 20) + rng.normal(0, 0.1, 600)
    clusters = np.repeat(np.arange(30), 20)
    a = cluster_bootstrap(clusters, lambda i: values[i].mean(), 500, seed=1)
    b = cluster_bootstrap(clusters, lambda i: values[i].mean(), 500, seed=1)
    naive = cluster_bootstrap(np.arange(600), lambda i: values[i].mean(), 500, seed=1)
    assert np.array_equal(a, b)
    assert a.std() > 3 * naive.std()


def test_interval_brackets_the_mean():
    lo, hi = interval(np.random.default_rng(0).normal(5, 1, 4000))
    assert 2.9 < lo < 3.2 and 6.8 < hi < 7.1


def test_bins_put_edges_in_the_lower_group():
    b = analysis._bins(np.array([10.0, 50.0, 50.1, 1e7]), [0, 50, 100, 1e9])
    assert b.tolist() == [0, 0, 1, 2]


def test_curve_recovers_a_planted_trend():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 4, 4000)
    b = analysis._bins(x, [0, 1, 2, 3, 4])
    err = 0.1 * (1 + b) + rng.normal(0, 0.01, 4000)
    c = analysis._curve(lambda idx: np.array([err[idx][b[idx] == k].mean() for k in range(4)]), x, b, 4,
                        rng.integers(0, 50, 4000), n_boot=200, seed=0)
    assert abs(c["last_over_first"] - 4.0) < 0.1 and c["trend_spearman"] > 0.999
    assert c["last_over_first_ci"][0] < 4.0 < c["last_over_first_ci"][1]
    assert c["relative"][0] == 1.0


def test_committed_extracts_have_the_expected_shape():
    crop = pd.read_csv(analysis.DATA / "crop_fields.csv.gz")
    assert set(crop.design) >= {"random", "block", "tile"} and crop.y.between(0, 6).all() and (crop.distance_m > 0).all()
    assert crop.groupby(["design", "seed"]).size().nunique() == 1          # every field once per design and seed
    views = pd.read_csv(analysis.DATA / "view_tests.csv.gz")
    assert views.run_id.nunique() == 18 and len(views) == 18 * 24
    runtime = pd.read_csv(analysis.DATA / "runtime_cells.csv.gz")
    assert (runtime[runtime.design == "size"].extrapolation > 1).all() and (runtime.measured > 0).all()
