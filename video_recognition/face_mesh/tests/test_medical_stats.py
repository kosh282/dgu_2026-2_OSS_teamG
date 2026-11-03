"""Bootstrap confidence intervals and autocorrelation handling."""
import numpy as np
import pytest

from facemesh.medical.stats import (
    autocorrelation_time, block_bootstrap_ci, bootstrap_ci,
    effective_n, suggest_block_size, summarize,
)


def _ar1(n, phi, rng, mu=0.5, sigma=0.1):
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x + mu


def test_iid_bootstrap_coverage_near_nominal():
    rng = np.random.default_rng(0)
    hits, N = 0, 200
    for _ in range(N):
        x = rng.normal(0.5, 0.1, 60)
        ci = bootstrap_ci(x, n_boot=300, seed=int(rng.integers(1e6)))
        hits += ci.low <= 0.5 <= ci.high
    assert 0.88 <= hits / N <= 0.99


def test_naive_bootstrap_undercovers_autocorrelated_data():
    """The bug this module exists to prevent: falsely narrow intervals."""
    rng = np.random.default_rng(1)
    hits, N = 0, 120
    for _ in range(N):
        ci = bootstrap_ci(_ar1(200, 0.9, rng), n_boot=250, seed=int(rng.integers(1e6)))
        hits += ci.low <= 0.5 <= ci.high
    assert hits / N < 0.7          # demonstrably broken for time series


def test_block_bootstrap_beats_naive_on_autocorrelated_data():
    rng = np.random.default_rng(2)
    hn = hb = 0
    wn, wb = [], []
    N = 120
    for _ in range(N):
        x = _ar1(200, 0.9, rng)
        s = int(rng.integers(1e6))
        cn = bootstrap_ci(x, n_boot=250, seed=s)
        cb = block_bootstrap_ci(x, n_boot=250, seed=s)
        hn += cn.low <= 0.5 <= cn.high
        hb += cb.low <= 0.5 <= cb.high
        wn.append(cn.width)
        wb.append(cb.width)
    assert hb > hn
    assert np.mean(wb) > np.mean(wn)


def test_block_size_grows_with_autocorrelation():
    rng = np.random.default_rng(3)
    b_indep = suggest_block_size(rng.normal(0, 1, 240))
    b_corr = suggest_block_size(_ar1(240, 0.9, rng))
    assert b_corr > b_indep


def test_block_size_capped():
    rng = np.random.default_rng(4)
    x = _ar1(60, 0.99, rng)
    assert suggest_block_size(x) <= max(1, len(x) // 3)


def test_autocorrelation_time():
    rng = np.random.default_rng(5)
    assert autocorrelation_time(rng.normal(0, 1, 300)) < 2.0
    assert autocorrelation_time(_ar1(300, 0.9, rng)) > 5.0


def test_effective_n_shrinks_with_correlation():
    rng = np.random.default_rng(6)
    n_indep = effective_n(rng.normal(0, 1, 300))
    n_corr = effective_n(_ar1(300, 0.9, rng))
    assert n_corr < n_indep / 3


def test_edge_cases():
    empty = bootstrap_ci([])
    assert empty.n == 0 and np.isnan(empty.estimate)

    single = bootstrap_ci([0.7])
    assert single.n == 1 and single.width == 0 and single.method == "degenerate"

    const = bootstrap_ci([0.5] * 30)
    assert const.width == pytest.approx(0.0)


def test_ci_brackets_the_estimate():
    rng = np.random.default_rng(7)
    ci = block_bootstrap_ci(_ar1(200, 0.7, rng), n_boot=300)
    assert ci.low <= ci.estimate <= ci.high


def test_custom_statistic():
    ci = bootstrap_ci([1, 2, 3, 4, 100], statistic=np.median, n_boot=300)
    assert 1.0 <= ci.estimate <= 4.0     # median is robust to the outlier


def test_summarize_reports_effective_n():
    rng = np.random.default_rng(8)
    out = summarize(_ar1(240, 0.85, rng))
    assert out["method"] == "block-bootstrap"
    assert out["n"] == 240
    assert out["n_effective"] < 100      # autocorrelation costs information
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"]
