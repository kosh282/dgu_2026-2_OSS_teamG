"""Dataset validation harness: statistics and orientation."""
import numpy as np
import pytest

from scripts.validate_dataset import best_threshold, roc_auc


def _brute_auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    wins = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def test_auc_perfect_separation():
    s = np.array([1, 2, 3, 4, 10, 11, 12, 13.0])
    l = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert roc_auc(s, l) == pytest.approx(1.0)
    assert roc_auc(-s, l) == pytest.approx(0.0)


def test_auc_all_ties_is_half():
    assert roc_auc(np.ones(8), np.array([0, 0, 0, 0, 1, 1, 1, 1])) == pytest.approx(0.5)


def test_auc_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(6, 25))
        sc = np.round(rng.normal(0, 1, n), 1)     # deliberate ties
        lb = rng.integers(0, 2, n)
        if lb.sum() in (0, n):
            continue
        assert roc_auc(sc, lb) == pytest.approx(_brute_auc(sc, lb), abs=1e-9)


def test_auc_undefined_with_one_class():
    assert np.isnan(roc_auc(np.array([1.0, 2.0]), np.array([1, 1])))


def test_best_threshold_perfect():
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    l = np.array([0, 0, 0, 1, 1, 1])
    b = best_threshold(s, l)
    assert b["sensitivity"] == 1.0 and b["specificity"] == 1.0 and b["youden"] == 1.0


def test_best_threshold_confusion_counts_add_up():
    s = np.array([0.1, 0.5, 0.3, 0.55, 0.9, 0.4])
    l = np.array([0, 0, 0, 1, 1, 1])
    b = best_threshold(s, l)
    assert b["tp"] + b["fn"] == 3
    assert b["tn"] + b["fp"] == 3


def test_lower_is_positive_orientation():
    """Hypomimia: LOW expressivity indicates the condition."""
    expr = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1])
    lab = np.array([0, 0, 0, 1, 1, 1])
    assert roc_auc(expr, lab) == pytest.approx(0.0)     # wrong orientation
    assert roc_auc(-expr, lab) == pytest.approx(1.0)    # flipped, as the CLI does
