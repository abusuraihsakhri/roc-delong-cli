"""
Tests for roc_delong_cli.py core logic (ROC/AUC, Youden index, DeLong test).

Run with: python test_roc_delong_cli.py
"""

import numpy as np

from roc_delong_cli import (
    roc_curve,
    auc_trapezoidal,
    youden_cutpoint,
    delong_components,
    delong_auc_variance,
    delong_ci,
    delong_test,
    split_by_label,
)


def test_perfect_separation():
    # Positives strictly greater than negatives -> AUC = 1, perfect cutpoint.
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([10, 9, 8, 3, 2, 1], dtype=float)

    thresholds, fpr, tpr = roc_curve(scores, labels, 1)
    auc = auc_trapezoidal(fpr, tpr)
    assert abs(auc - 1.0) < 1e-9, f"expected AUC=1.0, got {auc}"

    pos, neg = split_by_label(scores, labels, 1)
    _, _, auc_mw = delong_components(pos, neg)
    assert abs(auc_mw - 1.0) < 1e-9

    y = youden_cutpoint(thresholds, fpr, tpr)
    assert abs(y["sensitivity"] - 1.0) < 1e-9
    assert abs(y["specificity"] - 1.0) < 1e-9
    assert abs(y["youden_j"] - 1.0) < 1e-9
    # Cutpoint must separate the two clusters: neg max = 3, pos min = 8.
    assert 3 < y["threshold"] <= 8


def test_known_manual_auc():
    # Hand-computed example: positives = [3, 5], negatives = [1, 2, 4].
    # Concordant pairs (pos > neg): (3,1),(3,2),(5,1),(5,2),(5,4) = 5
    # Discordant: (3,4) = 1  -> AUC = 5/6
    pos = np.array([3.0, 5.0])
    neg = np.array([1.0, 2.0, 4.0])

    v10, v01, auc_mw = delong_components(pos, neg)
    assert abs(auc_mw - 5.0 / 6.0) < 1e-9, f"expected 5/6, got {auc_mw}"

    labels = np.array([1, 1, 0, 0, 0])
    scores = np.array([3.0, 5.0, 1.0, 2.0, 4.0])
    thresholds, fpr, tpr = roc_curve(scores, labels, 1)
    auc_trap = auc_trapezoidal(fpr, tpr)
    assert abs(auc_trap - 5.0 / 6.0) < 1e-9, f"trapezoid AUC mismatch: {auc_trap}"

    # DeLong variance/CI should be finite and CI should contain the point estimate.
    var = delong_auc_variance(v10, v01)
    lo, hi, se = delong_ci(auc_mw, var, alpha=0.05)
    assert lo <= auc_mw <= hi
    assert se >= 0.0


def test_delong_identical_tests_no_difference():
    # Same scores used for both "tests" -> AUC diff must be exactly 0,
    # z must be 0, p-value must be 1 (degenerate zero-variance case).
    rng = np.random.RandomState(0)
    pos = rng.normal(loc=1.5, scale=1.0, size=20)
    neg = rng.normal(loc=0.0, scale=1.0, size=25)

    result = delong_test(pos, neg, pos, neg)
    assert abs(result["diff"]) < 1e-12
    assert abs(result["z"]) < 1e-9
    assert abs(result["p_value"] - 1.0) < 1e-9
    assert abs(result["auc_a"] - result["auc_b"]) < 1e-12


def test_delong_detects_clearly_better_test():
    # Test A perfectly separates the classes; test B is pure noise
    # (same value for everyone -> AUC = 0.5). The paired DeLong test should
    # find a large, significant difference in favor of A.
    n_pos, n_neg = 30, 30
    rng = np.random.RandomState(42)

    pos_a = rng.normal(loc=5.0, scale=0.2, size=n_pos)
    neg_a = rng.normal(loc=0.0, scale=0.2, size=n_neg)  # clean separation

    pos_b = rng.normal(loc=0.0, scale=1.0, size=n_pos)
    neg_b = rng.normal(loc=0.0, scale=1.0, size=n_neg)  # same distribution

    result = delong_test(pos_a, neg_a, pos_b, neg_b)
    assert result["auc_a"] > 0.95, f"expected near-perfect AUC_a, got {result['auc_a']}"
    assert result["diff"] > 0.3
    assert result["p_value"] < 0.01, f"expected significant p-value, got {result['p_value']}"


def test_youden_matches_definition():
    # Verify Youden's J is literally sensitivity + specificity - 1 at the
    # returned cutpoint, and that it is the maximum over all thresholds.
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([5, 4, 3, 1, 6, 2, 1, 0], dtype=float)

    thresholds, fpr, tpr = roc_curve(scores, labels, 1)
    sens_all = tpr
    spec_all = 1 - fpr
    j_all = sens_all + spec_all - 1

    y = youden_cutpoint(thresholds, fpr, tpr)
    assert abs(y["youden_j"] - (y["sensitivity"] + y["specificity"] - 1)) < 1e-12
    assert abs(y["youden_j"] - j_all.max()) < 1e-12


if __name__ == "__main__":
    tests = [
        test_perfect_separation,
        test_known_manual_auc,
        test_delong_identical_tests_no_difference,
        test_delong_detects_clearly_better_test,
        test_youden_matches_definition,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
