#!/usr/bin/env python3
"""
ROC DeLong CLI
==============

Computes an empirical ROC curve with the Youden-index optimal cutpoint for a
single diagnostic test / biomarker, and statistically compares two
correlated ROC curves (e.g. two biomarkers measured on the same subjects)
using the DeLong et al. (1988) nonparametric test.

Core methods implemented:
  - Empirical ROC curve via counting (equivalent to the trapezoidal rule
    over (FPR, TPR) points), with AUC computed as the Mann-Whitney U
    statistic (which is exactly the trapezoidal-rule AUC of the empirical
    ROC curve).
  - Youden index (sensitivity + specificity - 1) to select the optimal
    cutpoint.
  - DeLong (1988) covariance of AUCs via Mann-Whitney U "structural
    components", for a DeLong-method confidence interval on a single AUC
    and for a two-sample paired z-test comparing two correlated AUCs.
"""

import argparse
import csv
import sys

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Core ROC / Youden logic
# ---------------------------------------------------------------------------

def split_by_label(scores, labels, positive_label):
    """Split parallel score/label arrays into positive- and negative-class
    score arrays."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos_mask = labels == positive_label
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0:
        raise ValueError("No positive-class subjects found (check --positive-label).")
    if neg_mask.sum() == 0:
        raise ValueError("No negative-class subjects found (check --positive-label).")
    return scores[pos_mask], scores[neg_mask]


def roc_curve(scores, labels, positive_label):
    """Empirical ROC curve.

    Returns (thresholds, fpr, tpr) as 1-D numpy arrays, ordered so that
    threshold decreases from +inf down to the minimum observed score
    (i.e. FPR/TPR increase from (0,0) to (1,1)). A subject is classified
    positive when score >= threshold.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos_mask = labels == positive_label
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC curve requires at least one subject in each class.")

    uniq_desc = np.unique(scores)[::-1]
    thresholds = np.concatenate(([np.inf], uniq_desc))

    tpr = np.empty(thresholds.shape[0])
    fpr = np.empty(thresholds.shape[0])
    for i, t in enumerate(thresholds):
        predicted_pos = scores >= t
        tp = np.count_nonzero(predicted_pos & pos_mask)
        fp = np.count_nonzero(predicted_pos & neg_mask)
        tpr[i] = tp / n_pos
        fpr[i] = fp / n_neg
    return thresholds, fpr, tpr


_trapz = getattr(np, "trapezoid", None) or np.trapz


def auc_trapezoidal(fpr, tpr):
    """AUC via the trapezoidal rule over (FPR, TPR) points."""
    order = np.argsort(fpr, kind="mergesort")
    return float(_trapz(np.asarray(tpr)[order], np.asarray(fpr)[order]))


def youden_cutpoint(thresholds, fpr, tpr):
    """Find the threshold maximizing Youden's J = sensitivity + specificity - 1.

    Returns a dict with the cutpoint threshold, sensitivity, specificity and
    J statistic. The +inf threshold (the trivial all-negative point) is
    excluded from consideration.
    """
    sens = np.asarray(tpr, dtype=float)
    spec = 1.0 - np.asarray(fpr, dtype=float)
    j = sens + spec - 1.0

    finite_mask = np.isfinite(thresholds)
    idx_candidates = np.where(finite_mask)[0]
    best_local = idx_candidates[np.argmax(j[idx_candidates])]
    return {
        "threshold": float(thresholds[best_local]),
        "sensitivity": float(sens[best_local]),
        "specificity": float(spec[best_local]),
        "youden_j": float(j[best_local]),
    }


# ---------------------------------------------------------------------------
# DeLong (1988) method
# ---------------------------------------------------------------------------

def delong_components(pos_scores, neg_scores):
    """Mann-Whitney U structural components for one test (DeLong et al. 1988).

    pos_scores: 1-D array of scores for the m positive-class subjects.
    neg_scores: 1-D array of scores for the n negative-class subjects.

    Returns (V10, V01, auc) where:
      V10 has shape (m,): the average, over negatives, of psi(X_i, Y_j)
      V01 has shape (n,): the average, over positives, of psi(X_i, Y_j)
      auc is the Mann-Whitney U estimate of the AUC (mean of the psi matrix)

    psi(x, y) = 1 if x > y, 0.5 if x == y, 0 if x < y
    """
    x = np.asarray(pos_scores, dtype=float).reshape(-1, 1)  # (m, 1)
    y = np.asarray(neg_scores, dtype=float).reshape(1, -1)  # (1, n)
    diff = x - y
    psi = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))  # (m, n)

    v10 = psi.mean(axis=1)  # (m,)
    v01 = psi.mean(axis=0)  # (n,)
    auc = float(psi.mean())
    return v10, v01, auc


def delong_auc_variance(v10, v01):
    """DeLong variance estimate of a single AUC from its structural components."""
    m = v10.shape[0]
    n = v01.shape[0]
    s10 = np.var(v10, ddof=1) if m > 1 else 0.0
    s01 = np.var(v01, ddof=1) if n > 1 else 0.0
    return float(s10 / m + s01 / n)


def delong_ci(auc, variance, alpha=0.05):
    """Two-sided (1 - alpha) confidence interval for an AUC, clipped to [0, 1]."""
    se = np.sqrt(max(variance, 0.0))
    z = norm.ppf(1.0 - alpha / 2.0)
    lo = max(0.0, auc - z * se)
    hi = min(1.0, auc + z * se)
    return float(lo), float(hi), float(se)


def delong_test(pos_a, neg_a, pos_b, neg_b):
    """DeLong (1988) test comparing two correlated (paired) AUCs.

    pos_a, pos_b: scores for test A and test B on the SAME positive
                  subjects, in the same subject order (length m).
    neg_a, neg_b: scores for test A and test B on the SAME negative
                  subjects, in the same subject order (length n).

    Returns a dict with both AUCs, their difference, the DeLong covariance
    matrix, the z statistic and a two-sided p-value.
    """
    v10_a, v01_a, auc_a = delong_components(pos_a, neg_a)
    v10_b, v01_b, auc_b = delong_components(pos_b, neg_b)

    m = v10_a.shape[0]
    n = v01_a.shape[0]

    v10 = np.vstack([v10_a, v10_b])  # (2, m)
    v01 = np.vstack([v01_a, v01_b])  # (2, n)

    s10 = np.cov(v10, ddof=1) if m > 1 else np.zeros((2, 2))
    s01 = np.cov(v01, ddof=1) if n > 1 else np.zeros((2, 2))
    s = s10 / m + s01 / n  # (2, 2) covariance matrix of (auc_a, auc_b)

    diff = auc_a - auc_b
    var_diff = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])
    if var_diff <= 0.0:
        # Degenerate (e.g. identical tests): no distinguishable difference.
        z = 0.0
        p_value = 1.0
    else:
        z = float(diff / np.sqrt(var_diff))
        p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "diff": float(diff),
        "var_diff": var_diff,
        "z": z,
        "p_value": p_value,
        "cov_matrix": s,
    }


# ---------------------------------------------------------------------------
# CSV I/O helpers
# ---------------------------------------------------------------------------

def read_columns(path, columns):
    """Read named columns from a CSV file into a dict of lists (as strings)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"'{path}' has no header row.")
        missing = [c for c in columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Column(s) {missing} not found in '{path}'. "
                f"Available columns: {reader.fieldnames}"
            )
        data = {c: [] for c in columns}
        for row in reader:
            for c in columns:
                data[c].append(row[c])
    if not data[columns[0]]:
        raise ValueError(f"'{path}' contains no data rows.")
    return data


def parse_labels(raw_labels, positive_label):
    """Convert raw label strings to a numpy array, matching positive_label
    either as a string or as a number."""
    try:
        pos_val = float(positive_label)
        try:
            return np.array([float(v) for v in raw_labels]), pos_val
        except ValueError:
            pass
    except ValueError:
        pass
    return np.array(raw_labels), positive_label


def write_roc_csv(path, rows):
    """rows: list of dicts with identical keys."""
    if not rows:
        raise ValueError("Nothing to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def roc_table_rows(test_name, thresholds, fpr, tpr, youden):
    rows = []
    for t, fp, tp in zip(thresholds, fpr, tpr):
        sens = tp
        spec = 1.0 - fp
        rows.append({
            "test": test_name,
            "threshold": t,
            "sensitivity": sens,
            "specificity": spec,
            "fpr": fp,
            "youden_j": sens + spec - 1.0,
            "is_optimal_cutpoint": (
                np.isfinite(t) and abs(t - youden["threshold"]) < 1e-12
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_roc(path, fpr, tpr, auc, youden, test_name="Test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="tab:blue", lw=2,
            label=f"{test_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1, label="Chance")

    cut_fpr = 1.0 - youden["specificity"]
    cut_tpr = youden["sensitivity"]
    ax.scatter([cut_fpr], [cut_tpr], color="tab:red", zorder=5, s=70,
               label=f"Youden cutpoint = {youden['threshold']:.3g}\n"
                     f"(Sens={youden['sensitivity']:.3f}, "
                     f"Spec={youden['specificity']:.3f})")

    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC Curve")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare_roc(path, fpr_a, tpr_a, auc_a, youden_a, name_a,
                      fpr_b, tpr_b, auc_b, youden_b, name_b, p_value):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(fpr_a, tpr_a, color="tab:blue", lw=2,
            label=f"{name_a} (AUC = {auc_a:.3f})")
    ax.plot(fpr_b, tpr_b, color="tab:orange", lw=2,
            label=f"{name_b} (AUC = {auc_b:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1, label="Chance")

    ax.scatter([1.0 - youden_a["specificity"]], [youden_a["sensitivity"]],
               color="tab:blue", edgecolor="k", zorder=5, s=70,
               label=f"{name_a} cutpoint = {youden_a['threshold']:.3g}")
    ax.scatter([1.0 - youden_b["specificity"]], [youden_b["sensitivity"]],
               color="tab:orange", edgecolor="k", zorder=5, s=70,
               label=f"{name_b} cutpoint = {youden_b['threshold']:.3g}")

    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(f"ROC Comparison (DeLong p = {p_value:.4g})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def cmd_compute(args):
    data = read_columns(args.data, [args.label_col, args.score_col])
    labels, pos_label = parse_labels(data[args.label_col], args.positive_label)
    scores = np.array([float(v) for v in data[args.score_col]])

    thresholds, fpr, tpr = roc_curve(scores, labels, pos_label)
    auc = auc_trapezoidal(fpr, tpr)

    pos_scores, neg_scores = split_by_label(scores, labels, pos_label)
    v10, v01, auc_mw = delong_components(pos_scores, neg_scores)
    variance = delong_auc_variance(v10, v01)
    ci_lo, ci_hi, se = delong_ci(auc_mw, variance, args.alpha)

    youden = youden_cutpoint(thresholds, fpr, tpr)

    print(f"n_positive = {pos_scores.shape[0]}, n_negative = {neg_scores.shape[0]}")
    print(f"AUC (trapezoidal)      = {auc:.4f}")
    print(f"AUC (Mann-Whitney U)   = {auc_mw:.4f}")
    print(f"DeLong SE(AUC)         = {se:.4f}")
    print(f"{int((1 - args.alpha) * 100)}% CI (DeLong)        = "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]")
    print("Youden-optimal cutpoint:")
    print(f"  threshold    = {youden['threshold']:.6g}")
    print(f"  sensitivity  = {youden['sensitivity']:.4f}")
    print(f"  specificity  = {youden['specificity']:.4f}")
    print(f"  Youden's J   = {youden['youden_j']:.4f}")

    if args.output_csv:
        rows = roc_table_rows(args.score_col, thresholds, fpr, tpr, youden)
        write_roc_csv(args.output_csv, rows)
        print(f"Wrote sensitivity/specificity table to '{args.output_csv}'")

    if args.plot:
        plot_single_roc(args.plot, fpr, tpr, auc_mw, youden, args.score_col)
        print(f"Wrote ROC plot to '{args.plot}'")


def cmd_compare(args):
    data = read_columns(
        args.data, [args.label_col, args.score_a, args.score_b]
    )
    labels, pos_label = parse_labels(data[args.label_col], args.positive_label)
    scores_a = np.array([float(v) for v in data[args.score_a]])
    scores_b = np.array([float(v) for v in data[args.score_b]])

    pos_mask = labels == pos_label
    neg_mask = ~pos_mask
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        raise ValueError("Both classes must be present (check --positive-label).")

    pos_a, neg_a = scores_a[pos_mask], scores_a[neg_mask]
    pos_b, neg_b = scores_b[pos_mask], scores_b[neg_mask]

    result = delong_test(pos_a, neg_a, pos_b, neg_b)

    th_a, fpr_a, tpr_a = roc_curve(scores_a, labels, pos_label)
    th_b, fpr_b, tpr_b = roc_curve(scores_b, labels, pos_label)
    youden_a = youden_cutpoint(th_a, fpr_a, tpr_a)
    youden_b = youden_cutpoint(th_b, fpr_b, tpr_b)

    print(f"n_positive = {pos_mask.sum()}, n_negative = {neg_mask.sum()}")
    print(f"AUC {args.score_a} = {result['auc_a']:.4f}  "
          f"(Youden cutpoint {youden_a['threshold']:.6g}, "
          f"Sens={youden_a['sensitivity']:.3f}, Spec={youden_a['specificity']:.3f})")
    print(f"AUC {args.score_b} = {result['auc_b']:.4f}  "
          f"(Youden cutpoint {youden_b['threshold']:.6g}, "
          f"Sens={youden_b['sensitivity']:.3f}, Spec={youden_b['specificity']:.3f})")
    print(f"AUC difference ({args.score_a} - {args.score_b}) = {result['diff']:.4f}")
    print("DeLong test for two correlated ROC curves:")
    print(f"  z statistic = {result['z']:.4f}")
    print(f"  p-value     = {result['p_value']:.4g}")

    if args.output_csv:
        rows = roc_table_rows(args.score_a, th_a, fpr_a, tpr_a, youden_a)
        rows += roc_table_rows(args.score_b, th_b, fpr_b, tpr_b, youden_b)
        write_roc_csv(args.output_csv, rows)
        print(f"Wrote sensitivity/specificity table to '{args.output_csv}'")

    if args.plot:
        plot_compare_roc(
            args.plot, fpr_a, tpr_a, result["auc_a"], youden_a, args.score_a,
            fpr_b, tpr_b, result["auc_b"], youden_b, args.score_b,
            result["p_value"],
        )
        print(f"Wrote ROC comparison plot to '{args.plot}'")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="roc_delong_cli",
        description="Empirical ROC/AUC with Youden cutpoint, and DeLong "
                     "comparison of two correlated ROC curves.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_compute = sub.add_parser(
        "compute",
        help="Compute the ROC curve, DeLong CI, and Youden cutpoint for one test.",
    )
    p_compute.add_argument("--data", required=True, help="Path to input CSV file.")
    p_compute.add_argument("--label-col", required=True,
                            help="Column name holding the true class label.")
    p_compute.add_argument("--score-col", required=True,
                            help="Column name holding the test/biomarker score.")
    p_compute.add_argument("--positive-label", default="1",
                            help="Label value that marks the positive class (default: 1).")
    p_compute.add_argument("--alpha", type=float, default=0.05,
                            help="Significance level for the DeLong CI (default: 0.05).")
    p_compute.add_argument("--output-csv", default=None,
                            help="Path to write the full sensitivity/specificity table.")
    p_compute.add_argument("--plot", default=None,
                            help="Path to write the ROC plot (e.g. roc.png).")
    p_compute.set_defaults(func=cmd_compute)

    p_compare = sub.add_parser(
        "compare",
        help="DeLong test comparing two correlated (paired) ROC curves.",
    )
    p_compare.add_argument("--data", required=True, help="Path to input CSV file.")
    p_compare.add_argument("--label-col", required=True,
                            help="Column name holding the true class label.")
    p_compare.add_argument("--score-a", required=True,
                            help="Column name holding scores for test/biomarker A.")
    p_compare.add_argument("--score-b", required=True,
                            help="Column name holding scores for test/biomarker B.")
    p_compare.add_argument("--positive-label", default="1",
                            help="Label value that marks the positive class (default: 1).")
    p_compare.add_argument("--alpha", type=float, default=0.05,
                            help="Significance level (default: 0.05, unused directly "
                                 "but kept for symmetry with 'compute').")
    p_compare.add_argument("--output-csv", default=None,
                            help="Path to write the combined sensitivity/specificity table.")
    p_compare.add_argument("--plot", default=None,
                            help="Path to write the ROC comparison plot (e.g. roc_compare.png).")
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
