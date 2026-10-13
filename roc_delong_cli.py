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

def _validated_samples(scores, labels):
    """Validate parallel 1-D arrays; do not silently drop incomplete subjects."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.ndim != 1 or scores.size != labels.size:
        raise ValueError("Scores and labels must be one-dimensional arrays of equal length.")
    if not scores.size:
        raise ValueError("At least one observation is required.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be finite numbers (no NaN or infinity).")
    if np.unique(labels).size != 2:
        raise ValueError("Outcome labels must contain exactly two distinct classes.")
    return scores, labels


def _validated_score_vector(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain a nonempty one-dimensional array of finite scores.")
    return values


def split_by_label(scores, labels, positive_label):
    """Split complete, paired observations by positive class."""
    scores, labels = _validated_samples(scores, labels)
    pos_mask = labels == positive_label
    if not np.any(pos_mask) or np.all(pos_mask):
        raise ValueError("Both positive and negative subjects are required (check --positive-label).")
    return scores[pos_mask], scores[~pos_mask]


def roc_curve(scores, labels, positive_label):
    """Empirical ROC; thresholds descend from infinity, grouping tied scores."""
    scores, labels = _validated_samples(scores, labels)
    pos_mask = labels == positive_label
    n_pos = np.count_nonzero(pos_mask)
    n_neg = scores.size - n_pos
    if not n_pos or not n_neg:
        raise ValueError("ROC curve requires at least one subject in each class.")

    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ordered_positive = pos_mask[order]
    group_ends = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    thresholds = np.r_[np.inf, ordered_scores[group_ends]]
    tp = np.r_[0, np.cumsum(ordered_positive)[group_ends]]
    fp = np.r_[0, np.cumsum(~ordered_positive)[group_ends]]
    return thresholds, fp / n_neg, tp / n_pos



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
    """DeLong placement vectors with half-credit for ties, O(N log N) time.

    The two vectors preserve subject order for paired covariance estimation.
    """
    x = _validated_score_vector(pos_scores, "Positive scores")
    y = _validated_score_vector(neg_scores, "Negative scores")
    m, n = x.size, y.size
    sy = np.sort(y)
    sx = np.sort(x)

    neg_below = np.searchsorted(sy, x, side="left")
    neg_at_or_below = np.searchsorted(sy, x, side="right")
    v10 = (neg_below + neg_at_or_below) / (2.0 * n)

    pos_below = np.searchsorted(sx, y, side="left")
    pos_at_or_below = np.searchsorted(sx, y, side="right")
    v01 = 1.0 - (pos_below + pos_at_or_below) / (2.0 * m)
    return v10, v01, float(v10.mean())



def delong_auc_variance(v10, v01):
    """Estimated DeLong AUC variance; both outcome classes need >=2 subjects."""
    v10 = _validated_score_vector(v10, "Positive placements")
    v01 = _validated_score_vector(v01, "Negative placements")
    if min(v10.size, v01.size) < 2:
        raise ValueError("DeLong variance requires at least two subjects in each class.")
    return float(np.var(v10, ddof=1) / v10.size + np.var(v01, ddof=1) / v01.size)


def delong_ci(auc, variance, alpha=0.05):
    """Two-sided normal-approximation DeLong CI, truncated to [0, 1]."""
    if not (np.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between zero and one.")
    if not (np.isfinite(auc) and 0.0 <= auc <= 1.0):
        raise ValueError("AUC must lie between zero and one.")
    if not (np.isfinite(variance) and variance >= 0.0):
        raise ValueError("Estimated variance must be finite and nonnegative.")
    se = float(np.sqrt(variance))
    z = norm.ppf(1.0 - alpha / 2.0)
    return float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se)), se


def delong_test(pos_a, neg_a, pos_b, neg_b):
    """Paired, two-sided DeLong comparison (same subjects in the same order)."""
    pos_a = _validated_score_vector(pos_a, "Positive A scores")
    neg_a = _validated_score_vector(neg_a, "Negative A scores")
    pos_b = _validated_score_vector(pos_b, "Positive B scores")
    neg_b = _validated_score_vector(neg_b, "Negative B scores")
    if pos_a.size != pos_b.size or neg_a.size != neg_b.size:
        raise ValueError("Paired tests must have equal positive and negative sample counts.")
    if min(pos_a.size, neg_a.size) < 2:
        raise ValueError("DeLong comparison requires at least two subjects in each class.")

    v10_a, v01_a, auc_a = delong_components(pos_a, neg_a)
    v10_b, v01_b, auc_b = delong_components(pos_b, neg_b)
    m, n = pos_a.size, neg_a.size
    s10 = np.cov(np.vstack((v10_a, v10_b)), ddof=1)
    s01 = np.cov(np.vstack((v01_a, v01_b)), ddof=1)
    covariance = s10 / m + s01 / n
    diff = auc_a - auc_b

    # Estimate variance directly from within-subject placement differences,
    # avoiding catastrophic cancellation of nearly identical covariance terms.
    d10, d01 = v10_a - v10_b, v01_a - v01_b
    var_diff = float(np.var(d10, ddof=1) / m + np.var(d01, ddof=1) / n)
    if var_diff == 0.0:
        if abs(diff) > 1e-12:
            raise ValueError("Paired AUC difference has zero estimated variance; the DeLong z-test is undefined.")
        z, p_value = 0.0, 1.0
    else:
        z = float(diff / np.sqrt(var_diff))
        p_value = float(2.0 * norm.sf(abs(z)))

    return {
        "auc_a": auc_a, "auc_b": auc_b, "diff": float(diff),
        "var_diff": var_diff, "z": z, "p_value": p_value,
        "cov_matrix": covariance,
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
    except (ValueError, TypeError, OSError, csv.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
