"""Regression tests for numerical and CLI edge cases."""
import numpy as np
import pytest
from roc_delong_cli import (
    roc_curve, auc_trapezoidal, delong_components, delong_auc_variance,
    delong_ci, delong_test, split_by_label, main,
)


def test_roc_ties_and_large_sample():
    scores = np.repeat(np.arange(1000, dtype=float), 4)
    labels = np.tile([0, 1, 0, 1], 1000)
    th, fpr, tpr = roc_curve(scores, labels, 1)
    assert len(th) == 1001
    assert (fpr[0], tpr[0]) == (0, 0)
    assert (fpr[-1], tpr[-1]) == (1, 1)
    assert auc_trapezoidal(fpr, tpr) == pytest.approx(0.5)


def test_delong_components_match_pairwise_tie_reference():
    pos = np.array([2., 2., 0., -3.])
    neg = np.array([2., 1., -3., -3., 5.])
    psi = (pos[:, None] > neg[None, :]).astype(float)
    psi += 0.5 * (pos[:, None] == neg[None, :])
    v10, v01, auc = delong_components(pos, neg)
    assert np.allclose(v10, psi.mean(axis=1))
    assert np.allclose(v01, psi.mean(axis=0))
    assert auc == pytest.approx(psi.mean())


@pytest.mark.parametrize("scores,labels", [
    ([0.1, float("nan")], [0, 1]),
    ([float("inf"), 0.1], [0, 1]),
    ([0.1, 0.3], [1]),
    ([[0.1, 0.3]], [1, 0]),
])
def test_invalid_scores_rejected(scores, labels):
    with pytest.raises(ValueError):
        roc_curve(scores, labels, 1)


def test_paired_length_mismatch_and_minimum():
    with pytest.raises(ValueError, match="equal"):
        delong_test([1, 2], [0, 1], [1], [0, 1])
    with pytest.raises(ValueError, match="at least two"):
        delong_test([1], [0, 1], [1], [0, 1])
    with pytest.raises(ValueError, match="at least two"):
        delong_auc_variance(np.array([1.]), np.array([0., 1.]))


def test_zero_variance_nonzero_difference_is_not_p_one():
    with pytest.raises(ValueError, match="undefined"):
        delong_test([1., 1.], [0., 0.], [0., 0.], [1., 1.])


@pytest.mark.parametrize("alpha", [0, 1, -0.2, float("nan")])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        delong_ci(0.5, 0.1, alpha)


def test_cli_invalid_scores_report_error(tmp_path, capsys):
    csv = tmp_path / "bad.csv"
    csv.write_text("y,s\n1,0.2\n0,nan\n")
    with pytest.raises(SystemExit) as exc:
        main(["compute", "--data", str(csv), "--label-col", "y", "--score-col", "s"])
    assert exc.value.code == 1
    assert "finite" in capsys.readouterr().err.lower()


def test_multiclass_labels_rejected():
    with pytest.raises(ValueError, match="exactly two"):
        roc_curve([0.1, 0.2, 0.3], ["A", "B", "C"], "A")


def test_batch_cannot_overwrite_input(tmp_path, monkeypatch, capsys):
    from cli import main as audit_main
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "records.csv"
    content = "task_id,target_identifier,primary_metric\\nT1,KEY-01,5\\n"
    path.write_text(content)
    status = audit_main(["batch", "-i", "records.csv", "-o", "records.csv"])
    assert status == 1
    assert path.read_text() == content
    assert "must differ" in capsys.readouterr().err
