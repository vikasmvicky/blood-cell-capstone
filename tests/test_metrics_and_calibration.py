import numpy as np
import pytest

from src.evaluation import calibration as cal
from src.evaluation import metrics as M

NAMES = ["a", "b", "c"]


def test_metrics_hand_computed():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    m = M.compute_metrics(y_true, y_pred, NAMES)
    assert m["test_accuracy"] == pytest.approx(4 / 6)
    # per-class precision: a=1/2, b=2/3, c=1/1 ; recall: a=1/2, b=1, c=1/2
    assert m["per_class"]["a"]["precision"] == pytest.approx(0.5)
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["c"]["recall"] == pytest.approx(0.5)
    assert m["macro_precision"] == pytest.approx((0.5 + 2 / 3 + 1) / 3)
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_perfect_predictions_and_input_validation():
    m = M.compute_metrics([0, 1, 2], [0, 1, 2], NAMES)
    assert m["test_accuracy"] == m["macro_f1"] == 1.0
    with pytest.raises(ValueError):
        M.compute_metrics([], [], NAMES)
    with pytest.raises(ValueError):
        M.compute_metrics([0, 1], [0], NAMES)


def test_top_confusions_ordering():
    df = M.top_confusions([[5, 3, 0], [1, 6, 0], [0, 0, 4]], NAMES, k=2)
    assert list(df.iloc[0][["true_class", "predicted_class", "count"]]) == ["a", "b", 3]
    assert len(df) == 2


def test_confusion_matrix_plot_written(tmp_path):
    M.plot_confusion_matrix([[2, 0], [1, 1]], ["x", "y"], tmp_path / "cm.png")
    assert (tmp_path / "cm.png").stat().st_size > 1000


def test_softmax_ece_nll_and_temperature():
    logits = np.array([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [4.0, 0.0]])
    labels = np.array([0, 1, 0, 1])                       # last prediction is wrong with high confidence
    p = cal.softmax(logits)
    assert np.allclose(p.sum(1), 1)
    assert 0 < cal.expected_calibration_error(p, labels) <= 1
    t = cal.fit_temperature(logits, labels)
    assert t > 1.0                                        # over-confident -> soften
    assert cal.negative_log_likelihood(cal.softmax(logits, t), labels) < cal.negative_log_likelihood(p, labels)
    assert (cal.softmax(logits, t).argmax(1) == p.argmax(1)).all()   # accuracy unchanged
