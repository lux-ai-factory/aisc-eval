import numpy as np
import pandas as pd
from itertools import chain

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.prediction_metric_registry import prediction_metric


@prediction_metric(name="Classification Calibration metrics: ECE, MCE, SCE")
def classification_calibration_score_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    y_pred_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
    n_bins: int = 10,
) -> list[Measure]:
    """
    Computes ECE, MCE, and SCE.

    Parameters
    ----------
    datashape : DataShape
    model : Model
    dataset : Dataset
    y_pred_proba : np.ndarray

    Returns
    -------
    list[Measure] - A list of `Measure` objects:
    - "ECE": Expected Calibration Error is the weighted average absolute difference
        between predicted confidence and empirical accuracy across probability bins.
    - "MCE": Maximum Calibration Error is the largest absolute difference across bins.
    - "Calibration Error Bin" For each bin with data points within:
        - "num_in_bin_i": Number of data points win the ith bin
        - "x_i": Confidence in the ith bin
        - "y_i": Accuracy in the ith bin
    Parameters
    ----------
    datashape : DataShape
    model : Model
    dataset : Dataset
    y_pred_proba : np.ndarray

    Returns
    -------
    list[Measure] - A list of `Measure` objects (in the following order):
    - "Calibration Error Bin" For each bin with data points within:
        - "num_in_bin_i": Number of data points win the ith bin
        - "x_i": Confidence in the ith bin
        - "y_i": Accuracy in the ith bin
    - "SCE": Static Calibration Error is the weighted average of the absolute
        confidence–accuracy gap across probability bins and classes
    - "ECE": Expected Calibration Error is the weighted average absolute difference
        between predicted confidence and empirical accuracy across probability bins.
    - "MCE": Maximum Calibration Error is the largest absolute difference across bins.
    """
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()
    y_true = dataset.data[datashape.target.name].to_numpy()

    if y_pred is None:
        y_pred = np.argmax(y_pred_proba, axis=1)

    n_samples = y_true.shape[0]

    confidences = np.max(y_pred_proba, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracies = (y_pred == y_true).astype(np.float32)

    bin_counts, avg_conf, avg_acc, nonzero = _compute_bin_stats(
        confidences, accuracies, n_bins
    )

    per_bin_values = list(
        chain.from_iterable(
            (
                Measure(
                    name="Calibration Error Bin",
                    description=f"x_{index}",
                    score=conf_in_bin,
                    time=date,
                ),
                Measure(
                    name="Calibration Error Bin",
                    description=f"y_{index}",
                    score=acc_in_bin,
                    time=date,
                ),
                Measure(
                    name="Calibration Error Bin",
                    description=f"num_in_bin_{index}",
                    score=in_bin,
                    time=date,
                ),
            )
            for index, conf_in_bin, acc_in_bin, in_bin in zip(
                np.where(nonzero)[0],
                avg_conf[nonzero],
                avg_acc[nonzero],
                bin_counts[nonzero],
            )
        )
    )

    gap = np.abs(avg_acc - avg_conf)

    return [
        *per_bin_values,
        Measure(
            name="SCE",
            score=static_calibration_error(y_true, y_pred_proba, n_bins),
            time=date,
        ),
        Measure(
            name="ECE",
            score=expected_calibration_error(
                bin_counts, avg_conf, avg_acc, nonzero, n_samples, gap
            ),
            time=date,
        ),
        Measure(
            name="MCE",
            score=maximum_calibration_error(
                bin_counts, avg_conf, avg_acc, nonzero, gap
            ),
            time=date,
        ),
    ]


def _compute_bin_stats(confidences, accuracies, n_bins):
    """
    Returns per-bin counts, average confidence, average accuracy, and non-zero bins
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    conf_sum = np.bincount(bin_indices, weights=confidences, minlength=n_bins)
    acc_sum = np.bincount(bin_indices, weights=accuracies, minlength=n_bins)

    nonzero = bin_counts > 0

    avg_conf = np.zeros(n_bins)
    avg_acc = np.zeros(n_bins)

    avg_conf[nonzero] = conf_sum[nonzero] / bin_counts[nonzero]
    avg_acc[nonzero] = acc_sum[nonzero] / bin_counts[nonzero]

    return bin_counts, avg_conf, avg_acc, nonzero


def maximum_calibration_error(
    bin_counts, avg_conf, avg_acc, nonzero, gap: np.ndarray | None = None
):
    """
    Maximum Calibration Error (MCE)
    https://ojs.aaai.org/index.php/AAAI/article/view/9602
    """
    if gap is None:
        gap = np.abs(avg_acc - avg_conf)

    return np.max(gap[nonzero]) if np.any(nonzero) else 0.0


def expected_calibration_error(
    bin_counts, avg_conf, avg_acc, nonzero, n_samples, gap: np.ndarray | None = None
):
    """
    Expected Calibration Error (ECE)
    https://ojs.aaai.org/index.php/AAAI/article/view/9602
    """
    if gap is None:
        gap = np.abs(avg_acc - avg_conf)

    return np.sum((bin_counts[nonzero] / n_samples) * gap[nonzero])


def static_calibration_error(y_true, y_pred_proba, n_bins):
    """
    Static Calibration Error (SCE)
    https://arxiv.org/abs/1904.01685
    """
    n_samples, n_classes = y_pred_proba.shape
    sce = 0.0

    for c in range(n_classes):
        class_probs = y_pred_proba[:, c]
        class_acc = (y_true == c).astype(np.float32)

        bin_counts, avg_conf, avg_acc, nonzero = _compute_bin_stats(
            class_probs, class_acc, n_bins
        )

        gap = np.abs(avg_acc - avg_conf)
        sce += np.sum((bin_counts[nonzero] / n_samples) * gap[nonzero])

    return sce
