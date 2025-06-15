"""
A module for calculating correlation-based fitness metrics.

This module provides a single function, `calculate_correlation_metrics`, which
computes several standard correlation coefficients between two sequences of numbers.
These metrics are often used in scientific and machine learning contexts to
evaluate the "fitness" or agreement between predicted values and true values.

Dependencies:
    - numpy
    - scipy>=1.10 (for Chatterjee's Xi coefficient)
"""

from typing import NamedTuple, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

# Using a NamedTuple for a more descriptive and readable return type.
# The caller can access results by name (e.g., result.pearson) instead of
# just by index (e.g., result[0]).
class CorrelationMetrics(NamedTuple):
    """A container for holding multiple correlation metrics."""
    pearson: float
    spearman: float
    kendall: float
    chatterjee: float


def calculate_correlation_metrics(
    y_true: ArrayLike, y_pred: ArrayLike
) -> CorrelationMetrics:
    """
    Calculates a suite of correlation metrics between two numerical sequences.

    This function computes Pearson, Spearman, Kendall's Tau, and Chatterjee's Xi
    correlation coefficients, which together provide a comprehensive view of the
    linear, monotonic, and non-monotonic association between two variables.

    Args:
        y_true (ArrayLike): The ground truth or reference values.
            Must be a 1D sequence of numbers.
        y_pred (ArrayLike): The predicted or comparison values.
            Must be a 1D sequence of numbers of the same length as y_true.

    Returns:
        CorrelationMetrics: A named tuple containing the four calculated
        correlation coefficients: `(pearson, spearman, kendall, chatterjee)`.

    Raises:
        ValueError: If the input arrays are not 1-dimensional, have
                    mismatched lengths, or are empty.
    """
    # 1. Input validation and conversion
    try:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError("Inputs must be convertible to numeric arrays.") from e

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays.")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Input arrays must have the same shape. "
            f"Got {y_true_arr.shape} and {y_pred_arr.shape}."
        )

    if y_true_arr.size < 2: # Correlation is undefined for less than 2 points
        raise ValueError("Input arrays must contain at least two values.")

    # 2. Calculate metrics
    # Note: pearsonr, spearmanr, and kendalltau return a tuple (statistic, pvalue).
    # We only need the statistic, which is the first element.
    pearson_corr, _ = stats.pearsonr(y_true_arr, y_pred_arr)
    spearman_corr, _ = stats.spearmanr(y_true_arr, y_pred_arr)
    kendall_corr, _ = stats.kendalltau(y_true_arr, y_pred_arr)

    # chatterjeexi returns a SignificanceResult object. We need its .statistic attribute.
    chatterjee_corr = stats.chatterjeexi(y_true_arr, y_pred_arr).statistic

    # 3. Return results in a structured format
    return CorrelationMetrics(
        pearson=pearson_corr,
        spearman=spearman_corr,
        kendall=kendall_corr,
        chatterjee=chatterjee_corr,
    )