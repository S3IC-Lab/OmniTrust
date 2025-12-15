"""
Evaluation Metrics
==================

Provides standard metrics computation utilities for trustworthiness evaluation.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math


@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metrics with automatic computation.

    Attributes:
        predictions: Model predictions
        ground_truth: Ground truth labels
        computed_metrics: Dictionary of computed metrics
    """
    predictions: List[Any] = field(default_factory=list)
    ground_truth: List[Any] = field(default_factory=list)
    computed_metrics: Dict[str, float] = field(default_factory=dict)

    def compute_all(self) -> Dict[str, float]:
        """
        Compute all available metrics.

        Returns:
            Dictionary of all computed metrics
        """
        if not self.predictions or not self.ground_truth:
            return {}

        self.computed_metrics = {
            'accuracy': compute_accuracy(self.predictions, self.ground_truth),
        }

        # For binary/multiclass classification
        if self._is_classification():
            self.computed_metrics.update({
                'precision': compute_precision(self.predictions, self.ground_truth),
                'recall': compute_recall(self.predictions, self.ground_truth),
                'f1': compute_f1(self.predictions, self.ground_truth),
            })

        return self.computed_metrics

    def _is_classification(self) -> bool:
        """Check if this is a classification task."""
        unique_preds = set(self.predictions)
        unique_truth = set(self.ground_truth)
        return len(unique_preds) < 50 and len(unique_truth) < 50

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'predictions_count': len(self.predictions),
            'ground_truth_count': len(self.ground_truth),
            'metrics': self.computed_metrics
        }


def compute_accuracy(
    predictions: List[Any],
    ground_truth: List[Any]
) -> float:
    """
    Compute accuracy metric.

    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels

    Returns:
        Accuracy score (0-1)
    """
    if not predictions or not ground_truth:
        return 0.0

    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def compute_precision(
    predictions: List[Any],
    ground_truth: List[Any],
    positive_label: Any = 1,
    average: str = 'binary'
) -> float:
    """
    Compute precision metric.

    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels
        positive_label: Label for positive class (binary classification)
        average: Averaging method ('binary', 'macro', 'weighted')

    Returns:
        Precision score (0-1)
    """
    if not predictions or not ground_truth:
        return 0.0

    if average == 'binary':
        true_positives = sum(
            p == positive_label and g == positive_label
            for p, g in zip(predictions, ground_truth)
        )
        predicted_positives = sum(p == positive_label for p in predictions)

        if predicted_positives == 0:
            return 0.0
        return true_positives / predicted_positives

    elif average in ['macro', 'weighted']:
        labels = set(ground_truth)
        precisions = []
        weights = []

        for label in labels:
            true_positives = sum(
                p == label and g == label
                for p, g in zip(predictions, ground_truth)
            )
            predicted_positives = sum(p == label for p in predictions)

            if predicted_positives > 0:
                precisions.append(true_positives / predicted_positives)
            else:
                precisions.append(0.0)

            weights.append(sum(g == label for g in ground_truth))

        if average == 'macro':
            return sum(precisions) / len(precisions) if precisions else 0.0
        else:  # weighted
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            return sum(p * w for p, w in zip(precisions, weights)) / total_weight

    return 0.0


def compute_recall(
    predictions: List[Any],
    ground_truth: List[Any],
    positive_label: Any = 1,
    average: str = 'binary'
) -> float:
    """
    Compute recall metric.

    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels
        positive_label: Label for positive class (binary classification)
        average: Averaging method ('binary', 'macro', 'weighted')

    Returns:
        Recall score (0-1)
    """
    if not predictions or not ground_truth:
        return 0.0

    if average == 'binary':
        true_positives = sum(
            p == positive_label and g == positive_label
            for p, g in zip(predictions, ground_truth)
        )
        actual_positives = sum(g == positive_label for g in ground_truth)

        if actual_positives == 0:
            return 0.0
        return true_positives / actual_positives

    elif average in ['macro', 'weighted']:
        labels = set(ground_truth)
        recalls = []
        weights = []

        for label in labels:
            true_positives = sum(
                p == label and g == label
                for p, g in zip(predictions, ground_truth)
            )
            actual_positives = sum(g == label for g in ground_truth)

            if actual_positives > 0:
                recalls.append(true_positives / actual_positives)
            else:
                recalls.append(0.0)

            weights.append(actual_positives)

        if average == 'macro':
            return sum(recalls) / len(recalls) if recalls else 0.0
        else:  # weighted
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            return sum(r * w for r, w in zip(recalls, weights)) / total_weight

    return 0.0


def compute_f1(
    predictions: List[Any],
    ground_truth: List[Any],
    positive_label: Any = 1,
    average: str = 'binary'
) -> float:
    """
    Compute F1 score.

    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels
        positive_label: Label for positive class (binary classification)
        average: Averaging method ('binary', 'macro', 'weighted')

    Returns:
        F1 score (0-1)
    """
    precision = compute_precision(predictions, ground_truth, positive_label, average)
    recall = compute_recall(predictions, ground_truth, positive_label, average)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_auc_roc(
    predictions: List[float],
    ground_truth: List[int],
    positive_label: int = 1
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        predictions: Predicted probabilities for positive class
        ground_truth: Binary ground truth labels
        positive_label: Label for positive class

    Returns:
        AUC-ROC score (0-1)
    """
    if not predictions or not ground_truth:
        return 0.0

    # Create list of (prediction, label) pairs
    pairs = list(zip(predictions, ground_truth))

    # Count positives and negatives
    n_pos = sum(g == positive_label for g in ground_truth)
    n_neg = len(ground_truth) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Sort by prediction in descending order
    pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)

    # Compute AUC using the Wilcoxon-Mann-Whitney statistic
    auc = 0.0
    cumulative_pos = 0

    for pred, label in pairs_sorted:
        if label == positive_label:
            cumulative_pos += 1
        else:
            auc += cumulative_pos

    return auc / (n_pos * n_neg)


def compute_confusion_matrix(
    predictions: List[Any],
    ground_truth: List[Any],
    labels: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Compute confusion matrix.

    Args:
        predictions: Model predictions
        ground_truth: Ground truth labels
        labels: Optional list of labels to include

    Returns:
        Dictionary with confusion matrix and related metrics
    """
    if not predictions or not ground_truth:
        return {}

    if labels is None:
        labels = sorted(set(ground_truth) | set(predictions))

    n_labels = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Initialize confusion matrix
    matrix = [[0] * n_labels for _ in range(n_labels)]

    # Fill confusion matrix
    for pred, true in zip(predictions, ground_truth):
        if pred in label_to_idx and true in label_to_idx:
            matrix[label_to_idx[true]][label_to_idx[pred]] += 1

    # Compute per-class metrics
    per_class_metrics = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n_labels)) - tp
        fn = sum(matrix[i][j] for j in range(n_labels)) - tp
        tn = sum(sum(row) for row in matrix) - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[str(label)] = {
            'true_positive': tp,
            'false_positive': fp,
            'false_negative': fn,
            'true_negative': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    return {
        'matrix': matrix,
        'labels': labels,
        'per_class_metrics': per_class_metrics
    }


# Additional utility metrics

def compute_mean_absolute_error(
    predictions: List[float],
    ground_truth: List[float]
) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: Predicted values
        ground_truth: True values

    Returns:
        MAE score
    """
    if not predictions or not ground_truth:
        return 0.0

    return sum(abs(p - g) for p, g in zip(predictions, ground_truth)) / len(predictions)


def compute_mean_squared_error(
    predictions: List[float],
    ground_truth: List[float]
) -> float:
    """
    Compute Mean Squared Error.

    Args:
        predictions: Predicted values
        ground_truth: True values

    Returns:
        MSE score
    """
    if not predictions or not ground_truth:
        return 0.0

    return sum((p - g) ** 2 for p, g in zip(predictions, ground_truth)) / len(predictions)


def compute_root_mean_squared_error(
    predictions: List[float],
    ground_truth: List[float]
) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: Predicted values
        ground_truth: True values

    Returns:
        RMSE score
    """
    return math.sqrt(compute_mean_squared_error(predictions, ground_truth))


def compute_pearson_correlation(
    x: List[float],
    y: List[float]
) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Correlation coefficient (-1 to 1)
    """
    if not x or not y or len(x) != len(y):
        return 0.0

    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denominator_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denominator_x == 0 or denominator_y == 0:
        return 0.0

    return numerator / (denominator_x * denominator_y)
