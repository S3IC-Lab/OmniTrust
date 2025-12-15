"""
OmniTrust Evaluation Module
===========================

This module provides the core evaluation infrastructure for all OmniTrust
trustworthiness assessments, including utility functions, metrics computation,
and evaluation helpers.

Key Components:
- Metrics: Standard metrics computation utilities
- Evaluation Utilities: Helper functions for evaluation workflows
- Result Processing: Tools for processing and aggregating results

Usage:
    from eval import compute_accuracy, compute_f1, EvaluationMetrics
    from eval.utils import batch_evaluate, aggregate_results
"""

from .metrics import (
    EvaluationMetrics,
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_auc_roc,
    compute_confusion_matrix
)

from .utils import (
    batch_evaluate,
    aggregate_results,
    format_results,
    save_results,
    load_results
)

__all__ = [
    # Metrics
    'EvaluationMetrics',
    'compute_accuracy',
    'compute_precision',
    'compute_recall',
    'compute_f1',
    'compute_auc_roc',
    'compute_confusion_matrix',
    # Utilities
    'batch_evaluate',
    'aggregate_results',
    'format_results',
    'save_results',
    'load_results',
]
