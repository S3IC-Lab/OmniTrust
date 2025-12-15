"""
Base Bias Evaluator
===================

Provides the abstract base class for all bias evaluators in the Fairness module.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import uuid
import logging

from .bias_types import (
    BiasType,
    BiasCategory,
    BiasDataPoint,
    BiasEvaluationResult,
    get_bias_category
)
from ..base_evaluator import BaseEvaluator, EvaluationType


class BiasEvaluatorInterface(ABC):
    """
    Abstract interface for bias evaluators.

    All bias evaluator implementations must inherit from this class
    and implement the required abstract methods.
    """

    @abstractmethod
    def evaluate(self, model: Any, dataset: Any, **kwargs) -> BiasEvaluationResult:
        """
        Evaluate model for bias.

        Args:
            model: The model to evaluate
            dataset: The dataset to use
            **kwargs: Additional parameters

        Returns:
            BiasEvaluationResult with metrics and raw results
        """
        pass

    @abstractmethod
    def get_bias_type(self) -> BiasType:
        """
        Get the type of bias this evaluator measures.

        Returns:
            BiasType enum value
        """
        pass

    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """
        Get list of metrics this evaluator computes.

        Returns:
            List of metric names
        """
        pass


class BaseBiasEvaluator(BaseEvaluator, BiasEvaluatorInterface):
    """
    Base implementation for bias evaluators.

    Provides common functionality for:
    - Pairwise comparison evaluation
    - Group-based analysis
    - Demographic disparity calculation
    - Bias score computation

    Subclasses should override:
    - _do_evaluate(): Core evaluation logic
    - _compute_metrics(): Metric computation
    - get_supported_metrics(): List of supported metrics
    - get_bias_type(): The bias type being evaluated

    Example:
        class GenderBiasEvaluator(BaseBiasEvaluator):
            def __init__(self, **kwargs):
                super().__init__(
                    name="gender_bias_evaluator",
                    bias_type=BiasType.GENDER,
                    **kwargs
                )

            def _do_evaluate(self, model, dataset, **kwargs):
                # Implementation
                return raw_results

            def _compute_metrics(self, raw_results):
                return {"gender_bias_score": 0.3}
    """

    def __init__(
        self,
        name: str,
        bias_type: BiasType,
        version: str = "1.0.0",
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the bias evaluator.

        Args:
            name: Evaluator name
            bias_type: Type of bias to evaluate
            version: Version string
            description: Brief description
            config: Configuration dictionary
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            evaluation_type=EvaluationType.FAIRNESS,
            version=version,
            description=description,
            config=config,
            **kwargs
        )

        self._bias_type = bias_type
        self._bias_category = get_bias_category(bias_type)

        # Default configuration for bias evaluation
        self._config.setdefault('pairwise_mode', True)
        self._config.setdefault('compute_group_metrics', True)
        self._config.setdefault('bias_threshold', 0.5)

    @property
    def bias_type(self) -> BiasType:
        """Get the bias type."""
        return self._bias_type

    @property
    def bias_category(self) -> BiasCategory:
        """Get the bias category."""
        return self._bias_category

    def get_bias_type(self) -> BiasType:
        """Get the type of bias this evaluator measures."""
        return self._bias_type

    def evaluate(self, model: Any, dataset: Any, **kwargs) -> BiasEvaluationResult:
        """
        Evaluate model for bias.

        This method orchestrates the bias evaluation:
        1. Validates inputs
        2. Performs evaluation via _do_evaluate()
        3. Computes metrics
        4. Calculates group-level metrics
        5. Performs pairwise comparisons
        6. Returns BiasEvaluationResult

        Args:
            model: The model to evaluate
            dataset: The dataset to use
            **kwargs: Additional parameters

        Returns:
            BiasEvaluationResult with metrics and analysis
        """
        self._logger.info(f"Starting bias evaluation: {self._name} ({self._bias_type.value})")

        # Validate inputs
        self._validate_inputs(model, dataset)

        # Perform evaluation
        raw_results = self._do_evaluate(model, dataset, **kwargs)

        # Compute metrics
        metrics = self._compute_metrics(raw_results)

        # Compute group metrics if enabled
        group_metrics = {}
        if self._config.get('compute_group_metrics', True):
            group_metrics = self._compute_group_metrics(raw_results)

        # Perform pairwise comparisons if enabled
        pairwise_comparisons = {}
        if self._config.get('pairwise_mode', True):
            pairwise_comparisons = self._compute_pairwise_comparisons(raw_results)

        # Create result
        result = BiasEvaluationResult(
            evaluator_id=self._id,
            bias_type=self._bias_type,
            metrics=metrics,
            raw_results=raw_results,
            group_metrics=group_metrics,
            pairwise_comparisons=pairwise_comparisons,
            metadata={
                "evaluator_name": self._name,
                "evaluator_version": self._version,
                "bias_category": self._bias_category.value,
                "config": self._config,
                "model_info": self._extract_model_info(model),
                "dataset_info": self._extract_dataset_info(dataset),
                **kwargs
            }
        )

        self._logger.info(f"Bias evaluation completed. Overall bias: {result.get_overall_bias_score():.4f}")
        return result

    @abstractmethod
    def _do_evaluate(self, model: Any, dataset: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        Core evaluation logic. Must be implemented by subclasses.

        Args:
            model: The model to evaluate
            dataset: The dataset to use
            **kwargs: Additional parameters

        Returns:
            List of raw evaluation results
        """
        pass

    @abstractmethod
    def _compute_metrics(self, raw_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute metrics from raw results. Must be implemented by subclasses.

        Args:
            raw_results: List of raw evaluation results

        Returns:
            Dictionary of computed metrics
        """
        pass

    def _compute_group_metrics(
        self,
        raw_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each demographic group.

        Override in subclasses for custom group analysis.

        Args:
            raw_results: List of raw evaluation results

        Returns:
            Dictionary mapping group names to their metrics
        """
        return {}

    def _compute_pairwise_comparisons(
        self,
        raw_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute pairwise comparisons between groups/conditions.

        Override in subclasses for custom comparison logic.

        Args:
            raw_results: List of raw evaluation results

        Returns:
            Dictionary of pairwise comparison results
        """
        return {}

    def get_evaluator_info(self) -> Dict[str, Any]:
        """Get information about this evaluator."""
        base_info = super().get_evaluator_info()
        base_info.update({
            "bias_type": self._bias_type.value,
            "bias_category": self._bias_category.value
        })
        return base_info

    # Utility methods for bias evaluation

    def compute_disparity(
        self,
        group_a_scores: List[float],
        group_b_scores: List[float]
    ) -> float:
        """
        Compute disparity between two groups.

        Args:
            group_a_scores: Scores for group A
            group_b_scores: Scores for group B

        Returns:
            Disparity score (0-1)
        """
        if not group_a_scores or not group_b_scores:
            return 0.0

        mean_a = sum(group_a_scores) / len(group_a_scores)
        mean_b = sum(group_b_scores) / len(group_b_scores)

        # Compute normalized disparity
        max_val = max(mean_a, mean_b) if max(mean_a, mean_b) > 0 else 1.0
        disparity = abs(mean_a - mean_b) / max_val

        return disparity

    def compute_bias_score(
        self,
        positive_count: int,
        negative_count: int,
        neutral_count: int = 0
    ) -> float:
        """
        Compute bias score from sentiment/association counts.

        Args:
            positive_count: Number of positive associations
            negative_count: Number of negative associations
            neutral_count: Number of neutral associations

        Returns:
            Bias score (-1 to 1, where 0 is no bias)
        """
        total = positive_count + negative_count + neutral_count
        if total == 0:
            return 0.0

        # Compute bias as imbalance
        bias = (positive_count - negative_count) / total
        return bias

    def compute_selection_rate(
        self,
        selections: List[int],
        total_options: int
    ) -> Dict[int, float]:
        """
        Compute selection rate for each option.

        Args:
            selections: List of selected option indices
            total_options: Total number of options

        Returns:
            Dictionary mapping option index to selection rate
        """
        if not selections:
            return {}

        counts = {i: 0 for i in range(total_options)}
        for s in selections:
            if s in counts:
                counts[s] += 1

        total = len(selections)
        rates = {k: v / total for k, v in counts.items()}
        return rates

    def is_biased(
        self,
        result: BiasEvaluationResult,
        threshold: Optional[float] = None
    ) -> bool:
        """
        Determine if the evaluation result indicates bias.

        Args:
            result: BiasEvaluationResult to check
            threshold: Optional custom threshold

        Returns:
            True if biased, False otherwise
        """
        if threshold is None:
            threshold = self._config.get('bias_threshold', 0.5)
        return result.is_biased(threshold)
