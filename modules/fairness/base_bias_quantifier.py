"""
Base Bias Quantifier
====================

Provides the abstract base class for bias quantifiers that perform
statistical analysis on bias evaluation results.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging

from .bias_types import BiasType, BiasEvaluationResult
from ..base_quantifier import BaseQuantifier, QuantificationResult


@dataclass
class BiasQuantificationResult:
    """
    Result container for bias quantification.

    Attributes:
        quantifier_id: Unique identifier for the quantifier
        evaluation_result: Source bias evaluation result
        quantified_metrics: Aggregated and normalized metrics
        statistical_tests: Statistical significance tests
        confidence_intervals: Confidence intervals for metrics
        effect_sizes: Effect size measurements
        group_comparisons: Cross-group comparisons
        bias_summary: Summary of bias findings
        timestamp: ISO format timestamp
        metadata: Additional metadata
    """
    quantifier_id: str
    evaluation_result: BiasEvaluationResult
    quantified_metrics: Dict[str, float]
    statistical_tests: Dict[str, Any] = None
    confidence_intervals: Dict[str, Tuple[float, float]] = None
    effect_sizes: Dict[str, float] = None
    group_comparisons: Dict[str, Any] = None
    bias_summary: Dict[str, Any] = None
    timestamp: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.statistical_tests is None:
            self.statistical_tests = {}
        if self.confidence_intervals is None:
            self.confidence_intervals = {}
        if self.effect_sizes is None:
            self.effect_sizes = {}
        if self.group_comparisons is None:
            self.group_comparisons = {}
        if self.bias_summary is None:
            self.bias_summary = {}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        eval_result = self.evaluation_result
        if hasattr(eval_result, 'to_dict'):
            eval_result = eval_result.to_dict()

        return {
            "quantifier_id": self.quantifier_id,
            "evaluation_result": eval_result,
            "quantified_metrics": self.quantified_metrics,
            "statistical_tests": self.statistical_tests,
            "confidence_intervals": self.confidence_intervals,
            "effect_sizes": self.effect_sizes,
            "group_comparisons": self.group_comparisons,
            "bias_summary": self.bias_summary,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    def get_bias_level(self) -> str:
        """
        Get qualitative bias level based on metrics.

        Returns:
            One of: "none", "low", "moderate", "high", "severe"
        """
        score = self.quantified_metrics.get('overall_bias', 0.0)

        if score < 0.1:
            return "none"
        elif score < 0.3:
            return "low"
        elif score < 0.5:
            return "moderate"
        elif score < 0.7:
            return "high"
        else:
            return "severe"


class BiasQuantifierInterface(ABC):
    """
    Abstract interface for bias quantifiers.
    """

    @abstractmethod
    def quantify(self, evaluation_result: BiasEvaluationResult) -> BiasQuantificationResult:
        """
        Quantify bias evaluation results with statistical analysis.

        Args:
            evaluation_result: BiasEvaluationResult to quantify

        Returns:
            BiasQuantificationResult with statistical analysis
        """
        pass

    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """
        Get list of metrics this quantifier can compute.

        Returns:
            List of metric names
        """
        pass


class BaseBiasQuantifier(BaseQuantifier, BiasQuantifierInterface):
    """
    Base implementation for bias quantifiers.

    Provides common functionality for:
    - Statistical significance testing
    - Effect size calculation
    - Confidence interval estimation
    - Bias score aggregation

    Subclasses should override:
    - _aggregate_metrics(): Aggregation logic
    - _run_statistical_tests(): Statistical tests
    - _compute_effect_sizes(): Effect size computation
    - get_supported_metrics(): List of supported metrics

    Example:
        class PairwiseBiasQuantifier(BaseBiasQuantifier):
            def __init__(self, **kwargs):
                super().__init__(name="pairwise_quantifier", **kwargs)

            def _aggregate_metrics(self, raw_results, evaluation_result):
                return {"overall_bias": 0.4}
    """

    def __init__(
        self,
        name: str = "bias_quantifier",
        version: str = "1.0.0",
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the bias quantifier.

        Args:
            name: Quantifier name
            version: Version string
            description: Brief description
            config: Configuration dictionary
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            version=version,
            description=description,
            config=config,
            **kwargs
        )

        # Default configuration for bias quantification
        self._config.setdefault('significance_level', 0.05)
        self._config.setdefault('bootstrap_samples', 1000)
        self._config.setdefault('compute_effect_sizes', True)

    def quantify(self, evaluation_result: BiasEvaluationResult) -> BiasQuantificationResult:
        """
        Quantify bias evaluation results with statistical analysis.

        Args:
            evaluation_result: BiasEvaluationResult to quantify

        Returns:
            BiasQuantificationResult with statistical analysis
        """
        self._logger.info(f"Starting bias quantification: {self._name}")

        # Extract raw results
        raw_results = evaluation_result.raw_results

        # Aggregate metrics
        quantified_metrics = self._aggregate_metrics(raw_results, evaluation_result)

        # Run statistical tests
        statistical_tests = self._run_statistical_tests(raw_results, quantified_metrics)

        # Compute confidence intervals
        confidence_intervals = self._compute_confidence_intervals(raw_results, quantified_metrics)

        # Compute effect sizes
        effect_sizes = {}
        if self._config.get('compute_effect_sizes', True):
            effect_sizes = self._compute_effect_sizes(raw_results, quantified_metrics)

        # Generate group comparisons
        group_comparisons = self._generate_group_comparisons(
            raw_results,
            evaluation_result.group_metrics
        )

        # Generate bias summary
        bias_summary = self._generate_bias_summary(
            quantified_metrics,
            statistical_tests,
            effect_sizes
        )

        # Create result
        result = BiasQuantificationResult(
            quantifier_id=self._id,
            evaluation_result=evaluation_result,
            quantified_metrics=quantified_metrics,
            statistical_tests=statistical_tests,
            confidence_intervals=confidence_intervals,
            effect_sizes=effect_sizes,
            group_comparisons=group_comparisons,
            bias_summary=bias_summary,
            metadata={
                "quantifier_name": self._name,
                "quantifier_version": self._version,
                "bias_type": evaluation_result.bias_type.value,
                "config": self._config
            }
        )

        self._logger.info(f"Bias quantification completed. Level: {result.get_bias_level()}")
        return result

    @abstractmethod
    def _aggregate_metrics(
        self,
        raw_results: List[Dict[str, Any]],
        evaluation_result: BiasEvaluationResult
    ) -> Dict[str, float]:
        """
        Aggregate metrics from raw results. Must be implemented by subclasses.

        Args:
            raw_results: List of raw evaluation results
            evaluation_result: Original evaluation result

        Returns:
            Dictionary of aggregated metrics
        """
        pass

    def _compute_effect_sizes(
        self,
        raw_results: List[Dict[str, Any]],
        metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute effect sizes for bias measurements.

        Override in subclasses for custom effect size calculations.

        Args:
            raw_results: List of raw evaluation results
            metrics: Aggregated metrics

        Returns:
            Dictionary of effect sizes
        """
        return {}

    def _generate_group_comparisons(
        self,
        raw_results: List[Dict[str, Any]],
        group_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Generate cross-group comparisons.

        Args:
            raw_results: List of raw evaluation results
            group_metrics: Metrics by group

        Returns:
            Dictionary of group comparison results
        """
        if not group_metrics:
            return {}

        comparisons = {}
        groups = list(group_metrics.keys())

        # Pairwise comparisons
        for i, group_a in enumerate(groups):
            for group_b in groups[i+1:]:
                key = f"{group_a}_vs_{group_b}"
                metrics_a = group_metrics[group_a]
                metrics_b = group_metrics[group_b]

                # Compute differences for each metric
                differences = {}
                for metric in set(metrics_a.keys()) & set(metrics_b.keys()):
                    diff = metrics_a[metric] - metrics_b[metric]
                    differences[metric] = {
                        "difference": diff,
                        "group_a_value": metrics_a[metric],
                        "group_b_value": metrics_b[metric]
                    }

                comparisons[key] = differences

        return comparisons

    def _generate_bias_summary(
        self,
        metrics: Dict[str, float],
        statistical_tests: Dict[str, Any],
        effect_sizes: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate a summary of bias findings.

        Args:
            metrics: Quantified metrics
            statistical_tests: Statistical test results
            effect_sizes: Effect size measurements

        Returns:
            Summary dictionary
        """
        # Get overall bias score
        overall_bias = metrics.get('overall_bias', 0.0)

        # Determine bias level
        if overall_bias < 0.1:
            level = "none"
            interpretation = "No significant bias detected"
        elif overall_bias < 0.3:
            level = "low"
            interpretation = "Low level of bias detected"
        elif overall_bias < 0.5:
            level = "moderate"
            interpretation = "Moderate level of bias detected"
        elif overall_bias < 0.7:
            level = "high"
            interpretation = "High level of bias detected"
        else:
            level = "severe"
            interpretation = "Severe bias detected"

        # Check statistical significance
        significant = any(
            test.get('p_value', 1.0) < self._config.get('significance_level', 0.05)
            for test in statistical_tests.values()
            if isinstance(test, dict)
        )

        return {
            "overall_bias_score": overall_bias,
            "bias_level": level,
            "interpretation": interpretation,
            "statistically_significant": significant,
            "primary_metrics": {k: v for k, v in metrics.items() if v > 0.1},
            "recommendations": self._generate_recommendations(level, metrics)
        }

    def _generate_recommendations(
        self,
        bias_level: str,
        metrics: Dict[str, float]
    ) -> List[str]:
        """
        Generate recommendations based on bias findings.

        Args:
            bias_level: Detected bias level
            metrics: Quantified metrics

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if bias_level == "none":
            recommendations.append("Model shows no significant bias for the tested dimensions")
        elif bias_level in ["low", "moderate"]:
            recommendations.append("Consider fine-tuning with more balanced data")
            recommendations.append("Monitor model outputs in production")
        elif bias_level in ["high", "severe"]:
            recommendations.append("Immediate intervention recommended")
            recommendations.append("Review training data for systematic biases")
            recommendations.append("Consider debiasing techniques before deployment")

        return recommendations

    # Statistical utility methods

    @staticmethod
    def cohens_d(group_a: List[float], group_b: List[float]) -> float:
        """
        Compute Cohen's d effect size.

        Args:
            group_a: Values for group A
            group_b: Values for group B

        Returns:
            Cohen's d value
        """
        if not group_a or not group_b:
            return 0.0

        n1, n2 = len(group_a), len(group_b)
        mean1, mean2 = sum(group_a) / n1, sum(group_b) / n2

        var1 = sum((x - mean1) ** 2 for x in group_a) / (n1 - 1) if n1 > 1 else 0
        var2 = sum((x - mean2) ** 2 for x in group_b) / (n2 - 1) if n2 > 1 else 0

        # Pooled standard deviation
        pooled_std = ((var1 * (n1 - 1) + var2 * (n2 - 1)) / (n1 + n2 - 2)) ** 0.5

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        """
        Interpret Cohen's d effect size.

        Args:
            d: Cohen's d value

        Returns:
            Interpretation string
        """
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
