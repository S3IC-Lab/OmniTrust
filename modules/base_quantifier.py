"""
Base Quantifier Module
======================

Provides the abstract base class and interfaces for metric quantifiers
that analyze and summarize evaluation results with statistical rigor.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class QuantificationResult:
    """
    Standard quantification result container.

    Attributes:
        quantifier_id: Unique identifier for the quantifier
        evaluation_result: The source evaluation result
        quantified_metrics: Aggregated and normalized metrics
        statistical_tests: Results of statistical significance tests
        confidence_intervals: Confidence intervals for metrics
        comparisons: Cross-group or cross-model comparisons
        timestamp: ISO format timestamp
        metadata: Additional metadata
    """
    quantifier_id: str
    evaluation_result: Any  # EvaluationResult
    quantified_metrics: Dict[str, float]
    statistical_tests: Dict[str, Any] = None
    confidence_intervals: Dict[str, Tuple[float, float]] = None
    comparisons: Dict[str, Any] = None
    timestamp: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.statistical_tests is None:
            self.statistical_tests = {}
        if self.confidence_intervals is None:
            self.confidence_intervals = {}
        if self.comparisons is None:
            self.comparisons = {}
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
            "comparisons": self.comparisons,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class QuantifierInterface(ABC):
    """
    Abstract interface for all quantifiers.

    Quantifiers are responsible for:
    - Aggregating raw evaluation results
    - Computing statistical measures
    - Performing significance tests
    - Generating confidence intervals
    """

    @abstractmethod
    def quantify(self, evaluation_result: Any) -> QuantificationResult:
        """
        Quantify evaluation results with statistical analysis.

        Args:
            evaluation_result: EvaluationResult to quantify

        Returns:
            QuantificationResult with statistical analysis
        """
        pass

    @abstractmethod
    def get_quantifier_info(self) -> Dict[str, Any]:
        """
        Get information about this quantifier.

        Returns:
            Dictionary containing quantifier metadata
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


class BaseQuantifier(QuantifierInterface):
    """
    Base implementation of QuantifierInterface with common functionality.

    Provides:
    - Statistical test utilities
    - Confidence interval computation
    - Metric aggregation
    - Result formatting

    Subclasses should override:
    - _aggregate_metrics(): Aggregation logic
    - _run_statistical_tests(): Statistical analysis
    - _compute_confidence_intervals(): CI computation
    - get_supported_metrics(): List of supported metrics

    Example:
        class BiasQuantifier(BaseQuantifier):
            def __init__(self, **kwargs):
                super().__init__(name="bias_quantifier", **kwargs)

            def _aggregate_metrics(self, raw_results):
                return {"overall_bias": 0.5}
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the base quantifier.

        Args:
            name: Quantifier name
            version: Version string
            description: Brief description
            config: Configuration dictionary
            **kwargs: Additional parameters
        """
        self._id = str(uuid.uuid4())[:8]
        self._name = name
        self._version = version
        self._description = description
        self._config = config or {}
        self._kwargs = kwargs

        # Default configuration
        self._config.setdefault('confidence_level', 0.95)
        self._config.setdefault('bootstrap_samples', 1000)

        # Initialize logging
        import logging
        self._logger = logging.getLogger(f"{self.__class__.__name__}[{self._id}]")

    @property
    def quantifier_id(self) -> str:
        """Get unique quantifier instance ID."""
        return self._id

    @property
    def name(self) -> str:
        """Get quantifier name."""
        return self._name

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Update quantifier configuration.

        Args:
            config: Configuration dictionary to merge
        """
        self._config.update(config)
        self._logger.debug(f"Configuration updated: {config}")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def quantify(self, evaluation_result: Any) -> QuantificationResult:
        """
        Quantify evaluation results with statistical analysis.

        This method orchestrates the quantification process:
        1. Extracts raw results from evaluation
        2. Aggregates metrics
        3. Runs statistical tests
        4. Computes confidence intervals
        5. Packages results

        Args:
            evaluation_result: EvaluationResult to quantify

        Returns:
            QuantificationResult with statistical analysis
        """
        self._logger.info(f"Starting quantification: {self._name}")

        # Extract raw results
        raw_results = self._extract_raw_results(evaluation_result)

        # Aggregate metrics
        quantified_metrics = self._aggregate_metrics(raw_results, evaluation_result)

        # Run statistical tests
        statistical_tests = self._run_statistical_tests(raw_results, quantified_metrics)

        # Compute confidence intervals
        confidence_intervals = self._compute_confidence_intervals(raw_results, quantified_metrics)

        # Generate comparisons
        comparisons = self._generate_comparisons(raw_results, quantified_metrics)

        # Create result
        result = QuantificationResult(
            quantifier_id=self._id,
            evaluation_result=evaluation_result,
            quantified_metrics=quantified_metrics,
            statistical_tests=statistical_tests,
            confidence_intervals=confidence_intervals,
            comparisons=comparisons,
            metadata={
                "quantifier_name": self._name,
                "quantifier_version": self._version,
                "config": self._config
            }
        )

        self._logger.info(f"Quantification completed. Metrics: {quantified_metrics}")
        return result

    def _extract_raw_results(self, evaluation_result: Any) -> List[Dict[str, Any]]:
        """
        Extract raw results from evaluation result.

        Args:
            evaluation_result: EvaluationResult object

        Returns:
            List of raw result dictionaries
        """
        if hasattr(evaluation_result, 'raw_results'):
            return evaluation_result.raw_results
        elif isinstance(evaluation_result, dict):
            return evaluation_result.get('raw_results', [])
        return []

    @abstractmethod
    def _aggregate_metrics(
        self,
        raw_results: List[Dict[str, Any]],
        evaluation_result: Any
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

    def _run_statistical_tests(
        self,
        raw_results: List[Dict[str, Any]],
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Run statistical significance tests.

        Override in subclasses for specific tests (e.g., t-tests, chi-square).

        Args:
            raw_results: List of raw evaluation results
            metrics: Aggregated metrics

        Returns:
            Dictionary of test results
        """
        return {}

    def _compute_confidence_intervals(
        self,
        raw_results: List[Dict[str, Any]],
        metrics: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals for metrics.

        Override in subclasses for specific CI computation methods.

        Args:
            raw_results: List of raw evaluation results
            metrics: Aggregated metrics

        Returns:
            Dictionary mapping metric names to (lower, upper) tuples
        """
        return {}

    def _generate_comparisons(
        self,
        raw_results: List[Dict[str, Any]],
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate cross-group or cross-model comparisons.

        Override in subclasses for specific comparison logic.

        Args:
            raw_results: List of raw evaluation results
            metrics: Aggregated metrics

        Returns:
            Dictionary of comparison results
        """
        return {}

    def get_quantifier_info(self) -> Dict[str, Any]:
        """Get information about this quantifier."""
        return {
            "id": self._id,
            "name": self._name,
            "version": self._version,
            "description": self._description,
            "supported_metrics": self.get_supported_metrics(),
            "config": self._config
        }

    # Statistical utility methods

    @staticmethod
    def compute_mean(values: List[float]) -> float:
        """Compute arithmetic mean."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def compute_std(values: List[float], ddof: int = 1) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - ddof)
        return variance ** 0.5

    @staticmethod
    def compute_percentile(values: List[float], percentile: float) -> float:
        """Compute percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = (len(sorted_values) - 1) * percentile / 100
        lower = int(idx)
        upper = lower + 1
        if upper >= len(sorted_values):
            return sorted_values[-1]
        weight = idx - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
