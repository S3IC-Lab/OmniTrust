"""
Base Evaluator Module
=====================

Provides the abstract base class and interfaces for all evaluation modules
in OmniTrust, including Safety, Privacy, Detectability, Hallucination,
Fairness, and Fidelity evaluators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class EvaluationType(Enum):
    """Enumeration of evaluation types supported by OmniTrust."""
    SAFETY = "safety"
    PRIVACY = "privacy"
    DETECTABILITY = "detectability"
    HALLUCINATION = "hallucination"
    FAIRNESS = "fairness"
    FIDELITY = "fidelity"


@dataclass
class EvaluationResult:
    """
    Standard evaluation result container.

    Attributes:
        evaluator_id: Unique identifier for the evaluator
        evaluation_type: Type of evaluation performed
        metrics: Dictionary of computed metrics
        raw_results: List of individual evaluation results
        timestamp: ISO format timestamp of evaluation
        metadata: Additional metadata about the evaluation
        model_info: Information about the evaluated model
        dataset_info: Information about the dataset used
    """
    evaluator_id: str
    evaluation_type: EvaluationType
    metrics: Dict[str, float]
    raw_results: List[Dict[str, Any]]
    timestamp: str = None
    metadata: Dict[str, Any] = None
    model_info: Dict[str, Any] = None
    dataset_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}
        if self.model_info is None:
            self.model_info = {}
        if self.dataset_info is None:
            self.dataset_info = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "evaluator_id": self.evaluator_id,
            "evaluation_type": self.evaluation_type.value,
            "metrics": self.metrics,
            "raw_results": self.raw_results,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "model_info": self.model_info,
            "dataset_info": self.dataset_info
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create from dictionary representation."""
        data = data.copy()
        data['evaluation_type'] = EvaluationType(data['evaluation_type'])
        return cls(**data)


class EvaluatorInterface(ABC):
    """
    Abstract interface for all evaluators.

    All concrete evaluator implementations must inherit from this class
    and implement the required abstract methods.
    """

    @abstractmethod
    def evaluate(self, model: Any, dataset: Any, **kwargs) -> EvaluationResult:
        """
        Perform evaluation on the model using the provided dataset.

        Args:
            model: The model to evaluate (ModelInterface or compatible)
            dataset: The dataset to use (DatasetInterface or compatible)
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult containing metrics and raw results
        """
        pass

    @abstractmethod
    def get_evaluator_info(self) -> Dict[str, Any]:
        """
        Get information about this evaluator.

        Returns:
            Dictionary containing evaluator metadata:
            - name: Evaluator name
            - version: Version string
            - description: Brief description
            - evaluation_type: Type of evaluation
            - supported_metrics: List of metrics computed
        """
        pass

    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """
        Get list of metrics this evaluator can compute.

        Returns:
            List of metric names
        """
        pass

    @abstractmethod
    def get_evaluation_type(self) -> EvaluationType:
        """
        Get the type of evaluation this evaluator performs.

        Returns:
            EvaluationType enum value
        """
        pass


class BaseEvaluator(EvaluatorInterface):
    """
    Base implementation of EvaluatorInterface with common functionality.

    Provides:
    - Unique evaluator ID generation
    - Configuration management
    - Logging setup
    - Common utility methods

    Subclasses should override:
    - _do_evaluate(): Core evaluation logic
    - _compute_metrics(): Metric computation
    - get_supported_metrics(): List of supported metrics
    - get_evaluation_type(): Evaluation type

    Example:
        class MyFairnessEvaluator(BaseEvaluator):
            def __init__(self, **kwargs):
                super().__init__(
                    name="my_fairness_evaluator",
                    evaluation_type=EvaluationType.FAIRNESS,
                    **kwargs
                )

            def _do_evaluate(self, model, dataset, **kwargs):
                # Implementation
                return raw_results

            def _compute_metrics(self, raw_results):
                return {"bias_score": 0.5}
    """

    def __init__(
        self,
        name: str,
        evaluation_type: EvaluationType,
        version: str = "1.0.0",
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the base evaluator.

        Args:
            name: Evaluator name
            evaluation_type: Type of evaluation
            version: Version string
            description: Brief description
            config: Configuration dictionary
            **kwargs: Additional parameters
        """
        self._id = str(uuid.uuid4())[:8]
        self._name = name
        self._evaluation_type = evaluation_type
        self._version = version
        self._description = description
        self._config = config or {}
        self._kwargs = kwargs

        # Initialize logging
        import logging
        self._logger = logging.getLogger(f"{self.__class__.__name__}[{self._id}]")

    @property
    def evaluator_id(self) -> str:
        """Get unique evaluator instance ID."""
        return self._id

    @property
    def name(self) -> str:
        """Get evaluator name."""
        return self._name

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Update evaluator configuration.

        Args:
            config: Configuration dictionary to merge
        """
        self._config.update(config)
        self._logger.debug(f"Configuration updated: {config}")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def evaluate(self, model: Any, dataset: Any, **kwargs) -> EvaluationResult:
        """
        Perform evaluation on the model using the provided dataset.

        This method orchestrates the evaluation process:
        1. Validates inputs
        2. Calls _do_evaluate() for core logic
        3. Computes metrics via _compute_metrics()
        4. Packages results into EvaluationResult

        Args:
            model: The model to evaluate
            dataset: The dataset to use
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult containing metrics and raw results
        """
        self._logger.info(f"Starting evaluation: {self._name}")

        # Validate inputs
        self._validate_inputs(model, dataset)

        # Perform evaluation
        raw_results = self._do_evaluate(model, dataset, **kwargs)

        # Compute metrics
        metrics = self._compute_metrics(raw_results)

        # Get model and dataset info
        model_info = self._extract_model_info(model)
        dataset_info = self._extract_dataset_info(dataset)

        # Create result
        result = EvaluationResult(
            evaluator_id=self._id,
            evaluation_type=self._evaluation_type,
            metrics=metrics,
            raw_results=raw_results,
            model_info=model_info,
            dataset_info=dataset_info,
            metadata={
                "evaluator_name": self._name,
                "evaluator_version": self._version,
                "config": self._config,
                **kwargs
            }
        )

        self._logger.info(f"Evaluation completed. Metrics: {metrics}")
        return result

    def _validate_inputs(self, model: Any, dataset: Any) -> None:
        """
        Validate evaluation inputs.

        Args:
            model: The model to validate
            dataset: The dataset to validate

        Raises:
            ValueError: If inputs are invalid
        """
        if model is None:
            raise ValueError("Model cannot be None")
        if dataset is None:
            raise ValueError("Dataset cannot be None")

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

    def _extract_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Extract model information.

        Args:
            model: The model object

        Returns:
            Dictionary of model information
        """
        if hasattr(model, 'get_model_info'):
            return model.get_model_info()
        return {"type": type(model).__name__}

    def _extract_dataset_info(self, dataset: Any) -> Dict[str, Any]:
        """
        Extract dataset information.

        Args:
            dataset: The dataset object

        Returns:
            Dictionary of dataset information
        """
        if hasattr(dataset, 'get_metadata'):
            return dataset.get_metadata()
        return {"type": type(dataset).__name__}

    def get_evaluator_info(self) -> Dict[str, Any]:
        """Get information about this evaluator."""
        return {
            "id": self._id,
            "name": self._name,
            "version": self._version,
            "description": self._description,
            "evaluation_type": self._evaluation_type.value,
            "supported_metrics": self.get_supported_metrics(),
            "config": self._config
        }

    def get_evaluation_type(self) -> EvaluationType:
        """Get the type of evaluation this evaluator performs."""
        return self._evaluation_type
