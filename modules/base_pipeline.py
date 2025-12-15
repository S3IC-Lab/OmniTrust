"""
Base Pipeline Module
====================

Provides the abstract base class and interfaces for evaluation pipelines
that orchestrate the complete evaluation workflow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import uuid
import json
import time


@dataclass
class PipelineResult:
    """
    Standard pipeline result container.

    Attributes:
        pipeline_id: Unique identifier for the pipeline run
        evaluation_results: Results from evaluation phase
        quantification_results: Results from quantification phase
        visualization_paths: Paths to generated visualizations
        execution_time: Total execution time in seconds
        timestamp: ISO format timestamp
        metadata: Additional metadata
        errors: List of errors encountered
    """
    pipeline_id: str
    evaluation_results: Any = None
    quantification_results: Any = None
    visualization_paths: List[str] = None
    execution_time: float = 0.0
    timestamp: str = None
    metadata: Dict[str, Any] = None
    errors: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.visualization_paths is None:
            self.visualization_paths = []
        if self.metadata is None:
            self.metadata = {}
        if self.errors is None:
            self.errors = []

    @property
    def success(self) -> bool:
        """Check if pipeline completed without errors."""
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        eval_result = self.evaluation_results
        if hasattr(eval_result, 'to_dict'):
            eval_result = eval_result.to_dict()

        quant_result = self.quantification_results
        if hasattr(quant_result, 'to_dict'):
            quant_result = quant_result.to_dict()

        return {
            "pipeline_id": self.pipeline_id,
            "evaluation_results": eval_result,
            "quantification_results": quant_result,
            "visualization_paths": self.visualization_paths,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "errors": self.errors,
            "success": self.success
        }


class PipelineInterface(ABC):
    """
    Abstract interface for all evaluation pipelines.

    Pipelines are responsible for:
    - Orchestrating evaluation workflow
    - Managing component lifecycle
    - Handling errors and recovery
    - Saving and loading results
    """

    @abstractmethod
    def run(
        self,
        model: Any,
        dataset: Any,
        evaluator: Any,
        quantifier: Optional[Any] = None,
        visualizer: Optional[Any] = None,
        output_dir: str = "results",
        **kwargs
    ) -> PipelineResult:
        """
        Run the complete evaluation pipeline.

        Args:
            model: Model to evaluate
            dataset: Dataset to use
            evaluator: Evaluator instance
            quantifier: Quantifier instance (optional)
            visualizer: Visualizer instance (optional)
            output_dir: Output directory for results
            **kwargs: Additional parameters

        Returns:
            PipelineResult containing all outputs
        """
        pass

    @abstractmethod
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about this pipeline.

        Returns:
            Dictionary containing pipeline metadata
        """
        pass


class BasePipeline(PipelineInterface):
    """
    Base implementation of PipelineInterface with common functionality.

    Provides:
    - Workflow orchestration
    - Error handling and recovery
    - Result persistence
    - Progress tracking

    Subclasses can override:
    - _pre_evaluate(): Pre-evaluation hooks
    - _post_evaluate(): Post-evaluation hooks
    - _on_error(): Error handling

    Example:
        class BiasEvaluationPipeline(BasePipeline):
            def __init__(self, **kwargs):
                super().__init__(name="bias_pipeline", **kwargs)

            def _pre_evaluate(self, model, dataset):
                # Custom preprocessing
                pass
    """

    def __init__(
        self,
        name: str = "base_pipeline",
        version: str = "1.0.0",
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the base pipeline.

        Args:
            name: Pipeline name
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
        self._config.setdefault('cache_enabled', True)
        self._config.setdefault('save_intermediate', True)
        self._config.setdefault('continue_on_error', False)

        # Pipeline state
        self._results_cache: Dict[str, Any] = {}
        self._current_run_id: Optional[str] = None

        # Initialize logging
        import logging
        self._logger = logging.getLogger(f"{self.__class__.__name__}[{self._id}]")

    @property
    def pipeline_id(self) -> str:
        """Get unique pipeline instance ID."""
        return self._id

    @property
    def name(self) -> str:
        """Get pipeline name."""
        return self._name

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Update pipeline configuration.

        Args:
            config: Configuration dictionary to merge
        """
        self._config.update(config)
        self._logger.debug(f"Configuration updated: {config}")

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()

    def run(
        self,
        model: Any,
        dataset: Any,
        evaluator: Any,
        quantifier: Optional[Any] = None,
        visualizer: Optional[Any] = None,
        output_dir: str = "results",
        **kwargs
    ) -> PipelineResult:
        """
        Run the complete evaluation pipeline.

        Workflow:
        1. Pre-evaluation hooks
        2. Model availability check
        3. Dataset loading
        4. Evaluation
        5. Quantification (if quantifier provided)
        6. Visualization (if visualizer provided)
        7. Result saving
        8. Post-evaluation hooks

        Args:
            model: Model to evaluate
            dataset: Dataset to use
            evaluator: Evaluator instance
            quantifier: Quantifier instance (optional)
            visualizer: Visualizer instance (optional)
            output_dir: Output directory for results
            **kwargs: Additional parameters

        Returns:
            PipelineResult containing all outputs
        """
        run_id = str(uuid.uuid4())[:8]
        self._current_run_id = run_id
        start_time = time.time()

        self._logger.info(f"Starting pipeline run: {run_id}")

        # Initialize result
        result = PipelineResult(
            pipeline_id=run_id,
            metadata={
                "pipeline_name": self._name,
                "pipeline_version": self._version,
                "config": self._config,
                **kwargs
            }
        )

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Pre-evaluation hooks
            self._pre_evaluate(model, dataset, evaluator)

            # Check model availability
            if hasattr(model, 'is_available') and not model.is_available():
                raise RuntimeError("Model is not available")

            # Load dataset
            self._logger.info("Loading dataset")
            if hasattr(dataset, 'load_data'):
                dataset.load_data()

            # Run evaluation
            self._logger.info("Running evaluation")
            evaluation_result = evaluator.evaluate(model, dataset)
            result.evaluation_results = evaluation_result

            # Save intermediate results
            if self._config.get('save_intermediate', True):
                self._save_intermediate(evaluation_result, output_path / "evaluation_result.json")

            # Run quantification
            if quantifier is not None:
                self._logger.info("Running quantification")
                quantification_result = quantifier.quantify(evaluation_result)
                result.quantification_results = quantification_result

                if self._config.get('save_intermediate', True):
                    self._save_intermediate(quantification_result, output_path / "quantification_result.json")

            # Run visualization
            if visualizer is not None:
                self._logger.info("Running visualization")
                vis_input = result.quantification_results or result.evaluation_results
                vis_path = visualizer.visualize(vis_input, str(output_path / "visualizations"))
                result.visualization_paths.append(vis_path)

            # Post-evaluation hooks
            self._post_evaluate(result)

        except Exception as e:
            self._logger.error(f"Pipeline error: {e}")
            result.errors.append({
                "type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })

            # Handle error
            self._on_error(e, result)

            if not self._config.get('continue_on_error', False):
                raise

        # Finalize result
        result.execution_time = time.time() - start_time

        # Save final result
        self._save_result(result, output_path / f"pipeline_result_{run_id}.json")

        # Update metadata with model and dataset info
        if hasattr(model, 'get_model_info'):
            result.metadata['model_info'] = model.get_model_info()
        if hasattr(dataset, 'get_metadata'):
            result.metadata['dataset_info'] = dataset.get_metadata()
        if hasattr(evaluator, 'get_evaluator_info'):
            result.metadata['evaluator_info'] = evaluator.get_evaluator_info()

        self._logger.info(f"Pipeline completed in {result.execution_time:.2f}s. Success: {result.success}")
        return result

    def _pre_evaluate(self, model: Any, dataset: Any, evaluator: Any) -> None:
        """
        Pre-evaluation hook. Override for custom preprocessing.

        Args:
            model: Model to evaluate
            dataset: Dataset to use
            evaluator: Evaluator instance
        """
        pass

    def _post_evaluate(self, result: PipelineResult) -> None:
        """
        Post-evaluation hook. Override for custom postprocessing.

        Args:
            result: Pipeline result
        """
        pass

    def _on_error(self, error: Exception, result: PipelineResult) -> None:
        """
        Error handling hook. Override for custom error handling.

        Args:
            error: The exception that occurred
            result: Pipeline result
        """
        pass

    def _save_intermediate(self, data: Any, path: Path) -> None:
        """
        Save intermediate results.

        Args:
            data: Data to save
            path: File path
        """
        try:
            if hasattr(data, 'to_dict'):
                data = data.to_dict()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self._logger.warning(f"Failed to save intermediate result: {e}")

    def _save_result(self, result: PipelineResult, path: Path) -> None:
        """
        Save final pipeline result.

        Args:
            result: Pipeline result
            path: File path
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            self._logger.info(f"Result saved to: {path}")
        except Exception as e:
            self._logger.error(f"Failed to save result: {e}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about this pipeline."""
        return {
            "id": self._id,
            "name": self._name,
            "version": self._version,
            "description": self._description,
            "config": self._config
        }

    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """
        Get cached result.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None
        """
        return self._results_cache.get(cache_key)

    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._results_cache.clear()
        self._logger.debug("Cache cleared")
