"""
Bias Evaluation Pipeline
========================

Provides a complete pipeline for bias evaluation, combining
evaluator, quantifier, and visualizer components.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .bias_types import BiasType, BiasEvaluationResult
from .base_bias_evaluator import BaseBiasEvaluator
from .base_bias_quantifier import BaseBiasQuantifier, BiasQuantificationResult
from .base_bias_visualizer import BaseBiasVisualizer
from ..base_pipeline import BasePipeline, PipelineResult


class BiasPipeline(BasePipeline):
    """
    Complete pipeline for bias evaluation.

    Orchestrates the full bias evaluation workflow:
    1. Model evaluation with bias evaluator
    2. Statistical quantification of bias
    3. Visualization of results
    4. Report generation

    Example:
        pipeline = BiasPipeline()

        result = pipeline.run(
            model=my_model,
            dataset=bias_dataset,
            evaluator=gender_bias_evaluator,
            quantifier=pairwise_quantifier,
            visualizer=bias_visualizer,
            output_dir="results/gender_bias"
        )

        print(f"Bias level: {result.metadata.get('bias_level')}")
    """

    def __init__(
        self,
        name: str = "bias_pipeline",
        version: str = "1.0.0",
        description: str = "Complete bias evaluation pipeline",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the bias pipeline.

        Args:
            name: Pipeline name
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

        # Default configuration
        self._config.setdefault('generate_report', True)
        self._config.setdefault('save_raw_results', True)
        self._config.setdefault('bias_threshold', 0.5)

    def run(
        self,
        model: Any,
        dataset: Any,
        evaluator: BaseBiasEvaluator,
        quantifier: Optional[BaseBiasQuantifier] = None,
        visualizer: Optional[BaseBiasVisualizer] = None,
        output_dir: str = "results",
        **kwargs
    ) -> PipelineResult:
        """
        Run the complete bias evaluation pipeline.

        Args:
            model: Model to evaluate
            dataset: Bias evaluation dataset
            evaluator: Bias evaluator instance
            quantifier: Bias quantifier (optional)
            visualizer: Bias visualizer (optional)
            output_dir: Output directory for results
            **kwargs: Additional parameters

        Returns:
            PipelineResult containing evaluation outputs
        """
        self._logger.info(f"Starting bias pipeline: {evaluator.name}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(
            pipeline_id=self._current_run_id or "unknown",
            metadata={
                "pipeline_name": self._name,
                "bias_type": evaluator.bias_type.value if hasattr(evaluator, 'bias_type') else "unknown"
            }
        )

        try:
            # Step 1: Run bias evaluation
            self._logger.info("Running bias evaluation...")
            evaluation_result = evaluator.evaluate(model, dataset, **kwargs)
            result.evaluation_results = evaluation_result

            # Save evaluation results
            if self._config.get('save_raw_results', True):
                self._save_evaluation_result(evaluation_result, output_path)

            # Step 2: Run quantification if provided
            quantification_result = None
            if quantifier is not None:
                self._logger.info("Running bias quantification...")
                quantification_result = quantifier.quantify(evaluation_result)
                result.quantification_results = quantification_result

                # Save quantification results
                self._save_quantification_result(quantification_result, output_path)

                # Add bias level to metadata
                result.metadata['bias_level'] = quantification_result.get_bias_level()

            # Step 3: Run visualization if provided
            if visualizer is not None:
                self._logger.info("Generating visualizations...")
                vis_input = quantification_result or evaluation_result
                vis_path = visualizer.visualize(
                    vis_input,
                    str(output_path / "visualizations")
                )
                result.visualization_paths.append(vis_path)

            # Add summary to metadata
            result.metadata['evaluation_summary'] = self._create_summary(
                evaluation_result,
                quantification_result
            )

            self._logger.info("Bias pipeline completed successfully")

        except Exception as e:
            self._logger.error(f"Bias pipeline error: {e}")
            result.errors.append({
                "type": type(e).__name__,
                "message": str(e)
            })

            if not self._config.get('continue_on_error', False):
                raise

        return result

    def _save_evaluation_result(
        self,
        result: BiasEvaluationResult,
        output_dir: Path
    ) -> None:
        """
        Save evaluation result to file.

        Args:
            result: BiasEvaluationResult to save
            output_dir: Output directory
        """
        import json
        output_path = output_dir / "evaluation_result.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        self._logger.debug(f"Saved evaluation result to: {output_path}")

    def _save_quantification_result(
        self,
        result: BiasQuantificationResult,
        output_dir: Path
    ) -> None:
        """
        Save quantification result to file.

        Args:
            result: BiasQuantificationResult to save
            output_dir: Output directory
        """
        import json
        output_path = output_dir / "quantification_result.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        self._logger.debug(f"Saved quantification result to: {output_path}")

    def _create_summary(
        self,
        evaluation_result: BiasEvaluationResult,
        quantification_result: Optional[BiasQuantificationResult]
    ) -> Dict[str, Any]:
        """
        Create a summary of the pipeline execution.

        Args:
            evaluation_result: Evaluation result
            quantification_result: Quantification result (optional)

        Returns:
            Summary dictionary
        """
        summary = {
            "bias_type": evaluation_result.bias_type.value,
            "total_samples": len(evaluation_result.raw_results),
            "metrics": evaluation_result.metrics,
        }

        if quantification_result:
            summary.update({
                "bias_level": quantification_result.get_bias_level(),
                "quantified_metrics": quantification_result.quantified_metrics,
                "bias_summary": quantification_result.bias_summary
            })

        return summary

    def run_batch(
        self,
        model: Any,
        dataset: Any,
        evaluators: List[BaseBiasEvaluator],
        quantifier: Optional[BaseBiasQuantifier] = None,
        visualizer: Optional[BaseBiasVisualizer] = None,
        output_dir: str = "results",
        **kwargs
    ) -> Dict[str, PipelineResult]:
        """
        Run pipeline with multiple evaluators.

        Args:
            model: Model to evaluate
            dataset: Bias evaluation dataset
            evaluators: List of bias evaluators
            quantifier: Shared quantifier (optional)
            visualizer: Shared visualizer (optional)
            output_dir: Output directory
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping evaluator names to results
        """
        results = {}
        output_base = Path(output_dir)

        for evaluator in evaluators:
            eval_name = evaluator.name
            eval_output = output_base / eval_name

            self._logger.info(f"Running pipeline for: {eval_name}")

            try:
                result = self.run(
                    model=model,
                    dataset=dataset,
                    evaluator=evaluator,
                    quantifier=quantifier,
                    visualizer=visualizer,
                    output_dir=str(eval_output),
                    **kwargs
                )
                results[eval_name] = result
            except Exception as e:
                self._logger.error(f"Failed to run {eval_name}: {e}")
                results[eval_name] = PipelineResult(
                    pipeline_id="failed",
                    errors=[{"type": type(e).__name__, "message": str(e)}]
                )

        # Generate combined summary
        self._generate_batch_summary(results, output_base)

        return results

    def _generate_batch_summary(
        self,
        results: Dict[str, PipelineResult],
        output_dir: Path
    ) -> None:
        """
        Generate summary for batch evaluation.

        Args:
            results: Dictionary of pipeline results
            output_dir: Output directory
        """
        import json

        summary = {
            "total_evaluators": len(results),
            "successful": sum(1 for r in results.values() if r.success),
            "failed": sum(1 for r in results.values() if not r.success),
            "results": {}
        }

        for name, result in results.items():
            summary["results"][name] = {
                "success": result.success,
                "bias_level": result.metadata.get("bias_level", "unknown"),
                "errors": result.errors
            }

        output_path = output_dir / "batch_summary.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self._logger.info(f"Batch summary saved to: {output_path}")
