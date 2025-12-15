"""
OmniTrust Module Infrastructure
===============================

This module provides the foundational base classes and interfaces for all
evaluation modules in OmniTrust (Safety, Privacy, Detectability, Hallucination,
Fairness, Fidelity).

Core Components:
- BaseEvaluator: Abstract base class for all evaluators
- BaseQuantifier: Abstract base class for metric quantifiers
- BaseVisualizer: Abstract base class for result visualizers
- BasePipeline: Abstract base class for evaluation pipelines

Usage:
    from modules import BaseEvaluator, BaseQuantifier, EvaluationResult

    class MyEvaluator(BaseEvaluator):
        def evaluate(self, model, dataset):
            # Implementation
            pass
"""

from .base_evaluator import (
    BaseEvaluator,
    EvaluatorInterface,
    EvaluationResult,
    EvaluationType
)

from .base_quantifier import (
    BaseQuantifier,
    QuantifierInterface,
    QuantificationResult
)

from .base_visualizer import (
    BaseVisualizer,
    VisualizerInterface
)

from .base_pipeline import (
    BasePipeline,
    PipelineInterface,
    PipelineResult
)

from .registry import (
    ModuleRegistry,
    register_evaluator,
    register_quantifier,
    register_visualizer
)

__all__ = [
    # Evaluator
    'BaseEvaluator',
    'EvaluatorInterface',
    'EvaluationResult',
    'EvaluationType',
    # Quantifier
    'BaseQuantifier',
    'QuantifierInterface',
    'QuantificationResult',
    # Visualizer
    'BaseVisualizer',
    'VisualizerInterface',
    # Pipeline
    'BasePipeline',
    'PipelineInterface',
    'PipelineResult',
    # Registry
    'ModuleRegistry',
    'register_evaluator',
    'register_quantifier',
    'register_visualizer',
]
