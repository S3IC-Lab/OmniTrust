"""
OmniTrust Fairness Module
=========================

This module provides evaluation tools for assessing fairness and bias
in Large Language Models (LLMs) and Vision-Language Models (VLMs).

Bias Types Supported:
- Stereotyping: Evaluates stereotypical associations
- Demographic Bias: Measures bias across demographic groups (gender, race, age, etc.)
- Social Bias: Assesses broader social biases in model outputs
- Representation Harm: Evaluates harmful representations
- Selection Bias: Detects biased selection in model responses

Key Components:
- BiasEvaluator: Base class for bias evaluation
- BiasQuantifier: Statistical quantification of bias metrics
- BiasVisualizer: Visualization tools for bias analysis
- BiasDataset: Dataset utilities for bias evaluation

Usage:
    from modules.fairness import (
        BiasEvaluator,
        BiasQuantifier,
        BiasType,
        BiasDataPoint
    )

    evaluator = BiasEvaluator(bias_type=BiasType.DEMOGRAPHIC)
    result = evaluator.evaluate(model, dataset)
"""

from .bias_types import (
    BiasType,
    BiasCategory,
    DemographicGroup,
    BiasDataPoint,
    BiasEvaluationResult
)

from .base_bias_evaluator import (
    BaseBiasEvaluator,
    BiasEvaluatorInterface
)

from .base_bias_quantifier import (
    BaseBiasQuantifier,
    BiasQuantifierInterface,
    BiasQuantificationResult
)

from .base_bias_visualizer import (
    BaseBiasVisualizer,
    BiasVisualizerInterface
)

from .bias_pipeline import (
    BiasPipeline
)

__all__ = [
    # Types and Data Structures
    'BiasType',
    'BiasCategory',
    'DemographicGroup',
    'BiasDataPoint',
    'BiasEvaluationResult',
    # Evaluator
    'BaseBiasEvaluator',
    'BiasEvaluatorInterface',
    # Quantifier
    'BaseBiasQuantifier',
    'BiasQuantifierInterface',
    'BiasQuantificationResult',
    # Visualizer
    'BaseBiasVisualizer',
    'BiasVisualizerInterface',
    # Pipeline
    'BiasPipeline',
]
