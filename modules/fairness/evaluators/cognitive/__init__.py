# OmniTrust Fairness Module - Cognitive Bias Evaluators
# Copyright (c) 2025 OmniTrust Team

"""
Cognitive bias evaluators for LLM fairness assessment.

Available evaluators:
- OrderBiasEvaluator: Detects position preference bias
- CompassionBiasEvaluator: Detects compassion fade effect
- BandwagonBiasEvaluator: Detects social proof influence
- DistractionBiasEvaluator: Detects irrelevant info influence
- SelectiveBiasEvaluator: Detects expectation conformity
- FrequencyBiasEvaluator: Detects repetition/exposure bias
"""

from .base import CognitiveBiasEvaluator, CognitiveBiasResult
from .order import OrderBiasEvaluator
from .compassion import CompassionBiasEvaluator
from .bandwagon import BandwagonBiasEvaluator
from .distraction import DistractionBiasEvaluator
from .selective import SelectiveBiasEvaluator
from .frequency import FrequencyBiasEvaluator

__all__ = [
    # Base class
    "CognitiveBiasEvaluator",
    "CognitiveBiasResult",
    # Evaluators
    "OrderBiasEvaluator",
    "CompassionBiasEvaluator",
    "BandwagonBiasEvaluator",
    "DistractionBiasEvaluator",
    "SelectiveBiasEvaluator",
    "FrequencyBiasEvaluator",
]

# Mapping of bias names to evaluator classes
COGNITIVE_EVALUATORS = {
    "order": OrderBiasEvaluator,
    "compassion": CompassionBiasEvaluator,
    "bandwagon": BandwagonBiasEvaluator,
    "distraction": DistractionBiasEvaluator,
    "selective": SelectiveBiasEvaluator,
    "frequency": FrequencyBiasEvaluator,
}


def get_cognitive_evaluator(name: str, **kwargs):
    """
    Get a cognitive bias evaluator by name.

    Args:
        name: Evaluator name (order, compassion, bandwagon, distraction, selective, frequency)
        **kwargs: Additional arguments to pass to evaluator

    Returns:
        Evaluator instance

    Raises:
        ValueError: If evaluator name not found
    """
    name_lower = name.lower()
    if name_lower not in COGNITIVE_EVALUATORS:
        available = ", ".join(COGNITIVE_EVALUATORS.keys())
        raise ValueError(f"Unknown evaluator '{name}'. Available: {available}")
    return COGNITIVE_EVALUATORS[name_lower](**kwargs)
