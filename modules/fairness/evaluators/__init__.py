# OmniTrust Fairness Module - Evaluators
# Copyright (c) 2025 OmniTrust Team

"""
Bias evaluators for LLM fairness assessment.

This package contains evaluators for:
- Cognitive biases: order, compassion, bandwagon, distraction, selective, frequency
- Social biases: age, gender, race, religion, nationality, political
"""

from .cognitive import (
    CognitiveBiasEvaluator,
    OrderBiasEvaluator,
    CompassionBiasEvaluator,
    BandwagonBiasEvaluator,
    DistractionBiasEvaluator,
    SelectiveBiasEvaluator,
    FrequencyBiasEvaluator,
)

__all__ = [
    # Base class
    "CognitiveBiasEvaluator",
    # Cognitive bias evaluators
    "OrderBiasEvaluator",
    "CompassionBiasEvaluator",
    "BandwagonBiasEvaluator",
    "DistractionBiasEvaluator",
    "SelectiveBiasEvaluator",
    "FrequencyBiasEvaluator",
]
