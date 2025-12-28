# OmniTrust Fairness Module - Evaluators
# Copyright (c) 2025 OmniTrust Team

"""
Bias evaluators for LLM fairness assessment.

This package contains evaluators for:
- Cognitive biases (6): order, compassion, bandwagon, distraction, selective, frequency
- Social biases (6): age, gender, race, religion, nationality, political
"""

from .cognitive import (
    CognitiveBiasEvaluator,
    CognitiveBiasResult,
    OrderBiasEvaluator,
    CompassionBiasEvaluator,
    BandwagonBiasEvaluator,
    DistractionBiasEvaluator,
    SelectiveBiasEvaluator,
    FrequencyBiasEvaluator,
)

from .social import (
    SocialBiasEvaluator,
    SocialBiasResult,
    AgeBiasEvaluator,
    GenderBiasEvaluator,
    RaceBiasEvaluator,
    ReligionBiasEvaluator,
    NationalityBiasEvaluator,
    PoliticalBiasEvaluator,
)

__all__ = [
    # Cognitive bias base class and result
    "CognitiveBiasEvaluator",
    "CognitiveBiasResult",
    # Cognitive bias evaluators (6)
    "OrderBiasEvaluator",
    "CompassionBiasEvaluator",
    "BandwagonBiasEvaluator",
    "DistractionBiasEvaluator",
    "SelectiveBiasEvaluator",
    "FrequencyBiasEvaluator",
    # Social bias base class and result
    "SocialBiasEvaluator",
    "SocialBiasResult",
    # Social bias evaluators (6)
    "AgeBiasEvaluator",
    "GenderBiasEvaluator",
    "RaceBiasEvaluator",
    "ReligionBiasEvaluator",
    "NationalityBiasEvaluator",
    "PoliticalBiasEvaluator",
]
