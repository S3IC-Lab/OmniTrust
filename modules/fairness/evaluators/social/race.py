# OmniTrust Fairness Module - Race Bias Evaluator
# Copyright (c) 2025 OmniTrust Team

"""
Race Bias Evaluator

Detects race-related stereotypes and biases in LLM responses.
"""

from typing import Any, Dict

from ...bias_types import BiasType
from ..registry import EvaluatorRegistry
from .base import SocialBiasEvaluator, SocialBiasResult


@EvaluatorRegistry.register("race")
class RaceBiasEvaluator(SocialBiasEvaluator):
    """
    Evaluator for race bias detection.

    Race bias includes stereotypes about:
    - Different racial groups
    - Ethnic minorities
    - Cultural assumptions based on race

    Detection method:
    1. Present race-related stereotypical statement
    2. Ask model if it agrees
    3. Higher agreement rate indicates more bias
    """

    def __init__(self, **kwargs):
        super().__init__(bias_type=BiasType.RACE, name="race", **kwargs)

    def _get_bias_type_filter(self) -> str:
        """Return the Type field value for race bias in dataset."""
        return "Race"
