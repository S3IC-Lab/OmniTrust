# OmniTrust Fairness Module - Nationality Bias Evaluator
# Copyright (c) 2025 OmniTrust Team

"""
Nationality Bias Evaluator

Detects nationality-related stereotypes and biases in LLM responses.
"""

from typing import Any, Dict

from ...bias_types import BiasType
from ..registry import EvaluatorRegistry
from .base import SocialBiasEvaluator, SocialBiasResult


@EvaluatorRegistry.register("nationality")
class NationalityBiasEvaluator(SocialBiasEvaluator):
    """
    Evaluator for nationality bias detection.

    Nationality bias includes stereotypes about:
    - People from different countries
    - National characteristics
    - Cultural assumptions based on nationality

    Detection method:
    1. Present nationality-related stereotypical statement
    2. Ask model if it agrees
    3. Higher agreement rate indicates more bias
    """

    def __init__(self, **kwargs):
        super().__init__(bias_type=BiasType.NATIONALITY, name="nationality", **kwargs)

    def _get_bias_type_filter(self) -> str:
        """Return the Type field value for nationality bias in dataset."""
        return "Nationality"
