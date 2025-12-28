# OmniTrust Fairness Module - Gender Bias Evaluator
# Copyright (c) 2025 OmniTrust Team

"""
Gender Bias Evaluator

Detects gender-related stereotypes and biases in LLM responses.
"""

from typing import Any, Dict

from ...bias_types import BiasType
from ..registry import EvaluatorRegistry
from .base import SocialBiasEvaluator, SocialBiasResult


@EvaluatorRegistry.register("gender")
class GenderBiasEvaluator(SocialBiasEvaluator):
    """
    Evaluator for gender bias detection.

    Gender bias includes stereotypes about:
    - Women (e.g., "Women are more emotional")
    - Men (e.g., "Men are better at math")
    - Non-binary individuals

    Detection method:
    1. Present gender-related stereotypical statement
    2. Ask model if it agrees
    3. Higher agreement rate indicates more bias
    """

    def __init__(self, **kwargs):
        super().__init__(bias_type=BiasType.GENDER, name="gender", **kwargs)

    def _get_bias_type_filter(self) -> str:
        """Return the Type field value for gender bias in dataset."""
        return "Gender"
