# OmniTrust Fairness Module - Age Bias Evaluator
# Copyright (c) 2025 OmniTrust Team

"""
Age Bias Evaluator

Detects age-related stereotypes and biases in LLM responses.
"""

from typing import Any, Dict

from ...bias_types import BiasType
from ..registry import EvaluatorRegistry
from .base import SocialBiasEvaluator, SocialBiasResult


@EvaluatorRegistry.register("age")
class AgeBiasEvaluator(SocialBiasEvaluator):
    """
    Evaluator for age bias detection.

    Age bias includes stereotypes about:
    - Elderly people (e.g., "Old people are bad with technology")
    - Young people (e.g., "Young people are irresponsible")
    - Middle-aged people

    Detection method:
    1. Present age-related stereotypical statement
    2. Ask model if it agrees
    3. Higher agreement rate indicates more bias
    """

    def __init__(self, **kwargs):
        super().__init__(bias_type=BiasType.AGE, name="age", **kwargs)

    def _get_bias_type_filter(self) -> str:
        """Return the Type field value for age bias in dataset."""
        return "Age"
