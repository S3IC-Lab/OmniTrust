# OmniTrust Fairness Module - Religion Bias Evaluator
# Copyright (c) 2025 OmniTrust Team

"""
Religion Bias Evaluator

Detects religion-related stereotypes and biases in LLM responses.
"""

from typing import Any, Dict

from ...bias_types import BiasType
from ..registry import EvaluatorRegistry
from .base import SocialBiasEvaluator, SocialBiasResult


@EvaluatorRegistry.register("religion")
class ReligionBiasEvaluator(SocialBiasEvaluator):
    """
    Evaluator for religion bias detection.

    Religion bias includes stereotypes about:
    - Different religious groups
    - Religious practices
    - Assumptions based on religious affiliation

    Detection method:
    1. Present religion-related stereotypical statement
    2. Ask model if it agrees
    3. Higher agreement rate indicates more bias
    """

    def __init__(self, **kwargs):
        super().__init__(bias_type=BiasType.RELIGION, name="religion", **kwargs)

    def _get_bias_type_filter(self) -> str:
        """Return the Type field value for religion bias in dataset."""
        return "Religion"
