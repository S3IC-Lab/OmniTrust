"""
Social bias evaluators for LLM fairness assessment.

Available evaluators (6 types):
- AgeBiasEvaluator: Detects age-related stereotypes
- GenderBiasEvaluator: Detects gender-related stereotypes
- RaceBiasEvaluator: Detects race-related stereotypes
- ReligionBiasEvaluator: Detects religion-related stereotypes
- NationalityBiasEvaluator: Detects nationality-related stereotypes
- PoliticalBiasEvaluator: Detects political leaning bias
"""

from .base import SocialBiasEvaluator, SocialBiasResult
from .age import AgeBiasEvaluator
from .gender import GenderBiasEvaluator
from .race import RaceBiasEvaluator
from .religion import ReligionBiasEvaluator
from .nationality import NationalityBiasEvaluator
from .political import PoliticalBiasEvaluator

__all__ = [
    # Base class
    "SocialBiasEvaluator",
    "SocialBiasResult",
    # Evaluators (6 types)
    "AgeBiasEvaluator",
    "GenderBiasEvaluator",
    "RaceBiasEvaluator",
    "ReligionBiasEvaluator",
    "NationalityBiasEvaluator",
    "PoliticalBiasEvaluator",
]

# Mapping of bias names to evaluator classes
SOCIAL_EVALUATORS = {
    "age": AgeBiasEvaluator,
    "gender": GenderBiasEvaluator,
    "race": RaceBiasEvaluator,
    "religion": ReligionBiasEvaluator,
    "nationality": NationalityBiasEvaluator,
    "political": PoliticalBiasEvaluator,
}


def get_social_evaluator(name: str, **kwargs):
    """
    Get a social bias evaluator by name.

    Args:
        name: Evaluator name (age, gender, race, religion, nationality, political)
        **kwargs: Additional arguments to pass to evaluator

    Returns:
        Evaluator instance

    Raises:
        ValueError: If evaluator name not found
    """
    name_lower = name.lower()
    if name_lower not in SOCIAL_EVALUATORS:
        available = ", ".join(SOCIAL_EVALUATORS.keys())
        raise ValueError(f"Unknown evaluator '{name}'. Available: {available}")
    return SOCIAL_EVALUATORS[name_lower](**kwargs)
