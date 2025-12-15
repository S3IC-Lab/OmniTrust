"""
Bias Types and Data Structures
==============================

Defines the core data types, enumerations, and data structures used
throughout the Fairness module for bias evaluation.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime


class BiasType(Enum):
    """
    Enumeration of bias types supported by the Fairness module.

    These categories align with common bias taxonomies in AI fairness research.
    """
    # Cognitive Biases (from AwesomeLLMSecurityPlatform)
    ORDER = "order"                   # Order/position bias
    COMPASSION = "compassion"         # Compassion fade bias
    BANDWAGON = "bandwagon"           # Bandwagon effect
    DISTRACTION = "distraction"       # Distraction bias
    FREQUENCY = "frequency"           # Frequency/repetition bias
    SELECTIVE = "selective"           # Selective attention bias

    # Social/Demographic Biases
    GENDER = "gender"                 # Gender bias
    RACIAL = "racial"                 # Racial bias
    AGE = "age"                       # Age bias
    RELIGIOUS = "religious"           # Religious bias
    POLITICAL = "political"           # Political bias
    SOCIOECONOMIC = "socioeconomic"   # Socioeconomic bias
    DISABILITY = "disability"         # Disability bias
    NATIONALITY = "nationality"       # Nationality/origin bias

    # Representation Biases
    STEREOTYPE = "stereotype"         # Stereotyping
    REPRESENTATION = "representation" # Representation harm
    TOXICITY = "toxicity"             # Toxic associations

    # General
    DEMOGRAPHIC = "demographic"       # General demographic bias
    SOCIAL = "social"                 # General social bias
    OTHER = "other"                   # Other bias types


class BiasCategory(Enum):
    """
    Higher-level categorization of bias types.
    """
    COGNITIVE = "cognitive"           # Cognitive biases (order, bandwagon, etc.)
    DEMOGRAPHIC = "demographic"       # Demographic biases (gender, race, etc.)
    REPRESENTATION = "representation" # Representation biases
    SOCIAL = "social"                 # Social biases


class DemographicGroup(Enum):
    """
    Enumeration of demographic groups for bias analysis.
    """
    # Gender
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"

    # Age
    YOUNG = "young"
    MIDDLE_AGED = "middle_aged"
    ELDERLY = "elderly"

    # Race/Ethnicity (using broad categories)
    WHITE = "white"
    BLACK = "black"
    ASIAN = "asian"
    HISPANIC = "hispanic"
    INDIGENOUS = "indigenous"
    MIXED = "mixed"

    # Religion
    CHRISTIAN = "christian"
    MUSLIM = "muslim"
    JEWISH = "jewish"
    HINDU = "hindu"
    BUDDHIST = "buddhist"
    ATHEIST = "atheist"
    OTHER_RELIGION = "other_religion"

    # Socioeconomic
    LOW_INCOME = "low_income"
    MIDDLE_INCOME = "middle_income"
    HIGH_INCOME = "high_income"

    # Other
    OTHER = "other"


@dataclass
class BiasDataPoint:
    """
    Standard data structure for a single bias evaluation sample.

    Attributes:
        id: Unique identifier for the data point
        instruction: The instruction/prompt for the model
        responses: Dictionary mapping group/condition to responses
        reference: Optional reference/ground truth answer
        bias_type: Type of bias being tested
        demographic_groups: List of demographic groups involved
        metadata: Additional metadata
    """
    id: str
    instruction: str
    responses: Dict[str, List[str]] = field(default_factory=dict)
    reference: Optional[str] = None
    bias_type: Optional[BiasType] = None
    demographic_groups: List[DemographicGroup] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "instruction": self.instruction,
            "responses": self.responses,
            "reference": self.reference,
            "bias_type": self.bias_type.value if self.bias_type else None,
            "demographic_groups": [g.value for g in self.demographic_groups],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiasDataPoint':
        """Create from dictionary representation."""
        data = data.copy()
        if data.get('bias_type'):
            data['bias_type'] = BiasType(data['bias_type'])
        if data.get('demographic_groups'):
            data['demographic_groups'] = [
                DemographicGroup(g) for g in data['demographic_groups']
            ]
        return cls(**data)


@dataclass
class BiasEvaluationResult:
    """
    Result container for bias evaluation.

    Attributes:
        evaluator_id: Unique identifier for the evaluator
        bias_type: Type of bias evaluated
        metrics: Dictionary of computed metrics
        raw_results: List of individual evaluation results
        group_metrics: Metrics broken down by demographic group
        pairwise_comparisons: Pairwise comparison results
        timestamp: ISO format timestamp
        metadata: Additional metadata
    """
    evaluator_id: str
    bias_type: BiasType
    metrics: Dict[str, float]
    raw_results: List[Dict[str, Any]]
    group_metrics: Dict[str, Dict[str, float]] = None
    pairwise_comparisons: Dict[str, Any] = None
    timestamp: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.group_metrics is None:
            self.group_metrics = {}
        if self.pairwise_comparisons is None:
            self.pairwise_comparisons = {}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "evaluator_id": self.evaluator_id,
            "bias_type": self.bias_type.value,
            "metrics": self.metrics,
            "raw_results": self.raw_results,
            "group_metrics": self.group_metrics,
            "pairwise_comparisons": self.pairwise_comparisons,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiasEvaluationResult':
        """Create from dictionary representation."""
        data = data.copy()
        data['bias_type'] = BiasType(data['bias_type'])
        return cls(**data)

    def get_overall_bias_score(self) -> float:
        """
        Get an overall bias score from the metrics.

        Returns:
            Overall bias score (0-1, where 0 is no bias)
        """
        # Look for common bias score metrics
        for key in ['bias_score', 'overall_bias', 'bias_rate', 'disparity']:
            if key in self.metrics:
                return self.metrics[key]
        return 0.0

    def is_biased(self, threshold: float = 0.5) -> bool:
        """
        Determine if the evaluation indicates bias.

        Args:
            threshold: Bias score threshold

        Returns:
            True if biased, False otherwise
        """
        return self.get_overall_bias_score() > threshold


# Mapping of bias types to their categories
BIAS_TYPE_TO_CATEGORY: Dict[BiasType, BiasCategory] = {
    BiasType.ORDER: BiasCategory.COGNITIVE,
    BiasType.COMPASSION: BiasCategory.COGNITIVE,
    BiasType.BANDWAGON: BiasCategory.COGNITIVE,
    BiasType.DISTRACTION: BiasCategory.COGNITIVE,
    BiasType.FREQUENCY: BiasCategory.COGNITIVE,
    BiasType.SELECTIVE: BiasCategory.COGNITIVE,
    BiasType.GENDER: BiasCategory.DEMOGRAPHIC,
    BiasType.RACIAL: BiasCategory.DEMOGRAPHIC,
    BiasType.AGE: BiasCategory.DEMOGRAPHIC,
    BiasType.RELIGIOUS: BiasCategory.DEMOGRAPHIC,
    BiasType.POLITICAL: BiasCategory.DEMOGRAPHIC,
    BiasType.SOCIOECONOMIC: BiasCategory.DEMOGRAPHIC,
    BiasType.DISABILITY: BiasCategory.DEMOGRAPHIC,
    BiasType.NATIONALITY: BiasCategory.DEMOGRAPHIC,
    BiasType.STEREOTYPE: BiasCategory.REPRESENTATION,
    BiasType.REPRESENTATION: BiasCategory.REPRESENTATION,
    BiasType.TOXICITY: BiasCategory.REPRESENTATION,
    BiasType.DEMOGRAPHIC: BiasCategory.DEMOGRAPHIC,
    BiasType.SOCIAL: BiasCategory.SOCIAL,
    BiasType.OTHER: BiasCategory.SOCIAL,
}


def get_bias_category(bias_type: BiasType) -> BiasCategory:
    """
    Get the category for a bias type.

    Args:
        bias_type: The bias type

    Returns:
        The corresponding category
    """
    return BIAS_TYPE_TO_CATEGORY.get(bias_type, BiasCategory.SOCIAL)


def get_cognitive_bias_types() -> List[BiasType]:
    """Get all cognitive bias types."""
    return [
        bt for bt, cat in BIAS_TYPE_TO_CATEGORY.items()
        if cat == BiasCategory.COGNITIVE
    ]


def get_demographic_bias_types() -> List[BiasType]:
    """Get all demographic bias types."""
    return [
        bt for bt, cat in BIAS_TYPE_TO_CATEGORY.items()
        if cat == BiasCategory.DEMOGRAPHIC
    ]
