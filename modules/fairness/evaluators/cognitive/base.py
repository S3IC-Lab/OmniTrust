# OmniTrust Fairness Module - Cognitive Bias Evaluator Base
# Copyright (c) 2025 OmniTrust Team

"""
Base class for cognitive bias evaluators.

Cognitive biases are systematic patterns of deviation from rationality in judgment.
This module provides the foundation for evaluating LLMs for such biases.
"""

import random
import itertools
from abc import abstractmethod
from math import comb
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ...base_bias_evaluator import BaseBiasEvaluator
from ...bias_types import BiasType, BiasCategory
from ..registry import evaluator_registry


@dataclass
class CognitiveBiasResult:
    """Result container for cognitive bias evaluation."""

    bias_type: str
    bias_count: int
    total_comparisons: int
    valid_responses: int
    consistency: int

    # Order-specific metrics (for order/compassion/frequency)
    first_order_bias: int = 0
    last_order_bias: int = 0

    # Me bias metrics (for compassion/frequency)
    me_bias: int = 0
    me_compared: int = 0

    # Detailed results
    rankings_info: Dict[int, Dict[str, int]] = field(default_factory=dict)
    response_list: List[str] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    evaluator_name: str = ""
    model_name: str = ""
    dataset_name: str = ""

    @property
    def bias_rate(self) -> float:
        """Calculate bias rate as percentage."""
        if self.total_comparisons == 0:
            return 0.0
        return self.bias_count / self.total_comparisons

    @property
    def valid_response_rate(self) -> float:
        """Calculate valid response rate."""
        if self.total_comparisons == 0:
            return 0.0
        return self.valid_responses / self.total_comparisons

    @property
    def consistency_rate(self) -> float:
        """Calculate consistency rate."""
        if self.total_comparisons == 0:
            return 0.0
        return self.consistency / self.total_comparisons

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bias_type": self.bias_type,
            "bias_count": self.bias_count,
            "bias_rate": self.bias_rate,
            "total_comparisons": self.total_comparisons,
            "valid_responses": self.valid_responses,
            "valid_response_rate": self.valid_response_rate,
            "consistency": self.consistency,
            "consistency_rate": self.consistency_rate,
            "first_order_bias": self.first_order_bias,
            "last_order_bias": self.last_order_bias,
            "me_bias": self.me_bias,
            "me_compared": self.me_compared,
            "timestamp": self.timestamp,
            "evaluator_name": self.evaluator_name,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
        }


class CognitiveBiasEvaluator(BaseBiasEvaluator):
    """
    Base class for cognitive bias evaluators.

    Cognitive biases evaluated:
    - Order bias: preference for first or last option
    - Compassion fade: reduced empathy with distance/scale
    - Bandwagon effect: following majority opinion
    - Distraction: influenced by irrelevant information
    - Selective attention: influenced by expectations
    - Frequency bias: influenced by repeated exposure

    All evaluators follow a pairwise comparison pattern:
    1. Generate pairs of responses from dataset
    2. Ask model to evaluate which is better
    3. Swap order and re-evaluate to detect bias
    4. Compute bias metrics
    """

    # Class constants
    SYSTEM_STAR = "System Star"
    SYSTEM_SQUARE = "System Square"
    RANDOM_SEED = 939

    def __init__(self, bias_type: BiasType, name: str = None, **kwargs):
        """
        Initialize cognitive bias evaluator.

        Args:
            bias_type: The type of cognitive bias to evaluate
            name: Optional evaluator name
            **kwargs: Additional parameters
        """
        super().__init__(bias_type=bias_type, name=name or bias_type.value, **kwargs)
        self.category = BiasCategory.COGNITIVE
        random.seed(self.RANDOM_SEED)

    def _get_model_keys(self, data: Dict) -> List[str]:
        """Extract model keys from dataset responses."""
        return list(data.get("responses", {}).keys())

    def _get_combinations(self, keys: List[str]) -> List[Tuple[str, str]]:
        """Generate shuffled combinations of model pairs."""
        combinations = list(itertools.combinations(keys, 2))
        random.shuffle(combinations)
        return combinations

    def _calculate_total_comparisons(self, n_samples: int, n_models: int) -> int:
        """Calculate total number of pairwise comparisons."""
        return n_samples * comb(n_models, 2)

    def _format_responses(self, response1: str, response2: str,
                          system1: str = None, system2: str = None) -> str:
        """Format two responses for comparison."""
        s1 = system1 or self.SYSTEM_STAR
        s2 = system2 or self.SYSTEM_SQUARE
        return f"{s1}: {response1}\n{s2}: {response2}"

    def _call_model(self, model: Any, prompt: str) -> str:
        """
        Call the model with a prompt and get response.

        Args:
            model: Model interface
            prompt: Prompt text

        Returns:
            Model response text
        """
        try:
            # Try standard generate interface
            if hasattr(model, 'generate'):
                result = model.generate([prompt])
                if isinstance(result, list) and len(result) > 0:
                    item = result[0]
                    if isinstance(item, dict) and 'generation' in item:
                        return item['generation']
                    elif isinstance(item, str):
                        return item
                elif isinstance(result, dict) and 'generation' in result:
                    return result['generation']
                return str(result)

            # Fallback: try call interface
            if callable(model):
                return str(model(prompt))

            raise ValueError(f"Unknown model interface: {type(model)}")

        except Exception as e:
            self.logger.error(f"Model call failed: {e}")
            return ""

    def _log_generation(self, response_list: List[str], models: List[str],
                        index: int, evaluation: str, preference: str,
                        validation: str = None, val_preference: str = None):
        """Log generation details for debugging."""
        response_list.append(
            f"========================Generation for [{', '.join(models)}] "
            f"for instance {index} ============================\n"
        )
        response_list.append(f"---------RAW GENERATION--------\n{evaluation}\n")
        response_list.append(f"---------PATTERN MATCHED-------\n{preference}\n")
        if validation:
            response_list.append(f"---------VAL RAW GENERATION--------\n{validation}\n")
        if val_preference:
            response_list.append(f"---------VAL PATTERN MATCHED-------\n{val_preference}\n\n")

    @abstractmethod
    def _evaluate_impl(self, model: Any, data: Dict) -> CognitiveBiasResult:
        """
        Implementation of bias evaluation logic.

        Subclasses must implement this method.

        Args:
            model: Model to evaluate
            data: Dataset dictionary with keys:
                - N: number of samples
                - evaluator: evaluator name
                - instructions: list of instructions
                - references: list of references
                - responses: dict of {model_name: [responses]}

        Returns:
            CognitiveBiasResult with evaluation metrics
        """
        pass

    def evaluate(self, model: Any, dataset: Any) -> Dict[str, Any]:
        """
        Evaluate model for cognitive bias.

        Args:
            model: Model interface with generate() method
            dataset: Dataset object with .data attribute or dict

        Returns:
            Dictionary with evaluation results
        """
        # Extract data from dataset
        if hasattr(dataset, 'data'):
            data = dataset.data
        elif isinstance(dataset, dict):
            data = dataset
        else:
            raise ValueError(f"Unknown dataset type: {type(dataset)}")

        # Run implementation
        result = self._evaluate_impl(model, data)

        # Add metadata
        result.evaluator_name = self.name
        if hasattr(model, 'get_model_info'):
            result.model_name = model.get_model_info().get('name', 'unknown')
        if hasattr(dataset, 'get_metadata'):
            result.dataset_name = dataset.get_metadata().get('name', 'unknown')

        return result.to_dict()

    def _do_evaluate(self, model: Any, dataset: Any, **kwargs) -> List[Dict]:
        """Implementation of abstract method from BaseBiasEvaluator."""
        result = self.evaluate(model, dataset)
        return [result]

    def _compute_metrics(self, raw_results: List[Dict]) -> Dict[str, Any]:
        """Compute metrics from raw results."""
        if not raw_results:
            return {}

        result = raw_results[0]
        return {
            "bias_rate": result.get("bias_rate", 0),
            "valid_response_rate": result.get("valid_response_rate", 0),
            "consistency_rate": result.get("consistency_rate", 0),
            "total_comparisons": result.get("total_comparisons", 0),
        }

    def get_supported_metrics(self) -> List[str]:
        """Return list of supported metrics."""
        return [
            "bias_rate",
            "bias_count",
            "valid_response_rate",
            "consistency_rate",
            "total_comparisons",
            "first_order_bias",
            "last_order_bias",
            "me_bias",
        ]
