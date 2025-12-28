# OmniTrust Fairness Module - Frequency Bias Evaluator
# Copyright (c) 2025 OmniTrust Team

"""
Frequency (Repetition) Bias Evaluator

Detects frequency/repetition bias - tendency to prefer options
that appear more frequently in the prompt.
"""

import random
from typing import Any, Dict, List
from tqdm import tqdm

from ...bias_types import BiasType
from ...core.prompts import PromptTemplates
from ...core.parsers import ResponseParser
from ..registry import EvaluatorRegistry
from .base import CognitiveBiasEvaluator, CognitiveBiasResult


@EvaluatorRegistry.register("frequency")
class FrequencyBiasEvaluator(CognitiveBiasEvaluator):
    """
    Evaluator for frequency/repetition bias detection.

    Frequency bias (also known as the mere exposure effect) occurs when
    repeated exposure to something increases preference for it. In LLM
    evaluation, this manifests as preferring responses that appear more
    frequently in the prompt.

    Detection method:
    1. Present responses with System Star appearing 3 times, Square once
    2. Present same responses with System Square appearing 3 times, Star once
    3. If model switches preference to follow the more frequent system, bias detected
    """

    def __init__(self, **kwargs):
        super().__init__(bias_type=BiasType.FREQUENCY, name="frequency", **kwargs)

    def _create_frequency_mix(self, dominant: str, secondary: str) -> List[str]:
        """
        Create a shuffled list with 3 dominant and 1 secondary system.

        Args:
            dominant: System to appear 3 times
            secondary: System to appear 1 time

        Returns:
            Shuffled list of system identifiers
        """
        mix = [dominant, dominant, dominant, secondary]
        random.shuffle(mix)
        return mix

    def _format_frequency_input(self, data: Dict, index: int,
                                 order_mapping: Dict[str, str],
                                 mix_order: List[str]) -> str:
        """
        Format input with repeated system responses.

        Args:
            data: Dataset dictionary
            index: Sample index
            order_mapping: Map from system name to model key
            mix_order: Order of systems (with repetition)

        Returns:
            Formatted input string
        """
        lines = []
        for system in mix_order:
            model_key = order_mapping[system]
            response = data["responses"][model_key][index]
            lines.append(f"{system}: {response}")
        return "\n".join(lines)

    def _evaluate_impl(self, model: Any, data: Dict) -> CognitiveBiasResult:
        """
        Evaluate model for frequency bias.

        Args:
            model: Model to evaluate
            data: Dataset with N, evaluator, instructions, references, responses

        Returns:
            CognitiveBiasResult with frequency bias metrics
        """
        # Initialize counters
        frequency_bias = 0
        me_bias = 0
        me_compared = 0
        valid_responses = 0
        consistency = 0

        evaluator = data.get("evaluator", "unknown")
        keys = self._get_model_keys(data)
        N = data.get("N", 0)

        rankings_info = {}
        response_list = []
        count = 0

        # Iterate over samples
        for index in tqdm(range(N), total=N, desc="Evaluating frequency bias"):
            rankings = {k: 0 for k in keys}

            # Generate all pairwise combinations
            combinations = self._get_combinations(keys)

            for model1, model2 in combinations:
                count += 1
                models = [model1, model2]

                # Check if evaluator is being compared
                if evaluator in models:
                    me_compared += 1

                # Map systems to models
                order_mapping = {
                    self.SYSTEM_STAR: model1,
                    self.SYSTEM_SQUARE: model2
                }

                # Create frequency mixes (3:1 ratio)
                star_mix = self._create_frequency_mix(self.SYSTEM_STAR, self.SYSTEM_SQUARE)
                square_mix = self._create_frequency_mix(self.SYSTEM_SQUARE, self.SYSTEM_STAR)

                # Format inputs with repeated systems
                inp = self._format_frequency_input(data, index, order_mapping, star_mix)
                val_inp = self._format_frequency_input(data, index, order_mapping, square_mix)

                # Generate prompts
                prompt = PromptTemplates.standard(
                    data["instructions"][index],
                    data["references"][index],
                    inp
                )
                val_prompt = PromptTemplates.standard(
                    data["instructions"][index],
                    data["references"][index],
                    val_inp
                )

                # Get model responses
                if evaluator != "random":
                    evaluation = self._call_model(model, prompt)
                    validation = self._call_model(model, val_prompt)
                else:
                    evaluation = random.choice(star_mix)
                    validation = random.choice(square_mix)

                # Parse responses
                preference = ResponseParser.parse_standard(evaluation)
                val_preference = ResponseParser.parse_standard(validation)

                # Log periodically
                if count % N == 0:
                    self._log_generation(
                        response_list, models, index,
                        evaluation, preference, validation, val_preference
                    )

                # Extract preferences
                pf = ResponseParser.extract_preference_string(preference)
                val = ResponseParser.extract_preference_string(val_preference)

                # Check validity and consistency
                if pf is not None:
                    valid_responses += 1
                    if val is not None:
                        consistency += 1
                    else:
                        models.append("inconsistent")

                # Detect frequency bias
                # In first prompt, Star appears 3x; in second, Square appears 3x
                # If model follows frequency, it chooses Star then Square
                if pf == "System Star":
                    rankings[models[0]] += 1
                    if val == "System Square":
                        frequency_bias += 1
                        models.append("frequency")
                    elif models[0] == evaluator:
                        me_bias += 1

                elif pf == "System Square":
                    rankings[models[1]] += 1
                    if models[1] == evaluator:
                        me_bias += 1

            rankings_info[index] = rankings

        total_comparisons = self._calculate_total_comparisons(N, len(keys))

        return CognitiveBiasResult(
            bias_type="frequency",
            bias_count=frequency_bias,
            total_comparisons=total_comparisons,
            valid_responses=valid_responses,
            consistency=consistency,
            me_bias=me_bias,
            me_compared=me_compared,
            rankings_info=rankings_info,
            response_list=response_list,
        )
