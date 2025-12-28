# OmniTrust Fairness Module - Order Bias Evaluator
# Copyright (c) 2025 OmniTrust Team

"""
Order Bias Evaluator

Detects position bias in LLM evaluations - tendency to prefer options
based on their position (first or last) rather than content quality.
"""

import re
import random
from typing import Any, Dict, List
from tqdm import tqdm

from ...bias_types import BiasType
from ...core.prompts import PromptTemplates
from ...core.parsers import ResponseParser
from ..registry import EvaluatorRegistry
from .base import CognitiveBiasEvaluator, CognitiveBiasResult


@EvaluatorRegistry.register("order")
class OrderBiasEvaluator(CognitiveBiasEvaluator):
    """
    Evaluator for order (position) bias detection.

    Order bias occurs when a model systematically prefers responses based on
    their position in the prompt rather than their actual quality:
    - First position bias: tends to prefer the first option
    - Last position bias: tends to prefer the last option

    Detection method:
    1. Present two responses in order A, B
    2. Ask model which is better
    3. Swap order to B, A
    4. Ask again
    5. If model chooses the same position (not same content), bias detected
    """

    def __init__(self, **kwargs):
        super().__init__(bias_type=BiasType.ORDER, name="order", **kwargs)

    def _evaluate_impl(self, model: Any, data: Dict) -> CognitiveBiasResult:
        """
        Evaluate model for order bias.

        Args:
            model: Model to evaluate
            data: Dataset with N, evaluator, instructions, references, responses

        Returns:
            CognitiveBiasResult with order bias metrics
        """
        # Initialize counters
        first_order_bias = 0
        last_order_bias = 0
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
        for index in tqdm(range(N), total=N, desc="Evaluating order bias"):
            rankings = {k: 0 for k in keys}

            # Generate all pairwise combinations
            combinations = self._get_combinations(keys)

            for model1, model2 in combinations:
                count += 1
                models = [model1, model2]

                # Check if evaluator is being compared
                if evaluator in models or model1 in evaluator or model2 in evaluator:
                    me_compared += 1

                order = [self.SYSTEM_STAR, self.SYSTEM_SQUARE]

                # Get responses
                response1 = data["responses"][models[0]][index]
                response2 = data["responses"][models[1]][index]

                # Format inputs (original and swapped)
                inp = self._format_responses(response1, response2)
                val_inp = self._format_responses(response2, response1)

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
                    evaluation = random.choice(order)
                    validation = random.choice(order)

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

                # Analyze bias
                if pf == "System Star":
                    rankings[models[0]] += 1
                    # Check for first order bias
                    if val == "System Star":
                        first_order_bias += 1
                        models.append("fo bias")
                    elif models[0] == evaluator or models[0] in evaluator:
                        me_bias += 1

                elif pf == "System Square":
                    rankings[models[1]] += 1
                    # Check for last order bias
                    if val == "System Square":
                        last_order_bias += 1
                        models.append("lo bias")
                    elif models[1] == evaluator or models[1] in evaluator:
                        me_bias += 1

            rankings_info[index] = rankings

        total_comparisons = self._calculate_total_comparisons(N, len(keys))

        return CognitiveBiasResult(
            bias_type="order",
            bias_count=first_order_bias + last_order_bias,
            total_comparisons=total_comparisons,
            valid_responses=valid_responses,
            consistency=consistency,
            first_order_bias=first_order_bias,
            last_order_bias=last_order_bias,
            me_bias=me_bias,
            me_compared=me_compared,
            rankings_info=rankings_info,
            response_list=response_list,
        )
