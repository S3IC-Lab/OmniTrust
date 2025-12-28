# OmniTrust Fairness Module - Compassion Bias Evaluator
# Copyright (c) 2025 OmniTrust Team

"""
Compassion Fade Bias Evaluator

Detects compassion fade effect - tendency to show less concern or preference
for models when they are identified by name vs. anonymous labels.
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


@EvaluatorRegistry.register("compassion")
class CompassionBiasEvaluator(CognitiveBiasEvaluator):
    """
    Evaluator for compassion fade bias detection.

    Compassion fade occurs when emotional response diminishes with
    psychological distance. In LLM evaluation context, this manifests
    as different treatment when model names are visible vs. anonymous.

    Detection method:
    1. Present responses with model names visible (e.g., "Model GPT-4", "Model Claude")
    2. Compare with anonymous evaluation (System Star/Square)
    3. Check for systematic preference based on model identity
    """

    def __init__(self, **kwargs):
        super().__init__(bias_type=BiasType.COMPASSION, name="compassion", **kwargs)

    def _evaluate_impl(self, model: Any, data: Dict) -> CognitiveBiasResult:
        """
        Evaluate model for compassion fade bias.

        Args:
            model: Model to evaluate
            data: Dataset with N, evaluator, instructions, references, responses

        Returns:
            CognitiveBiasResult with compassion bias metrics
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
        for index in tqdm(range(N), total=N, desc="Evaluating compassion bias"):
            rankings = {k: 0 for k in keys}

            # Generate all pairwise combinations
            combinations = self._get_combinations(keys)

            for model1, model2 in combinations:
                count += 1
                models = [model1, model2]

                # Check if evaluator is being compared
                if evaluator in models or model1 in evaluator or model2 in evaluator:
                    me_compared += 1

                # Prepare model labels (add "(You)" for self-comparison)
                label1 = model1 + " (You)" if model1 == evaluator else model1
                label2 = model2 + " (You)" if model2 == evaluator else model2

                # Get responses
                response1 = data["responses"][models[0]][index]
                response2 = data["responses"][models[1]][index]

                # Format inputs with model names (compassion-specific)
                inp = PromptTemplates.format_compassion_input(
                    response1, response2, label1, label2
                )
                val_inp = PromptTemplates.format_compassion_input(
                    response2, response1, label2, label1
                )

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
                    evaluation = random.choice(models)
                    validation = random.choice(models)

                # Parse responses (compassion uses model names)
                preference = ResponseParser.parse_compassion(evaluation)
                val_preference = ResponseParser.parse_compassion(validation)

                # Log periodically
                if count % N == 0:
                    self._log_generation(
                        response_list, models, index,
                        evaluation, preference, validation, val_preference
                    )

                # Remove "(You)" suffix for matching
                clean_model1 = model1.replace(" (You)", "")
                clean_model2 = model2.replace(" (You)", "")

                # Create patterns for matching
                pattern = f"(?i)(system {clean_model1}|system {clean_model2})"

                # Extract preferences
                pf = None
                val = None

                if re.findall(pattern, preference):
                    pf = re.findall(pattern, preference)[0].lower()
                if re.findall(pattern, val_preference):
                    val = re.findall(pattern, val_preference)[0].lower()

                # Check validity and consistency
                if pf is not None:
                    valid_responses += 1
                    if val is not None:
                        consistency += 1
                    else:
                        models.append("inconsistent")

                match1 = f"system {clean_model1}".lower()
                match2 = f"system {clean_model2}".lower()

                # Analyze bias
                if pf == match1:
                    rankings[models[0]] += 1
                    if val == match1:
                        first_order_bias += 1
                        models.append("fo bias")
                    elif models[0] == evaluator:
                        me_bias += 1

                elif pf == match2:
                    rankings[models[1]] += 1
                    if val == match2:
                        last_order_bias += 1
                        models.append("lo bias")
                    elif models[1] == evaluator:
                        me_bias += 1

            rankings_info[index] = rankings

        total_comparisons = self._calculate_total_comparisons(N, len(keys))

        return CognitiveBiasResult(
            bias_type="compassion",
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
