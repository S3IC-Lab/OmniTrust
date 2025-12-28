# OmniTrust Fairness Module - Distraction Bias Evaluator
# Copyright (c) 2025 OmniTrust Team

"""
Distraction Bias Evaluator

Detects distraction effect - tendency to be influenced by irrelevant
information when making evaluation decisions.
"""

import random
from typing import Any, Dict, List
from tqdm import tqdm

from ...bias_types import BiasType
from ...core.prompts import PromptTemplates
from ...core.parsers import ResponseParser
from ..registry import EvaluatorRegistry
from .base import CognitiveBiasEvaluator, CognitiveBiasResult


@EvaluatorRegistry.register("distraction")
class DistractionBiasEvaluator(CognitiveBiasEvaluator):
    """
    Evaluator for distraction bias detection.

    Distraction bias occurs when irrelevant information influences
    decision-making. In LLM evaluation, this manifests as being swayed
    by off-topic facts about systems.

    Detection method:
    1. Present two responses with irrelevant positive fact about System Star
    2. Present same responses with irrelevant positive fact about System Square
    3. If model switches preference based on the distraction, bias detected
    """

    def __init__(self, **kwargs):
        super().__init__(bias_type=BiasType.DISTRACTION, name="distraction", **kwargs)

    def _evaluate_impl(self, model: Any, data: Dict) -> CognitiveBiasResult:
        """
        Evaluate model for distraction bias.

        Args:
            model: Model to evaluate
            data: Dataset with N, evaluator, instructions, references, responses

        Returns:
            CognitiveBiasResult with distraction bias metrics
        """
        # Initialize counters
        distracted_count = 0
        valid_responses = 0
        consistency = 0

        evaluator = data.get("evaluator", "unknown")
        keys = self._get_model_keys(data)
        N = data.get("N", 0)
        instructions = data.get("instructions", [])
        references = data.get("references", [])
        responses = data.get("responses", {})

        rankings_info = {}
        response_list = []
        count = 0

        order = [self.SYSTEM_STAR, self.SYSTEM_SQUARE]

        # Iterate over samples
        for index in tqdm(range(N), total=N, desc="Evaluating distraction bias"):
            rankings = {k: 0 for k in keys}

            # Generate all pairwise combinations
            combinations = self._get_combinations(keys)

            for model1, model2 in combinations:
                count += 1
                models = [model1, model2]

                # Shuffle models for this comparison
                random.shuffle(models)

                # Get responses
                response1 = responses[models[0]][index]
                response2 = responses[models[1]][index]

                # Format input
                inp = self._format_responses(response1, response2)

                # Generate random distraction phrases for each system
                distraction_star = PromptTemplates.get_distraction_phrase(self.SYSTEM_STAR)
                distraction_square = PromptTemplates.get_distraction_phrase(self.SYSTEM_SQUARE)

                # Create prompts with different distractions
                prompt = PromptTemplates.distraction(
                    instructions[index],
                    references[index],
                    inp,
                    distraction_star
                )
                val_prompt = PromptTemplates.distraction(
                    instructions[index],
                    references[index],
                    inp,
                    distraction_square
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

                # Detect distraction bias
                # If model chose Star when positive fact about Star was mentioned,
                # and chose Square when positive fact about Square was mentioned,
                # that's distraction bias
                if pf == "System Star":
                    rankings[models[0]] += 1
                    if val == "System Square":
                        distracted_count += 1
                        models.append("distracted")

                elif pf == "System Square":
                    rankings[models[1]] += 1
                    # Note: Similar to bandwagon, count when following distraction

            rankings_info[index] = rankings

        total_comparisons = self._calculate_total_comparisons(N, len(keys))

        return CognitiveBiasResult(
            bias_type="distraction",
            bias_count=distracted_count,
            total_comparisons=total_comparisons,
            valid_responses=valid_responses,
            consistency=consistency,
            rankings_info=rankings_info,
            response_list=response_list,
        )
