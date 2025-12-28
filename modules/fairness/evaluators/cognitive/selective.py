# OmniTrust Fairness Module - Selective Attention Bias Evaluator
# Copyright (c) 2025 OmniTrust Team

"""
Selective Attention Bias Evaluator

Detects selective attention bias - tendency to align with stated
expectations even when instructed not to.
"""

import random
from typing import Any, Dict, List
from tqdm import tqdm

from ...bias_types import BiasType
from ...core.prompts import PromptTemplates
from ...core.parsers import ResponseParser
from ..registry import EvaluatorRegistry
from .base import CognitiveBiasEvaluator, CognitiveBiasResult


@EvaluatorRegistry.register("selective")
class SelectiveBiasEvaluator(CognitiveBiasEvaluator):
    """
    Evaluator for selective attention bias detection.

    Selective attention bias occurs when prior expectations influence
    perception and judgment. In LLM evaluation, this manifests as
    conforming to stated expectations even when asked to ignore them.

    Detection method:
    1. Present responses with expectation "We expect System Star/Square to be better"
    2. Explicitly instruct model not to let expectation influence answer
    3. If model's choice matches the stated expectation, bias detected
    """

    def __init__(self, **kwargs):
        super().__init__(bias_type=BiasType.SELECTIVE, name="selective", **kwargs)

    def _evaluate_impl(self, model: Any, data: Dict) -> CognitiveBiasResult:
        """
        Evaluate model for selective attention bias.

        Args:
            model: Model to evaluate
            data: Dataset with N, evaluator, instructions, references, responses

        Returns:
            CognitiveBiasResult with selective bias metrics
        """
        # Initialize counters
        selection_count = 0
        valid_responses = 0
        consistency = 0  # Not used in selective but kept for compatibility

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
        for index in tqdm(range(N), total=N, desc="Evaluating selective bias"):
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

                # Randomly choose which system to expect as better
                selective = random.choice(order)

                # Create prompt with expectation statement
                prompt = PromptTemplates.selective(
                    instructions[index],
                    references[index],
                    inp,
                    selective
                )

                # Get model response
                evaluation = self._call_model(model, prompt)

                # Parse response
                preference = ResponseParser.parse_standard(evaluation)

                # Log periodically
                if count % N == 0:
                    self._log_generation(
                        response_list, models, index,
                        evaluation, preference
                    )

                # Extract preference
                pf = ResponseParser.extract_preference_string(preference)

                # Check validity
                if pf is not None:
                    valid_responses += 1

                # Detect selective bias
                # If model's choice matches the stated expectation, count as bias
                if pf == "System Star":
                    rankings[models[0]] += 1
                    if pf == selective:
                        selection_count += 1
                        models.append("selective")

                elif pf == "System Square":
                    rankings[models[1]] += 1
                    if pf == selective:
                        selection_count += 1
                        models.append("selective")

            rankings_info[index] = rankings

        total_comparisons = self._calculate_total_comparisons(N, len(keys))

        return CognitiveBiasResult(
            bias_type="selective",
            bias_count=selection_count,
            total_comparisons=total_comparisons,
            valid_responses=valid_responses,
            consistency=consistency,
            rankings_info=rankings_info,
            response_list=response_list,
        )
