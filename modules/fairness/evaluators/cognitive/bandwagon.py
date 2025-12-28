# OmniTrust Fairness Module - Bandwagon Bias Evaluator
# Copyright (c) 2025 OmniTrust Team

"""
Bandwagon Bias Evaluator

Detects bandwagon effect - tendency to follow majority opinion or
social proof when making evaluation decisions.
"""

import random
from typing import Any, Dict, List
from tqdm import tqdm

from ...bias_types import BiasType
from ...core.prompts import PromptTemplates
from ...core.parsers import ResponseParser
from ..registry import EvaluatorRegistry
from .base import CognitiveBiasEvaluator, CognitiveBiasResult


@EvaluatorRegistry.register("bandwagon")
class BandwagonBiasEvaluator(CognitiveBiasEvaluator):
    """
    Evaluator for bandwagon effect bias detection.

    The bandwagon effect is a cognitive bias where individuals adopt beliefs
    or behaviors because others do so. In LLM evaluation, this manifests as
    being influenced by claims about majority preference.

    Detection method:
    1. Present two responses with claim "X% prefer System Star"
    2. Present same responses with claim "X% prefer System Square"
    3. If model switches preference based on the claim, bias detected
    """

    def __init__(self, **kwargs):
        super().__init__(bias_type=BiasType.BANDWAGON, name="bandwagon", **kwargs)

    def _evaluate_impl(self, model: Any, data: Dict) -> CognitiveBiasResult:
        """
        Evaluate model for bandwagon bias.

        Args:
            model: Model to evaluate
            data: Dataset with N, evaluator, instructions, references, responses

        Returns:
            CognitiveBiasResult with bandwagon bias metrics
        """
        # Initialize counters
        bandwagon_count = 0
        valid_responses = 0
        consistency = 0

        evaluator = data.get("evaluator", "unknown")
        keys = self._get_model_keys(data)
        N = data.get("N", 0)

        rankings_info = {}
        response_list = []
        count = 0

        order = [self.SYSTEM_STAR, self.SYSTEM_SQUARE]

        # Iterate over samples
        for index in tqdm(range(N), total=N, desc="Evaluating bandwagon bias"):
            rankings = {k: 0 for k in keys}

            # Generate all pairwise combinations
            combinations = self._get_combinations(keys)

            for model1, model2 in combinations:
                count += 1
                models = [model1, model2]

                # Shuffle models for this comparison
                random.shuffle(models)

                # Get responses
                response1 = data["responses"][models[0]][index]
                response2 = data["responses"][models[1]][index]

                # Format input
                inp = self._format_responses(response1, response2)

                # Generate random percentage for bandwagon claim
                percent = random.randint(
                    PromptTemplates.BANDWAGON_PERCENT_MIN,
                    PromptTemplates.BANDWAGON_PERCENT_MAX
                )

                # Create prompts with opposite bandwagon claims
                prompt = PromptTemplates.bandwagon(
                    data["instructions"][index],
                    data["references"][index],
                    inp,
                    self.SYSTEM_STAR,
                    percent
                )
                val_prompt = PromptTemplates.bandwagon(
                    data["instructions"][index],
                    data["references"][index],
                    inp,
                    self.SYSTEM_SQUARE,
                    percent
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

                # Detect bandwagon bias
                # If model chose Star when told Star is popular,
                # and chose Square when told Square is popular,
                # that's bandwagon bias
                if pf == "System Star":
                    rankings[models[0]] += 1
                    if val == "System Square":
                        bandwagon_count += 1
                        models.append("bandwagon")

                elif pf == "System Square":
                    rankings[models[1]] += 1
                    # Note: we only count when following the bandwagon claim
                    # The prompt claimed Star is better first, Square second

            rankings_info[index] = rankings

        total_comparisons = self._calculate_total_comparisons(N, len(keys))

        return CognitiveBiasResult(
            bias_type="bandwagon",
            bias_count=bandwagon_count,
            total_comparisons=total_comparisons,
            valid_responses=valid_responses,
            consistency=consistency,
            rankings_info=rankings_info,
            response_list=response_list,
        )
