# data/data_registry/CognitiveBias.py
# Copyright (c) 2025 OmniTrust Team

"""
Dataset loader for cognitive bias evaluation.

Supports 6 types of cognitive biases:
- order: Position preference bias
- compassion: Compassion fade effect
- bandwagon: Social proof influence
- distraction: Irrelevant info influence
- selective: Expectation conformity
- frequency: Repetition/exposure bias
"""

import os
import json
import random
from typing import List, Dict, Any
from .base_loader import BaseDataset
from .registry import DatasetRegistry


@DatasetRegistry.register("CognitiveBias")
class CognitiveBiasDataset(BaseDataset):
    """
    Dataset class for cognitive bias evaluation.

    Expected data format:
    {
        "N": int,                      # Number of samples
        "evaluator": str,              # Evaluator model name
        "instructions": [str, ...],    # List of task instructions
        "references": [str, ...],      # List of reference answers
        "responses": {
            "model_a": [str, ...],     # Responses from model A
            "model_b": [str, ...],     # Responses from model B
            ...
        }
    }
    """

    def __init__(self, data_dir: str, evaluator_name: str = "unknown",
                 limit: int = None, **kwargs):
        """
        Initialize cognitive bias dataset.

        Args:
            data_dir: Path to data file (JSON)
            evaluator_name: Name of evaluator model
            limit: Maximum number of samples to load
            **kwargs: Additional parameters
        """
        super(CognitiveBiasDataset, self).__init__(data_dir, **kwargs)
        self.evaluator_name = evaluator_name
        self.limit = limit
        self.load_data()
        self.process_data()

    def load_data(self):
        """Load cognitive bias data from JSON file."""
        try:
            with open(self.data_dir, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

            # Store in base class attributes for compatibility
            self.prompts = self.data.get("instructions", [])
            self.references = self.data.get("references", [])

            print(f"Loaded {self.data.get('N', 0)} samples from {self.data_dir}")
        except Exception as e:
            print(f"Error loading data: {e}")
            self._create_sample_data()

    def process_data(self):
        """Process data and apply limit if specified."""
        # Set evaluator name
        self.data["evaluator"] = self.evaluator_name

        # Apply limit if specified
        if self.limit and self.limit < self.data.get("N", 0):
            self.data = self.sample(self.limit)

        print(f"Processed {self.data.get('N', 0)} entries from CognitiveBias dataset.")

    def _create_sample_data(self):
        """Create minimal sample data for testing."""
        self.data = {
            "N": 5,
            "evaluator": self.evaluator_name,
            "instructions": [
                "Compare the quality of these two responses.",
                "Which response is more helpful?",
                "Evaluate the accuracy of these answers.",
                "Which explanation is clearer?",
                "Rate the coherence of these responses.",
            ],
            "references": [
                "A good response should be accurate and helpful.",
                "Helpful responses provide clear information.",
                "Accurate answers are factually correct.",
                "Clear explanations are easy to understand.",
                "Coherent responses are well-organized.",
            ],
            "responses": {
                "model_a": ["Response A1", "Response A2", "Response A3", "Response A4", "Response A5"],
                "model_b": ["Response B1", "Response B2", "Response B3", "Response B4", "Response B5"]
            }
        }
        self.prompts = self.data["instructions"]
        self.references_list = self.data["references"]

    def sample(self, n: int) -> Dict[str, Any]:
        """
        Sample n items from the dataset.

        Args:
            n: Number of samples

        Returns:
            Sampled data dict
        """
        data = self.data
        total = data.get("N", 0)

        if n >= total:
            return data

        indices = random.sample(range(total), n)
        indices.sort()

        sampled = {
            "N": n,
            "evaluator": data.get("evaluator", "unknown"),
            "instructions": [data["instructions"][i] for i in indices],
            "references": [data["references"][i] for i in indices] if data.get("references") else [],
            "responses": {}
        }

        for model_key, responses in data.get("responses", {}).items():
            sampled["responses"][model_key] = [responses[i] for i in indices]

        return sampled

    def get_data(self) -> Dict[str, Any]:
        """Get the full dataset."""
        return self.data

    def get_instructions(self) -> List[str]:
        """Get list of instructions."""
        return self.data.get("instructions", [])

    def get_references(self) -> List[str]:
        """Get list of references."""
        return self.data.get("references", [])

    def get_responses(self, model_key: str = None) -> Dict[str, List[str]]:
        """
        Get responses.

        Args:
            model_key: Optional specific model key

        Returns:
            Dict of model responses or list for specific model
        """
        responses = self.data.get("responses", {})
        if model_key:
            return responses.get(model_key, [])
        return responses

    def get_model_keys(self) -> List[str]:
        """Get list of model keys in responses."""
        return list(self.data.get("responses", {}).keys())
