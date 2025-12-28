# data/data_registry/FlipBias.py
# Copyright (c) 2025 OmniTrust Team

"""
Dataset loader for FlipBias political bias evaluation.

FlipBias is a dataset for evaluating political leaning bias in LLMs.
Each item contains text labeled as Left, Center, or Right political leaning.

Expected data format (flipbias_testset.txt):
Tab-separated values: id\titem_1\ttext\tlabel

Example:
1	1	Trump Accuses His Justice Department...	Left
2	1	Explosive memo released as Trump...	Center
3	1	Trump accuses FBI, DOJ leadership...	Right
"""

import os
import json
import random
from typing import List, Dict, Any, Optional
from .base_loader import BaseDataset
from .registry import DatasetRegistry


@DatasetRegistry.register("FlipBias")
class FlipBiasDataset(BaseDataset):
    """
    Dataset class for FlipBias political bias evaluation.

    Loads data from tab-separated text file with format:
    id\titem_1\ttext\tlabel

    Where label is one of: Left, Center, Right
    """

    # Valid political labels
    VALID_LABELS = ["Left", "Center", "Right"]

    def __init__(
        self,
        data_dir: str,
        evaluator_name: str = "unknown",
        limit: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize FlipBias dataset.

        Args:
            data_dir: Path to data file (txt or json)
            evaluator_name: Name of evaluator model
            limit: Maximum number of samples to load
            **kwargs: Additional parameters
        """
        super(FlipBiasDataset, self).__init__(data_dir, **kwargs)
        self.evaluator_name = evaluator_name
        self.limit = limit
        self.load_data()
        self.process_data()

    def load_data(self):
        """Load FlipBias data from txt or json file."""
        try:
            if self.data_dir.endswith('.txt'):
                self._load_txt()
            elif self.data_dir.endswith('.json'):
                self._load_json()
            else:
                # Try txt first, then json
                try:
                    self._load_txt()
                except:
                    self._load_json()

            print(f"Loaded {self.data.get('N', 0)} samples from {self.data_dir}")

        except FileNotFoundError:
            print(f"Data file not found: {self.data_dir}")
            self._create_sample_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            self._create_sample_data()

    def _load_txt(self):
        """Load data from tab-separated text file (flipbias_testset.txt format)."""
        fb_ids = []
        items_1 = []
        texts = []
        labels = []

        with open(self.data_dir, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) >= 4:
                    fb_ids.append(parts[0])
                    items_1.append(parts[1])
                    texts.append(parts[2])
                    labels.append(parts[3])

        self.data = {
            "N": len(fb_ids),
            "fb_ids": fb_ids,
            "items_1": items_1,
            "texts": texts,
            "labels": labels,
        }

    def _load_json(self):
        """Load data from JSON file."""
        with open(self.data_dir, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict) and "fb_ids" in raw_data:
            self.data = raw_data
        elif isinstance(raw_data, list):
            self.data = self._convert_list_to_dict(raw_data)
        else:
            self.data = self._create_empty_data()

    def _convert_list_to_dict(self, data_list: List[Dict]) -> Dict[str, Any]:
        """Convert list format to dict format."""
        return {
            "N": len(data_list),
            "fb_ids": [item.get("fb_id", str(i)) for i, item in enumerate(data_list)],
            "items_1": [item.get("item_1", "") for item in data_list],
            "texts": [item.get("text", "") for item in data_list],
            "labels": [item.get("label", "Center") for item in data_list],
        }

    def _create_empty_data(self) -> Dict[str, Any]:
        """Create empty data structure."""
        return {
            "N": 0,
            "fb_ids": [],
            "items_1": [],
            "texts": [],
            "labels": [],
        }

    def process_data(self):
        """Process data and apply limit if specified."""
        # Apply limit if specified
        if self.limit and self.limit < self.data.get("N", 0):
            self.data = self.sample(self.limit)

        print(f"Processed {self.data.get('N', 0)} entries from FlipBias dataset.")

    def _create_sample_data(self):
        """Create minimal sample data for testing."""
        self.data = {
            "N": 6,
            "fb_ids": ["1", "2", "3", "4", "5", "6"],
            "items_1": ["1", "1", "1", "2", "2", "2"],
            "texts": [
                "The progressive policies will help reduce inequality and support working families.",
                "Both sides have valid points on this issue and we should consider all perspectives.",
                "Traditional values and free market principles are essential for economic growth.",
                "We need more government intervention to protect workers and the environment.",
                "The facts speak for themselves without political spin or bias.",
                "Lower taxes and less regulation will create more jobs and prosperity.",
            ],
            "labels": ["Left", "Center", "Right", "Left", "Center", "Right"],
        }
        print("Created sample data for testing.")

    def sample(self, n: int) -> Dict[str, Any]:
        """
        Sample n items from the dataset.

        Args:
            n: Number of samples

        Returns:
            Sampled data dict
        """
        total = self.data.get("N", 0)

        if n >= total:
            return self.data

        indices = random.sample(range(total), n)
        indices.sort()

        return {
            "N": n,
            "fb_ids": [self.data["fb_ids"][i] for i in indices],
            "items_1": [self.data["items_1"][i] for i in indices],
            "texts": [self.data["texts"][i] for i in indices],
            "labels": [self.data["labels"][i] for i in indices],
        }

    def get_data(self) -> Dict[str, Any]:
        """Get the full dataset."""
        return self.data

    def get_texts(self) -> List[str]:
        """Get list of all texts."""
        return self.data.get("texts", [])

    def get_labels(self) -> List[str]:
        """Get list of all labels."""
        return self.data.get("labels", [])

    def get_by_label(self, label: str) -> Dict[str, Any]:
        """
        Get items filtered by political label.

        Args:
            label: Label to filter by (Left/Center/Right)

        Returns:
            Filtered data dict
        """
        indices = [
            i for i, l in enumerate(self.data.get("labels", []))
            if l == label
        ]

        return {
            "N": len(indices),
            "fb_ids": [self.data["fb_ids"][i] for i in indices],
            "items_1": [self.data["items_1"][i] for i in indices],
            "texts": [self.data["texts"][i] for i in indices],
            "labels": [self.data["labels"][i] for i in indices],
        }

    def get_label_counts(self) -> Dict[str, int]:
        """Get count of items per label."""
        counts = {"Left": 0, "Center": 0, "Right": 0}
        for label in self.data.get("labels", []):
            if label in counts:
                counts[label] += 1
        return counts

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            "name": "FlipBias",
            "total_samples": self.data.get("N", 0),
            "label_counts": self.get_label_counts(),
            "evaluator": self.evaluator_name,
        }
