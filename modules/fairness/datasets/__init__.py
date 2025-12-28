# OmniTrust Fairness Module - Datasets
# Copyright (c) 2025 OmniTrust Team

"""
Dataset loaders for fairness evaluation.

NOTE: This module re-exports from the centralized data framework.
For new datasets, please add them to /data/data_registry/ instead.
"""

# Re-export from centralized data module for backward compatibility
from data import CognitiveBiasDataset, DatasetRegistry, BaseDataset

__all__ = [
    "BaseDataset",
    "DatasetRegistry",
    "CognitiveBiasDataset",
]
