# data/c4.py

import json
import statistics

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from .base_loader import BaseDataset
from .registry import DatasetRegistry

@DatasetRegistry.register("c4")
class C4Dataset(BaseDataset):
    """
    Dataset class for C4 dataset.
    """
    # Initialize the C4 dataset.
    def __init__(self, data_dir: str, **kwargs):
        super(C4Dataset, self).__init__(data_dir, **kwargs)
        self.data_dir = data_dir
        self.load_data()

    def process_data(self):
        pass

    def load_data(self):
        """Load data from the C4 dataset file."""
        with open(self.data_dir, 'r') as f:
           lines = f.readlines()
        for line in lines[:200]:
            item = json.loads(line)
            self.prompts.append(item['prompt'])
            self.natural_texts.append(item['natural_text'])