# data/wmt16de_en.py

import json
import statistics

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from .base_loader import BaseDataset
from .registry import DatasetRegistry

@DatasetRegistry.register("wmt16de_en")
class WMT16DE_ENDataset(BaseDataset):
    """
    Dataset class for WMT16 DE-EN dataset.
    """
    # Initialize the WMT16 DE-EN dataset.
    def __init__(self, data_dir: str, **kwargs):
        super(WMT16DE_ENDataset, self).__init__(data_dir, **kwargs)
        self.data_dir = data_dir
        self.load_data()

    def process_data(self):
        pass

    def load_data(self):
        """Load data from the WMT16 DE-EN dataset file."""
        with open(self.data_dir, 'r') as f:
            lines = f.readlines()
        for line in lines[:200]:
            item = json.loads(line)
            self.prompts.append(item['de'])
            self.references.append(item['en'])
    
    