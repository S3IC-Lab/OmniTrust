# data/cnn_dailymail.py

import json
import statistics

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from .base_loader import BaseDataset
from .registry import DatasetRegistry

@DatasetRegistry.register("cnn_dailymail")
class CNN_DailyMailDataset(BaseDataset):
    """
    Dataset class for CNN/DailyMail dataset.
    """
    # Initialize the CNN_DailyMail dataset.
    def __init__(self, data_dir: str, max_samples: int = 200, global_prompt="Please summarize the following article: ", **kwargs) -> None:
        super(CNN_DailyMailDataset, self).__init__(data_dir, **kwargs)
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.global_prompt = global_prompt
        self.load_data()

    def process_data(self):
        pass
    
    def load_data(self):
        """Load data from the CNN/DailyMail dataset file."""
        with open(self.data_dir, 'r') as f:
            lines = f.readlines()
        for line in lines[:self.max_samples]:
            item = json.loads(line)
            self.prompts.append(f"{self.global_prompt}{item['article']}")
            self.references.append(item['highlights'])
