# data/HumanEval.py

import os
import json
from .base_loader import BaseDataset
from .registry import DatasetRegistry

@DatasetRegistry.register("HumanEval")
class HumanEvalDataset(BaseDataset):
    """
    Dataset class for HumanEval dataset.
    """
    def __init__(self, data_dir: str, **kwargs):
        super(HumanEvalDataset, self).__init__(data_dir, **kwargs)
        self.data_dir = data_dir
        self.load_data()
        self.total_entries  = 0  
    
    def load_data(self):
        """
        Load HumanEval data from a JSONL file.
        """
        # humaneval_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jsonl')]
        # if not humaneval_files:
        #     raise FileNotFoundError("No JSONL file found in the HumanEval dataset directory.")
        
        # humaneval_file = os.path.join(self.data_dir, humaneval_files[0])  # Assume first JSONL file
        # with open(humaneval_file, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         try:
        #             entry = json.loads(line.strip()) 
        #             self.data.append(entry)
        #         except json.JSONDecodeError as e:
        #             raise ValueError(f"Error decoding JSON from {humaneval_file}: {e}")
        
        # self.total_entries  = len(self.data)  
        # print(f"Loaded {self.total_entries } tasks entries from HumanEval dataset.")

        # Load data from the HumanEval dataset file.
        with open(self.data_dir, 'r') as f:
            lines = f.readlines()

        for line in lines[:164]:
            item = json.loads(line)
            # process prompt
            prompt = item['prompt']
            sections = prompt.split(">>>")
            prompt = sections[0]
            if len(sections) > 1:
                prompt += '\"\"\"'

            self.prompts.append(prompt)
            self.references.append({'task': prompt, 'test': item['test'], 'entry_point': item['entry_point']})
    
    def process_data(self):
        """
        Process HumanEval data to extract 'task_id', 'prompt', and 'canonical_solution' fields.
        """
        # processed = []
        # for entry in self.data:  
        #     processed_entry = {
        #         "task_id": entry.get("task_id"),
        #         "prompt": entry.get("prompt"),
        #         "canonical_solution": entry.get("canonical_solution")
        #     }
        #     processed.append(processed_entry)
        # self.data = processed 
        
        for entry in self.data:
            if not all(key in entry for key in ["task_id", "prompt", "canonical_solution"]):
                raise ValueError("Entry is missing required fields.")
        print(f"Processed {len(self.data)} entries from HumanEval dataset.")