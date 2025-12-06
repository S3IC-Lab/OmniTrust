## Dataset deployment - take the HumanEval dataset as an example

### Step0: Deploy on the premise

Download the dataset file to the `/data/dataset` directory

> For steps to *download* the target dataset, please refer to README files in /data/dataset/dataset_NAME


### Step1: Dataset registration

Create a new python file in the `/data_registry`

For example, `HumanEval.py`

```python
import os
import json
from .base_loader import BaseDataset
from .registry import DatasetRegistry
@DatasetRegistry.register("HumanEval")
```

### Step2: Inheritance class writing

#### initialize

```python
class HumanEvalDataset(BaseDataset):
    """
    Implementation for the HumanEval dataset.
    Each line in the JSONL file is a JSON object with task_id, prompt, and canonical_solution.
    """
    def __init__(self, **kwargs):
        super(HumanEvalDataset, self).__init__(**kwargs)
        self.total_entries  = 0  
```

#### All subclasses that inherit from BaseDataset must implement the 'load_data' and 'process_data' methods

##### load_data

Analyze the data set structure and use the appropriate loading method, for example, the `HumanEval` data set contains `a jsonl file`

Similar format

```json
{
    "task_id": "HumanEval/0", 
    "prompt": "Write a function to add two numbers.", 
    "canonical_solution": "def add(a, b):\n    return a + b"
}
```

```python
def load_data(self):
        """
        Load HumanEval data from a JSONL file in the data directory.
        Each line is a JSON object containing task_id, prompt, and canonical_solution.
        """
        humaneval_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jsonl')]
        if not humaneval_files:
            raise FileNotFoundError("No JSONL file found in the HumanEval dataset directory.")
        
        humaneval_file = os.path.join(self.data_dir, humaneval_files[0])  # Assume first JSONL file
        with open(humaneval_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip()) 
                    self.data.append(entry)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error decoding JSON from {humaneval_file}: {e}")
        
        self.total_entries  = len(self.data)  
        print(f"Loaded {self.total_entries } tasks entries from HumanEval dataset.")
```

#### process_data

```python
    def process_data(self):
        """
        Process HumanEval data to extract 'task_id', 'prompt', and 'canonical_solution' fields.
        """
#        processed = []
#        for entry in self.data:  # Directly iterate over self.data
#            processed_entry = {
#                "task_id": entry.get("task_id"),
#                "prompt": entry.get("prompt"),
#                "canonical_solution": entry.get("canonical_solution")
#            }
#            processed.append(processed_entry)
#        self.data = processed 
        for entry in self.data:
            if not all(key in entry for key in ["task_id", "prompt", "canonical_solution"]):
                raise ValueError("Entry is missing required fields.")
        print(f"Processed {len(self.data)} entries from HumanEval dataset.")
```

### Step3: Add content to `__init__.py`

```python
from .data_registry.HumanEval import HumanEvalDataset
```
