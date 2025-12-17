import os
import pyarrow.parquet as pq
import pandas as pd
from data.data_registry.base_loader import BaseDataset
from data.data_registry.registry import DatasetRegistry


@DatasetRegistry.register("wikitext")
class wikitextDataset(BaseDataset):
    def __init__(self, data_dir: str, datatype, **kwargs):
        super(wikitextDataset, self).__init__(data_dir, **kwargs)
        self.prompts = []
        self.labels = []
        self.total_entries = 0
        self.data_dir = data_dir
        self.type = datatype
    def load_data(self):
        """
        Load data by reading all relevant parquet files in the data directory.
        This method extracts prompts and labels from the parquet files.
        """

        for filename in os.listdir(self.data_dir):
            if filename.startswith(self.type) and filename.endswith('.parquet'):
                print(filename)
                filepath = os.path.join(self.data_dir, filename)
                # 读取 Parquet 文件
                table = pq.read_table(filepath)
                df = table.to_pandas()

                for index, row in df.iterrows():
                    prompt = row.get('text')
                    label = row.get('label')
                    if prompt:
                        self.prompts.append(prompt)
                        self.labels.append(label)

        self.total_entries = len(self.prompts)
        print(f"Loaded {self.total_entries} entries from wikitext dataset.")

        return self.prompts, self.labels

    def process_data(self, ):
        """
        Process the loaded data to create a list of dictionaries with prompts and labels.
        """
        processed = []
        for idx in range(self.total_entries):
            entry = {
                "id": idx + 1,
                "prompt": self.prompts[idx],
                "label": self.labels[idx]
            }
            processed.append(entry)
        self.data = processed
        print(f"Processed {len(self.data)} entries from wikitext dataset.")
