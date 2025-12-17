import os
import pyarrow.parquet as pq
import pandas as pd
import json
from data.data_registry.base_loader import BaseDataset
from data.data_registry.registry import DatasetRegistry

@DatasetRegistry.register("wikimia")
class WikiMIADataset(BaseDataset):
    def __init__(self, data_dir: str, datatype, **kwargs):
        super(WikiMIADataset, self).__init__(data_dir, **kwargs)
        self.prompts = []
        self.labels = []
        self.total_entries = 0
        self.data_dir = data_dir
        self.type = datatype
        
        # 添加路径验证
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory '{self.data_dir}' does not exist. Attempting to find alternative path...")
            # 尝试绝对路径
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dataset', 'wikimia', 'data'))
            if os.path.exists(abs_path):
                print(f"Using alternative data directory: {abs_path}")
                self.data_dir = abs_path
        
    def load_data(self):
        """
        Load WikiMIA data by reading relevant files in the data directory.
        This method extracts prompts and membership labels from the files.
        WikiMIA dataset typically contains member/non-member text samples.
        """
        # 添加详细的路径信息
        print(f"Loading WikiMIA data from: {self.data_dir}")
        print(f"Absolute path: {os.path.abspath(self.data_dir)}")
        
        # 确保目录存在
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # 列出目录内容
        files = os.listdir(self.data_dir)
        print(f"Files in directory: {files}")
        
        # 增强的文件匹配逻辑
        file_extensions = ['.parquet', '.json', '.jsonl', '.txt', '.csv']
        loaded_files = []
        
        # 首先尝试精确匹配 datatype
        for filename in files:
            # 检查文件是否与 datatype 相关（不区分大小写）
            if self.type.lower() in filename.lower() and any(filename.endswith(ext) for ext in file_extensions):
                filepath = os.path.join(self.data_dir, filename)
                self._load_file(filepath, filename)
                loaded_files.append(filename)
        
        # 如果未找到匹配文件，尝试加载所有支持格式的文件
        if not loaded_files:
            print(f"No files found matching datatype '{self.type}', loading all supported files...")
            for filename in files:
                if any(filename.endswith(ext) for ext in file_extensions):
                    filepath = os.path.join(self.data_dir, filename)
                    self._load_file(filepath, filename)
                    loaded_files.append(filename)
        
        self.total_entries = len(self.prompts)
        print(f"Loaded {self.total_entries} entries from WikiMIA dataset.")
        print(f"Files processed: {loaded_files}")

        return self.prompts, self.labels

    def _load_file(self, filepath, filename):
        """根据文件扩展名加载文件"""
        if filename.endswith('.parquet'):
            self._load_parquet(filepath)
        elif filename.endswith('.json'):
            self._load_json(filepath)
        elif filename.endswith('.jsonl'):
            self._load_jsonl(filepath)
        elif filename.endswith('.csv'):
            self._load_csv(filepath)
        elif filename.endswith('.txt'):
            self._load_txt(filepath)

    def _load_parquet(self, filepath):
        """Load data from parquet file"""
        print(f"Loading parquet file: {os.path.basename(filepath)}")
        table = pq.read_table(filepath)
        df = table.to_pandas()
        
        # 记录列名
        print(f"Columns in parquet file: {df.columns.tolist()}")
        
        for index, row in df.iterrows():
            # 尝试不同的文本列名
            text_content = None
            for col in ['text', 'content', 'passage', 'document', 'sample', 'input']:
                if col in df.columns and pd.notna(row[col]):
                    text_content = str(row[col])
                    break
            
            # 尝试不同的标签列名
            membership_label = None
            for col in ['label', 'member', 'membership', 'is_member', 'target']:
                if col in df.columns and pd.notna(row[col]):
                    try:
                        membership_label = int(row[col])
                    except ValueError:
                        # 尝试布尔值转换
                        membership_label = 1 if row[col] else 0
                    break
            
            if text_content:
                self.prompts.append(text_content)
                # 如果未找到标签，根据文件名推测
                if membership_label is None:
                    membership_label = 1 if 'member' in os.path.basename(filepath).lower() else 0
                    print(f"Using inferred label {membership_label} for file {os.path.basename(filepath)}")
                self.labels.append(membership_label)

    def _load_json(self, filepath):
        """Load data from JSON file"""
        print(f"Loading JSON file: {os.path.basename(filepath)}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                self._extract_from_dict(item)
        elif isinstance(data, dict):
            # If it's a dict, check if it contains a list of samples
            for key in ['data', 'samples', 'entries', 'texts']:
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        self._extract_from_dict(item)
                    break
            else:
                # If no list found, treat the dict itself as a single sample
                self._extract_from_dict(data)

    def _load_jsonl(self, filepath):
        """Load data from JSONL file"""
        print(f"Loading JSONL file: {os.path.basename(filepath)}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    self._extract_from_dict(item)

    def _load_csv(self, filepath):
        """Load data from CSV file"""
        print(f"Loading CSV file: {os.path.basename(filepath)}")
        df = pd.read_csv(filepath)
        
        for index, row in df.iterrows():
            # Try different possible column names for text content
            text_content = None
            for col in ['text', 'content', 'passage', 'document', 'sample']:
                if col in df.columns and pd.notna(row[col]):
                    text_content = str(row[col])
                    break
            
            # Try different possible column names for membership label
            membership_label = None
            for col in ['label', 'member', 'membership', 'is_member']:
                if col in df.columns and pd.notna(row[col]):
                    membership_label = int(row[col])
                    break
            
            if text_content:
                self.prompts.append(text_content)
                self.labels.append(membership_label if membership_label is not None else 0)

    def _load_txt(self, filepath):
        """Load data from plain text file"""
        print(f"Loading TXT file: {os.path.basename(filepath)}")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Split by double newlines (assuming each sample is separated by double newlines)
        samples = content.split('')
        for sample in samples:
            if sample.strip():
                self.prompts.append(sample.strip())
                # For plain text files, we assume all samples have the same membership status
                # This might need to be adjusted based on the actual WikiMIA format
                default_label = 1 if 'member' in filepath.lower() else 0
                self.labels.append(default_label)

    def _extract_from_dict(self, item):
        """Extract text and label from a dictionary item"""
        if not isinstance(item, dict):
            return
        
        # Try different possible keys for text content
        text_content = None
        for key in ['text', 'content', 'passage', 'document', 'sample', 'input']:
            if key in item and item[key]:
                text_content = str(item[key])
                break
        
        # Try different possible keys for membership label
        membership_label = None
        for key in ['label', 'member', 'membership', 'is_member', 'target']:
            if key in item:
                membership_label = int(item[key])
                break
        
        if text_content:
            self.prompts.append(text_content)
            self.labels.append(membership_label if membership_label is not None else 0)

    def process_data(self):
        """
        Process the loaded data to create a list of dictionaries with prompts and labels.
        For WikiMIA, labels typically indicate membership: 1 for member, 0 for non-member.
        """
        processed = []
        for idx in range(self.total_entries):
            entry = {
                "id": idx + 1,
                "prompt": self.prompts[idx],
                "label": self.labels[idx],  # 1 for member, 0 for non-member
                "membership": "member" if self.labels[idx] == 1 else "non-member"
            }
            processed.append(entry)
        
        self.data = processed
        print(f"Processed {len(self.data)} entries from WikiMIA dataset.")
        print(f"Members: {sum(1 for entry in self.data if entry['label'] == 1)}")
        print(f"Non-members: {sum(1 for entry in self.data if entry['label'] == 0)}")
        
        return self.data

    def get_member_samples(self):
        """Get only the member samples"""
        return [prompt for prompt, label in zip(self.prompts, self.labels) if label == 1]

    def get_non_member_samples(self):
        """Get only the non-member samples"""
        return [prompt for prompt, label in zip(self.prompts, self.labels) if label == 0]

    def get_statistics(self):
        """Get dataset statistics"""
        total = len(self.prompts)
        members = sum(1 for label in self.labels if label == 1)
        non_members = total - members
        
        stats = {
            "total_samples": total,
            "member_samples": members,
            "non_member_samples": non_members,
            "member_ratio": members / total if total > 0 else 0,
            "datatype": self.type,
            "data_dir": self.data_dir
        }
        
        return stats
