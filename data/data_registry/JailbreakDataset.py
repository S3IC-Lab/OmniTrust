"""
Jailbreak Dataset Loader

用于加载 Jailbreak 攻击的数据集
数据格式: JSON 文件，包含 prompt 字段
"""

import os
import json
from .base_loader import BaseDataset
from .registry import DatasetRegistry


@DatasetRegistry.register("JailbreakDataset")
class JailbreakDataset(BaseDataset):
    """
    Jailbreak 攻击数据集加载器
    
    支持的数据格式:
    1. JSON 列表格式: [{"prompt": "..."}, {"prompt": "..."}]
    2. JSONL 格式: 每行一个 JSON 对象
    """
    
    def __init__(self, data_dir: str, **kwargs):
        """
        初始化数据集
        
        Args:
            data_dir: 数据文件路径（支持 .json 和 .jsonl）
            **kwargs: 其他参数
        """
        super(JailbreakDataset, self).__init__(data_dir, **kwargs)
        self.load_data()
        self.process_data()
    
    def load_data(self):
        """
        从文件加载数据
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据文件不存在: {self.data_dir}")
        
        file_ext = os.path.splitext(self.data_dir)[1].lower()
        
        if file_ext == '.json':
            # 加载 JSON 格式
            with open(self.data_dir, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
                
        elif file_ext == '.jsonl':
            # 加载 JSONL 格式
            with open(self.data_dir, 'r', encoding='utf-8') as f:
                self.data = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"警告: 跳过无效的 JSON 行: {e}")
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .json 和 .jsonl")
        
        print(f"✓ 成功加载 {len(self.data)} 条数据")
    
    def process_data(self):
        """
        处理数据，提取 prompt 字段
        """
        # 验证数据格式
        for idx, entry in enumerate(self.data):
            if not isinstance(entry, dict):
                raise ValueError(f"数据第 {idx} 条不是字典格式")
            
            if 'prompt' not in entry:
                raise ValueError(f"数据第 {idx} 条缺少 'prompt' 字段")
            
            # 提取 prompt
            self.prompts.append(entry['prompt'])
        
        print(f"✓ 成功处理 {len(self.prompts)} 条 prompt")
    
    def get_prompt_by_index(self, index: int) -> str:
        """
        根据索引获取 prompt
        
        Args:
            index: 数据索引
            
        Returns:
            prompt 字符串
        """
        if 0 <= index < len(self.data):
            return self.data[index].get('prompt', '')
        else:
            raise IndexError(f"索引 {index} 超出范围 [0, {len(self.data)})")
    
    def get_all_prompts(self):
        """
        获取所有 prompts
        
        Returns:
            prompt 列表
        """
        return self.prompts
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, index):
        """支持索引访问"""
        return self.data[index]

