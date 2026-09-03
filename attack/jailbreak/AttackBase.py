"""
This module contains the abstract base class for all attack methods.

All specific attack classes should inherit from `AttackInterface` and
implement the `perform_attack` method.
"""

from abc import ABC, abstractmethod
from typing import Any

class AttackInterface(ABC):
    """
    攻击方法抽象基类
    使用 OmniTrust 统一模型管理系统
    """
    
    def __init__(self, args: Any, model_caller: Any):
        """
        初始化攻击接口
        
        Args:
            args: 攻击方法参数
            model_caller: 统一的模型调用器（从 get_supported_llm 获取）
        """
        self.args = args
        self.model_caller = model_caller

    # 获取使用当前方法后的有害prompt和响应
    @abstractmethod
    def get_harmful_prompt_and_response(self, prompt: str) -> Any:
        """
        执行攻击并获取响应
        
        Args:
            prompt: 原始提示
            
        Returns:
            tuple: (变异后的提示, 模型响应)
        """
        pass    