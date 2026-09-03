import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.jailbreak.AttackBase import AttackInterface
from attack.jailbreak.utils.multilingual.mutate import multilingual_mutate


class MultilingualModel(AttackInterface):
    """
    多语言越狱攻击模型
    使用 OmniTrust 统一模型管理系统
    """
    
    def __init__(self, args, model_caller):
        """
        初始化多语言攻击模型
        
        Args:
            args: 攻击参数（包含 mutation_language 等）
            model_caller: 统一的模型调用器（从 get_supported_llm 获取）
        """
        super().__init__(args, model_caller)
        
        self.harmful_prompt = None
        self.response = None

    def get_harmful_prompt_and_response(self, prompt):
        """
        获取变异后的有害提示和模型响应
        
        Args:
            prompt: 原始提示
            
        Returns:
            tuple: (变异后的提示, 模型响应)
        """
        # 应用多语言变异
        self.harmful_prompt = multilingual_mutate(prompt, self.args.mutation_language)
        
        # 使用统一的模型调用器生成响应
        # 构造消息格式（适配 OpenAI Chat 格式）
        messages = [{"role": "user", "content": self.harmful_prompt}]
        
        # 调用模型生成
        result = self.model_caller.generate(messages)
        
        # 处理返回结果（兼容不同返回格式）
        if isinstance(result, list) and len(result) > 0:
            self.response = result[0].get('generation', '').strip()
        else:
            self.response = result.get('generation', '').strip()

        return self.harmful_prompt, self.response