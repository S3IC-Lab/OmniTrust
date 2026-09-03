import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.jailbreak.AttackBase import AttackInterface
from attack.jailbreak.utils.dra.dra_utils import generation, jailbreak_check, log_red, log_blue
from attack.jailbreak.utils.dra.harmfulbench_utils import predict


class DRAModel(AttackInterface):
    """
    DRA (Diversion and Reconstruction Attack) 越狱攻击模型
    使用 OmniTrust 统一模型管理系统
    """
    
    def __init__(self, args, model_caller):
        """
        初始化 DRA 攻击模型
        
        Args:
            args: 攻击参数（包含 em_t, verbose 等）
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
        # 攻击参数
        attack_kwargs = {"toxic_trunc": 0.5, "benign_trunc": 0.5}
        
        # 每个行为攻击10次
        for step in range(10):
            # 生成变异的提示
            self.harmful_prompt = generation(
                prompt, 
                self.args.model_name, 
                verbose=self.args.verbose, 
                **attack_kwargs
            )
            
            if self.args.verbose:
                print(self.harmful_prompt)
                print('*' * 20)
            
            # 使用统一的模型调用器生成响应
            messages = [{"role": "user", "content": self.harmful_prompt}]
            result = self.model_caller.generate(messages)
            
            # 处理返回结果
            if isinstance(result, list) and len(result) > 0:
                self.response = result[0].get('generation', '').strip()
            else:
                self.response = result.get('generation', '').strip()
            
            if self.args.verbose:
                print(self.response)
                print('*' * 20)
            
            # 检查是否成功越狱
            jailbreak_check_GCG, jailbreak_check_AutoDAN, em = jailbreak_check(
                prompt, 
                self.response, 
                self.args.em_t
            )
            
            if not jailbreak_check_GCG:
                log_red('[!] Failed GCG check')
                attack_kwargs['toxic_trunc'] -= 0.1
                attack_kwargs['toxic_trunc'] = max(attack_kwargs['toxic_trunc'], 0.001)
                continue
            else:
                if not em:
                    log_red('[!] Failed em check')
                    attack_kwargs['benign_trunc'] += 0.1
                    attack_kwargs['benign_trunc'] = min(attack_kwargs['benign_trunc'], 0.999)
                    continue
            
            # 第二层判断：使用 harmbench_judge
            judger_result = predict(prompt, self.response)
            
            if not judger_result:
                log_red('[!] harmful judger check failed.')
            
            if jailbreak_check_GCG and em and judger_result:
                log_blue('[$] Passed GCG and em check')
                log_blue(f'[$] Total step: {step+1}')
                break
        
        return self.harmful_prompt, self.response

