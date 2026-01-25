# vlm_qa pipeline for hallusionbench, vh-test-oeq, vh-test-ynq
import sys
import os

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(current_file_path))

project_root = os.path.join(os.path.dirname(current_file_path), "../../../../")

# 使用 OmniTrust 的模型接口
from model.lvlm.Qwen2VL import Qwen2VL
from model.lvlm.LLaVA import LLaVA
from model.lvlm.LLaVANeXT import LLaVANeXT

from modules.hallucination.vlm.pipeline import vlm_pipeline
from PIL import Image
import torch
from tqdm import tqdm
import json
from datetime import datetime
from modules.hallucination.vlm.vlm_qa.vlm_utils import *

class vlm_qa_pipeline(vlm_pipeline):
    def __init__(self, args):
        # 从 args 获取参数
        method = getattr(args, 'method', None)
        if method not in ['hallusionbench', 'vh-test-oeq', 'vh-test-ynq']:
            raise ValueError(f"Unsupported method: {method}. Must be one of: hallusionbench, vh-test-oeq, vh-test-ynq")
        
        model_name = getattr(args, 'model_name', None) or getattr(args, 'lvlm', 'llava-1.5-13b-hf')
        data_path_dir = getattr(args, 'data_path_dir', None)
        
        # 调用父类初始化
        super().__init__(domain=method, data_path_dir=data_path_dir)
        
        self.args = args
        self.foundation_model = model_name
        self.domain = method

        self.model_output_entry = "model_prediction"
        self.model_correctness_entry = "gpt4v_output_gpt_check"

        # 设置结果保存路径
        result_dir = os.path.join(project_root, "modules/hallucination/vlm/vlm_qa/exp")
        os.makedirs(result_dir, exist_ok=True)
        self.save_result_path = os.path.join(result_dir, f"{method}_result.json")
        self.save_response_path = os.path.join(result_dir, f"{method}_output.json")

        # 如果结果文件存在，删除它（防止冲突）
        if os.path.exists(self.save_result_path):
            os.remove(self.save_result_path)
            print("original result file deleted")
        else:
            print("no original result file found")
        
        # 初始化模型
        self.lvlm = self.obtain_lvlm()

    def obtain_lvlm(self):
        """获取视觉语言模型实例"""
        # 优先使用 args.model_path_dir（如果提供）
        model_path_dir = getattr(self.args, 'model_path_dir', None)
        
        # 根据模型名称确定模型类型
        model_name = self.foundation_model.lower()
        
        # 如果模型名称是路径，尝试从路径推断
        if os.path.exists(self.foundation_model):
            # 从路径推断模型类型
            path_lower = self.foundation_model.lower()
            if 'qwen' in path_lower or 'qwen2-vl' in path_lower:
                # 从路径推断 Qwen2-VL 版本
                if '72b' in path_lower:
                    version = 'Qwen2-VL-72B-Instruct'
                elif '7b' in path_lower:
                    version = 'Qwen2-VL-7B-Instruct'
                elif '2b' in path_lower:
                    version = 'Qwen2-VL-2B-Instruct'
                else:
                    version = 'Qwen2-VL-2B-Instruct'  # 默认
                # 如果没有提供 model_path_dir，从路径中提取目录
                if model_path_dir is None:
                    model_path_dir = os.path.dirname(self.foundation_model)
                return Qwen2VL(version, model_path_dir)
            elif 'llava' in path_lower:
                if 'v1.6' in path_lower or 'next' in path_lower:
                    if 'vicuna' in path_lower:
                        return LLaVANeXT('llava-v1.6-vicuna-13b-hf', model_path_dir)
                    elif 'mistral' in path_lower:
                        return LLaVANeXT('llava-v1.6-mistral-7b-hf', model_path_dir)
                    else:
                        return LLaVANeXT('llava-v1.6-vicuna-13b-hf', model_path_dir)
                else:
                    if '13b' in path_lower:
                        return LLaVA('llava-1.5-13b-hf', model_path_dir)
                    elif '7b' in path_lower:
                        return LLaVA('llava-1.5-7b-hf', model_path_dir)
                    else:
                        return LLaVA('llava-1.5-13b-hf', model_path_dir)
        
        # 根据模型名称字符串判断（非路径情况）
        if 'qwen' in model_name or 'qwen2-vl' in model_name:
            # 确定 Qwen2-VL 版本
            if '72b' in model_name:
                version = 'Qwen2-VL-72B-Instruct'
            elif '7b' in model_name:
                version = 'Qwen2-VL-7B-Instruct'
            elif '2b' in model_name:
                version = 'Qwen2-VL-2B-Instruct'
            else:
                version = 'Qwen2-VL-2B-Instruct'  # 默认
            
            # 使用提供的 model_path_dir（如果存在），否则为 None（使用 HuggingFace）
            return Qwen2VL(version, model_path_dir)
        
        elif 'llava' in model_name:
            # 确定 LLaVA 版本
            if 'v1.6' in model_name:
                if 'vicuna' in model_name:
                    version = 'llava-v1.6-vicuna-13b-hf'
                elif 'mistral' in model_name:
                    version = 'llava-v1.6-mistral-7b-hf'
                else:
                    version = 'llava-v1.6-vicuna-13b-hf'
                return LLaVANeXT(version, model_path_dir)
            else:
                # LLaVA 1.5
                if '13b' in model_name:
                    version = 'llava-1.5-13b-hf'
                elif '7b' in model_name:
                    version = 'llava-1.5-7b-hf'
                else:
                    version = 'llava-1.5-13b-hf'
                return LLaVA(version, model_path_dir)
        
        # 默认使用 LLaVA
        return LLaVA('llava-1.5-13b-hf', model_path_dir=model_path_dir)

    def get_response(self):
        """获取模型响应"""
        dataList = []
        for item in tqdm(self.data):
            line = item.copy()
            image_path = item['filename']
            question = item['question']
            
            # 处理图像路径
            if not os.path.isabs(image_path):
                image_path = os.path.join(project_root, image_path)
            
            # 加载图像
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path

            # 使用统一的模型接口生成响应
            response = self.lvlm.generate(image, question, temp=0.2)
            line[self.model_output_entry] = response
            dataList.append(line)
            
            # 保存中间结果
            os.makedirs(os.path.dirname(self.save_response_path), exist_ok=True)
            with open(self.save_response_path, 'w', encoding='utf-8') as file:
                json.dump(dataList, file, ensure_ascii=False, indent=2)

    def check(self):
        """检查模型响应"""
        data_response_List = []
        with open(self.save_response_path, 'r', encoding='utf-8') as json_file:
            response_data = json.load(json_file)

        for data in tqdm(response_data):
            data_response_List.append(data)
        
        data_response_List = evaluate_by_chatgpt(data_response_List, self.model_output_entry, self.model_correctness_entry, load_json=True, save_json_path=self.save_result_path)
        data_result_List = check_same_by_chatgpt(data_response_List, self.model_output_entry, load_json=True, save_json_path=self.save_result_path)
        
        self.data_result_list = data_result_List
        return data_result_List
    
    def evaluation(self):
        """评估结果"""
        self.check()
        print("##### GPT Evaluate #####")
        data = assign_correctness(self.data_result_list, correctness_entry=self.model_correctness_entry)
        all_data = get_eval_all(self.domain, data, self.model_correctness_entry)

        overall_accuracy = round(100 * all_data["correct"]/all_data["total"], 4) if all_data["total"] > 0 else 0
        
        table1 = [["per question", "Total"],
                ["Overall", overall_accuracy]]
        tab1 = PrettyTable(table1[0])
        tab1.add_rows(table1[1:])
        print(tab1)
        
        # 保存评估结果到文件
        eval_result = {
            "domain": self.domain,
            "model": self.foundation_model,
            "overall_accuracy": overall_accuracy,
            "total": all_data["total"],
            "correct": all_data["correct"],
            "statistics": {
                "LH": all_data.get("LH", 0),
                "VI": all_data.get("VI", 0),
                "Mix": all_data.get("Mix", 0)
            },
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存评估结果文件
        eval_result_path = os.path.join(
            os.path.dirname(self.save_result_path), 
            f"{self.domain}_eval_result.json"
        )
        with open(eval_result_path, 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        
        print(f"Evaluation result saved to: {eval_result_path}")
        
        return tab1

    def run(self):
        """运行完整的检测流程"""
        # 获取模型响应
        self.get_response()
        # 评估结果
        result = self.evaluation()
        return result

