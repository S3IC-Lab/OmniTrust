# multi-dimension benchmark for vlm hallusination
import sys
import os

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(current_file_path))

project_root = os.path.join(os.path.dirname(current_file_path), "../../../../")

# 使用 OmniTrust 的模型接口
from model.lvlm.Qwen2VL import Qwen2VL
from model.lvlm.LLaVA import LLaVA
from model.lvlm.LLaVANeXT import LLaVANeXT
from datetime import datetime

from modules.hallucination.vlm.pipeline import vlm_pipeline
from PIL import Image
import json
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from tqdm import tqdm
import warnings
import argparse

nlp = spacy.load("en_core_web_lg")
warnings.filterwarnings("ignore", category=UserWarning)

# 模型映射
LVLM_MAP = {
    'Qwen2-VL-72B-Instruct': Qwen2VL,
    'Qwen2-VL-7B-Instruct': Qwen2VL,
    'Qwen2-VL-2B-Instruct': Qwen2VL,
    'llava-v1.6-vicuna-13b-hf': LLaVANeXT,
    'llava-v1.6-mistral-7b-hf': LLaVANeXT,
    'llava-1.5-13b-hf': LLaVA,
    'llava-1.5-7b-hf': LLaVA,
    'llava-v1.5-13b': LLaVA,  # 支持不带 -hf 后缀的版本
}

class vlm_autodetect_pipeline(vlm_pipeline):
    def __init__(self, args):
        # 从 args 获取参数
        model_name = getattr(args, 'model_name', None) or getattr(args, 'lvlm', 'llava-1.5-13b-hf')
        evaluation_type = getattr(args, 'autodetect_type', 'd')
        
        # 调用父类初始化
        super().__init__(domain="auto-detect", data_path_dir=args.data_path_dir)
        
        self.args = args
        self.foundation_model = model_name
        self.evaluation_type = evaluation_type
        
        # 设置默认路径
        self.word_association = os.path.join(project_root, "modules/hallucination/vlm/vlm_autodetect/utils/relation.json")
        self.safe_words = os.path.join(project_root, "modules/hallucination/vlm/vlm_autodetect/utils/safe_words.txt")
        self.annotation = os.path.join(project_root, "modules/hallucination/vlm/vlm_autodetect/utils/annotations.json")
        self.metrics = os.path.join(project_root, "modules/hallucination/vlm/vlm_autodetect/utils/metrics.txt")
        self.similarity_score = 0.8
        
        # 允许通过 args 覆盖路径（只有当值不为 None 时才覆盖）
        if hasattr(args, 'word_association') and args.word_association is not None:
            self.word_association = args.word_association
        if hasattr(args, 'safe_words') and args.safe_words is not None:
            self.safe_words = args.safe_words
        if hasattr(args, 'annotation') and args.annotation is not None:
            self.annotation = args.annotation
        if hasattr(args, 'metrics') and args.metrics is not None:
            self.metrics = args.metrics
        if hasattr(args, 'similarity_score') and args.similarity_score is not None:
            self.similarity_score = args.similarity_score
        
        self.root_dir = "."
        self.load_json = True
        self.model_output_entry = "response"
        
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
                # 如果提供了 model_path_dir，优先使用它
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
        

    def check_synonyms_word(self, word1, word2, similarity_score):
        token1 = nlp(word1)
        token2 = nlp(word2)
        similarity = token1.similarity(token2)
        return similarity > similarity_score

    def extract_nouns(self, text):
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
        return nouns
    
    def get_metric(self):
        metrics = {}
        if os.path.exists(self.metrics):
            with open(self.metrics, "r") as file:
                lines = file.readlines()

            for line in lines:
                parts = line.strip().split('=')
                if len(parts) == 2:
                    variable_name = parts[0].strip()
                    variable_value = eval(parts[1].strip())
                    metrics[variable_name] = variable_value
        else:
            # 默认指标
            metrics = {
                'chair_score': 0, 'chair_num': 0,
                'safe_cover_score': 0, 'safe_cover_num': 0,
                'hallu_cover_score': 0, 'hallu_cover_num': 0,
                'non_hallu_score': 0, 'non_hallu_num': 0,
                'qa_correct_score': 0, 'qa_correct_num': 0,
                'qa_no_score': 0, 'qa_no_num': 0,
                'qa_ans_no_score': 0, 'qa_ans_no_num': 0,
                'as_qa_correct_score': 0, 'as_qa_correct_num': 0,
                'as_qa_no_score': 0, 'as_qa_no_num': 0,
                'as_qa_ans_no_score': 0, 'as_qa_ans_no_num': 0,
                'an_qa_correct_score': 0, 'an_qa_correct_num': 0,
                'an_qa_no_score': 0, 'an_qa_no_num': 0,
                'an_qa_ans_no_score': 0, 'an_qa_ans_no_num': 0,
                'aa_qa_correct_score': 0, 'aa_qa_correct_num': 0,
                'aa_qa_no_score': 0, 'aa_qa_no_num': 0,
                'aa_qa_ans_no_score': 0, 'aa_qa_ans_no_num': 0,
                'ha_qa_correct_score': 0, 'ha_qa_correct_num': 0,
                'ha_qa_no_score': 0, 'ha_qa_no_num': 0,
                'ha_qa_ans_no_score': 0, 'ha_qa_ans_no_num': 0,
                'asso_qa_correct_score': 0, 'asso_qa_correct_num': 0,
                'asso_qa_no_score': 0, 'asso_qa_no_num': 0,
                'asso_qa_ans_no_score': 0, 'asso_qa_ans_no_num': 0,
            }
                
        return metrics

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
    
    def evaluation(self):
        """评估结果"""
        metrics = self.get_metric()
        
        if not os.path.exists(self.word_association) or not os.path.exists(self.annotation):
            print(f"Warning: Required data files not found. Please check paths:")
            print(f"  word_association: {self.word_association}")
            print(f"  annotation: {self.annotation}")
            return {}
        
        association = json.load(open(self.word_association, 'r', encoding='utf-8'))
        hallucination_words = []
        for word1 in association.keys():
            hallucination_words.append(word1)
            for word2 in association[word1]:
                hallucination_words.append(word2)
                
        global_safe_words = []
        if os.path.exists(self.safe_words):
            with open(self.safe_words, 'r', encoding='utf-8') as safe_file:
                for line in safe_file:
                    line = line.split('\n')[0]
                    global_safe_words.append(line)

        dimension = {'g': False,'de': False, 'da': False, 'dr': False}
        if self.evaluation_type == 'a':
            for key in dimension.keys():
                dimension[key] = True
        elif self.evaluation_type == 'g':
            dimension['g'] = True
        elif self.evaluation_type == 'd':
            dimension['de'] = True
            dimension['da'] = True
            dimension['dr'] = True
        else:
            dimension[self.evaluation_type] = True
        
        if not os.path.exists(self.save_response_path):
            print(f"Error: Response file not found: {self.save_response_path}")
            print("Please run get_response() first.")
            return {}
        
        inference_data = json.load(open(self.save_response_path, 'r', encoding='utf-8'))
        ground_truth = json.load(open(self.annotation, 'r', encoding='utf-8'))

        result = {}

        for i in tqdm(range(len(inference_data))):
            id = inference_data[i]['id']
            
            if ground_truth[id-1]['type'] == 'generative':
                nouns = self.extract_nouns(inference_data[i]['response'])
                after_process_nouns = []
                for noun in nouns:
                    if noun in hallucination_words:
                        after_process_nouns.append(noun)
                
                safe_words = []
                safe_list = []
                for idx, word in enumerate(ground_truth[id-1]['truth']):
                    safe_words += association[word]
                    safe_list += [idx] * len(association[word])
                    
                ha_words = []
                ha_list = []
                for idx, word in enumerate(ground_truth[id-1]['hallu']):
                    ha_words += association[word]
                    ha_list += [idx] * len(association[word])
                
                safe_words += ground_truth[id-1]['truth']
                safe_len = len(ground_truth[id-1]['truth'])
                safe_list += [0] * safe_len
                safe_flag_list = [0] * len(after_process_nouns)
                
                ha_words += ground_truth[id-1]['hallu']
                ha_len = len(ground_truth[id-1]['hallu'])
                ha_list += [0] * ha_len
                
                for idx, noun in enumerate(after_process_nouns):
                    if noun in global_safe_words:
                        continue
                    
                    if noun in safe_words:
                        for j in range(len(safe_words)):
                            if noun == safe_words[j]:
                                if j < (len(safe_list) - safe_len):
                                    safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                                else:
                                    safe_list[j] = 1
                                break
                        continue
                    
                    if noun in ha_words:
                        for j in range(len(ha_words)):
                            if noun == ha_words[j]:
                                if j < (len(ha_list) - ha_len):
                                    ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                                else:
                                    ha_list[j] = 1
                                break
                    
                    for j, check_word in enumerate(ha_words):
                        if self.check_synonyms_word(noun, check_word, self.similarity_score):
                            if j < (len(ha_list) - ha_len):
                                    ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                            else:
                                ha_list[j] = 1
                            break
                    
                    flag = False
                    for j, check_word in enumerate(safe_words):
                        if self.check_synonyms_word(noun, check_word, self.similarity_score):
                            flag = True
                            if j < (len(safe_list) - safe_len):
                                    safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                            else:
                                safe_list[j] = 1
                            break
                    if flag == True:
                        continue
                
                    safe_flag_list[idx] = 1

                metrics['chair_score'] += sum(safe_flag_list)
                metrics['chair_num'] += len(safe_flag_list)
                metrics['safe_cover_score'] += sum(safe_list[-safe_len:])
                metrics['safe_cover_num'] += len(safe_list[-safe_len:])
                metrics['hallu_cover_score'] += sum(ha_list[-ha_len:])
                metrics['hallu_cover_num'] += len(ha_list[-ha_len:])
                if sum(safe_flag_list) == 0:
                    metrics['non_hallu_score'] += 1
                metrics['non_hallu_num'] += 1
            
            else:
                metrics['qa_correct_num'] += 1
                if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                    metrics['as_qa_correct_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                    metrics['an_qa_correct_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                    metrics['aa_qa_correct_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                    metrics['ha_qa_correct_num'] += 1
                else:
                    metrics['asso_qa_correct_num'] += 1
                
                truth = ground_truth[id-1]['truth']
                response = inference_data[i]['response'].split(',')[0] # only split 'Yes' or 'No'
                if truth == 'yes':
                    if response == 'Yes':
                        metrics['qa_correct_score'] += 1
                        if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                            metrics['as_qa_correct_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                            metrics['an_qa_correct_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                            metrics['aa_qa_correct_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                            metrics['ha_qa_correct_score'] += 1
                        else:
                            metrics['asso_qa_correct_score'] += 1
                else:
                    metrics['qa_no_num'] += 1
                    if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                        metrics['as_qa_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                        metrics['an_qa_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                        metrics['aa_qa_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                        metrics['ha_qa_no_num'] += 1
                    else:
                        metrics['asso_qa_no_num'] += 1
                    
                    if response == 'No':
                        metrics['qa_correct_score'] += 1
                        metrics['qa_no_score'] += 1
                        if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                            metrics['as_qa_correct_score'] += 1
                            metrics['as_qa_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                            metrics['an_qa_correct_score'] += 1
                            metrics['an_qa_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                            metrics['aa_qa_correct_score'] += 1
                            metrics['aa_qa_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                            metrics['ha_qa_correct_score'] += 1
                            metrics['ha_qa_no_score'] += 1
                        else:
                            metrics['asso_qa_correct_score'] += 1
                            metrics['asso_qa_no_score'] += 1
                
                if response == 'No':
                    metrics['qa_ans_no_num'] += 1
                    if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                        metrics['as_qa_ans_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                        metrics['an_qa_ans_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                        metrics['aa_qa_ans_no_num'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                        metrics['ha_qa_ans_no_num'] += 1
                    else:
                        metrics['asso_qa_ans_no_num'] += 1
                    if truth == 'no':
                        metrics['qa_ans_no_score'] += 1
                        if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                            metrics['as_qa_ans_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                            metrics['an_qa_ans_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                            metrics['aa_qa_ans_no_score'] += 1
                        elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                            metrics['ha_qa_ans_no_score'] += 1
                        else:
                            metrics['asso_qa_ans_no_score'] += 1

        if dimension['g']:
            CHAIR = round(metrics['chair_score'] / metrics['chair_num'] * 100, 1) if metrics['chair_num'] > 0 else 0
            Cover = round(metrics['safe_cover_score'] / metrics['safe_cover_num'] * 100, 1) if metrics['safe_cover_num'] > 0 else 0
            Ha = round(metrics['hallu_cover_score'] / metrics['hallu_cover_num'] * 100, 1) if metrics['hallu_cover_num'] > 0 else 0
            Ha_p = round(100 - metrics['non_hallu_score'] / metrics['non_hallu_num'] * 100, 1) if metrics['non_hallu_num'] > 0 else 0

            print("Generative Task:")
            print("CHAIR:\t\t", CHAIR)
            print("Cover:\t\t", Cover)
            print("Hal:\t\t", Ha_p)
            print("Cog:\t\t", Ha, "\n")

            result["CHAIR"] = CHAIR
            result["Cover"] = Cover
            result["Hal"] = Ha_p
            result["Cog"] = Ha
        
        
        if dimension['de'] and dimension['da'] and dimension['dr']:
            Accuracy = round(metrics['qa_correct_score'] / metrics['qa_correct_num'] * 100, 1) if metrics['qa_correct_num'] > 0 else 0
            Precision = round(metrics['qa_ans_no_score'] / metrics['qa_ans_no_num'] * 100, 1) if metrics['qa_ans_no_num'] > 0 else 0
            Recall = round(metrics['qa_no_score'] / metrics['qa_no_num'] * 100, 1) if metrics['qa_no_num'] > 0 else 0
            F1 = round(2 * (Precision/100) * (Recall/100) / ((Precision/100) + (Recall/100) + 0.0001) * 100, 1)
            print("Descriminative Task:")
            print("Accuracy:\t", Accuracy)
            print("Precision:\t", Precision)
            print("Recall:\t\t", Recall)
            print("F1:\t\t", F1, "\n")

            result["Accuracy"] = Accuracy
            result["Precision"] = Precision
            result["Recall"] = Recall
            result["F1"] = F1

        return result

    def run(self):
        """运行完整的检测流程"""
        # 获取模型响应
        begin_time_str = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        self.get_response()
        # 评估结果
        result = self.evaluation()
        if not os.path.exists('modules/hallucination/vlm/vlm_autodetect/exp'):
            os.makedirs('modules/hallucination/vlm/vlm_autodetect/exp')
        with open(f'modules/hallucination/vlm/vlm_autodetect/exp/log_{begin_time_str}.json', "w") as f: 
            json.dump(result, f)
        return result
