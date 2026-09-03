import os
import sys
import json
import yaml
import argparse

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model.llm import get_supported_llm
from model.model_configs import model_configs
from data.data_registry.registry import DatasetRegistry

def jailbreak_pipeline_black(args):

    # 使用 OmniTrust 统一数据集管理系统载入数据集
    dataset = DatasetRegistry.get_dataset("JailbreakDataset", data_dir=args.data_path)
    data = dataset.get_data()
    
    # 统一载入模型：使用 OmniTrust 模型管理系统
    model_caller = get_supported_llm(yaml.safe_load(open(model_configs.get(args.model_name, args.model_config_path), 'r'))) if not args.use_custom_api else None

    # 载入攻击方法，传递统一的模型调用器
    if args.attack == "Multilingual":
        from attack.jailbreak.Multilingual_attack import MultilingualModel
        attack_model = MultilingualModel(args, model_caller)
    elif args.attack == "DeepInception":  
        from attack.jailbreak.DeepInception_attack import DeepInceptionModel
        attack_model = DeepInceptionModel(args, model_caller)
    elif args.attack == "CodeChameleon":
        from attack.jailbreak.CodeChameleon_attack import CodeChameleonModel
        attack_model = CodeChameleonModel(args, model_caller)
    elif args.attack == "DRA":
        from attack.jailbreak.DRA_attack import DRAModel
        attack_model = DRAModel(args, model_caller)
    # elif args.attack == "CipherChat":
    #     from attack.jailbreak.CipherChat_attack import CipherChatAttackModel
    #     attack_model = CipherChatAttackModel(args, args.model_name)
    # elif args.attack == "Jailbroken":  
    #     from attack.jailbreak.Jailbroken_attack import JailbrokenModel
    #     attack_model = JailbrokenModel(args, args.model_name)
    # elif args.attack == "AutoDAN": 
    #     from attack.jailbreak.AutoDAN_attack import AutoDANModel
    #     attack_model = AutoDANModel(args, args.model_name)
    # elif args.attack == "ICA": 
    #     from attack.jailbreak.ICA_attack import ICAModel
    #     attack_model = ICAModel(args, args.model_name)
    # elif args.attack == "ArtPrompt": 
    #     from attack.jailbreak.ArtPrompt_attack import ArtPromptAttackModel
    #     attack_model = ArtPromptAttackModel(args, args.model_name)
    # elif args.attack == "Laa":  
    #     from attack.jailbreak.Laa_attack import LaaModel
    #     attack_model = LaaModel(args, args.model_name)
    # elif args.attack == "ReNeLLM":
    #     from attack.jailbreak.ReNeLLM_attack import ReNeLLMModel
    #     attack_model = ReNeLLMModel(args, args.model_name)
    # elif args.attack == "Tap":
    #     from attack.jailbreak.Tap_attack import TapModel
    #     attack_model = TapModel(args, args.model_name)
    # elif args.attack == "MAC":
    #     from attack.jailbreak.MAC_attack import GCGModel
    #     attack_model = GCGModel(args, args.model_name)
    # elif args.attack == "PiF":
    #     from attack.jailbreak.PiF_attack import PiFModel
    #     attack_model = PiFModel(args, args.model_name)
    else:
        raise ValueError(f"暂不支持的攻击方法: {args.attack}，当前支持: Multilingual, DeepInception, CodeChameleon, DRA")

    # 创建输出目录
    output_dir = os.path.join(project_root, "examples/jailbreak_result")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"jailbreak_{args.attack}_{args.model_name.replace('/', '_')}.json")

    # 如果文件存在，先读取已有内容
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                dict_list = json.load(f)
            except:
                dict_list = []
    else:
        dict_list = []

    for idx, item in enumerate(data):
        # 检查是否已经处理过这个idx（支持断点续传）
        if any(d['idx'] == idx for d in dict_list):
            print(f"Skipping idx {idx}, already processed")
            continue
            
        prompt = item['prompt']
        harmful_prompt, response = attack_model.get_harmful_prompt_and_response(prompt)
        
        result = {
            "idx": idx,
            "original_prompt": prompt,
            "harmful_prompt": harmful_prompt,
            "response": response
        }
        dict_list.append(result)
        
        print("Original Prompt: ", prompt)
        print("Harmful Prompt: ", harmful_prompt)
        print("Response: ", response)
        print("\n\n")
        
        # 立即保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dict_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OmniTrust Jailbreak Attack Pipeline")

    ########################################
    ######## 模型载入参数 (OmniTrust 统一模型管理) #####
    ########################################
    parser.add_argument("--model_name", type=str, default="chatgpt", 
                        help="模型名称，对应 model_configs.py 中的 key (如 'chatgpt', 'gpt4', 'Qwen2-7B-Instruct')")
    parser.add_argument("--model_config_path", type=str, default=None,
                        help="可选：直接指定模型配置文件路径（覆盖 model_configs）")
    parser.add_argument("--use_custom_api", action="store_true",
                        help="是否使用自定义 API 参数（兼容旧方式）")
    
    # 以下参数仅在 use_custom_api=True 时使用（向后兼容）
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1",
                        help="[兼容参数] API base URL")
    parser.add_argument("--api_key", type=str, default="",
                        help="[兼容参数] API key")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="[兼容参数] 生成温度")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="[兼容参数] 最大生成token数")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="[兼容参数] Top-p sampling")
    parser.add_argument("--frequency_penalty", type=float, default=0.0,
                        help="[兼容参数] 频率惩罚")
    parser.add_argument("--presence_penalty", type=float, default=0.0,
                        help="[兼容参数] 存在惩罚")
    
    ########################################
    ######## 数据集载入参数 (OmniTrust 统一数据集管理) #####
    ########################################
    parser.add_argument("--data_path", type=str, 
                        default=os.path.join(project_root, "data/dataset/jailbreak_test.json"),
                        help="数据集文件路径（支持 .json 和 .jsonl 格式）")





    ########################################
    ########### ATTACK PARAMETERS ##########s
    ########################################
    # attack methods
    parser.add_argument("--attack", type=str, default="Multilingual",help="The attack method to use.")
    subparsers = parser.add_subparsers(dest="attack", help="Choose attack method")

    # Multilingual_attack
    multilingual_parser = subparsers.add_parser("Multilingual", help="Run Multilingual Attack")
    multilingual_parser.add_argument("--mutation_language", type=str, default="zh-CN", help="translate language")

    # DeepInception
    deepinception_parser = subparsers.add_parser("DeepInception", help="Run DeepInception attack")

    # CodeChameleon
    codechameleon_parser = subparsers.add_parser("CodeChameleon",help="Run CodeChameleon attack")
    codechameleon_parser.add_argument('--encrypt_rule', type=str, default='length', choices=['none', 'binary_tree', 'reverse','odd_even','length'], help='different encrypt methods')
    codechameleon_parser.add_argument('--prompt_style', type=str, default='code', choices=['text', 'code'], help='the style of prompt')

    # DRA
    dra_parser = subparsers.add_parser("DRA", help="Run DRA attack")
    dra_parser.add_argument("--em_t", type=float, default=0.7, help="em threshold")
    dra_parser.add_argument("--verbose", action="store_true", help="whether to print the intermediate process")

    # # CipherChat
    # cipherchat_parser = subparsers.add_parser("CipherChat", help="Run CipherChat attack")
    # cipherchat_parser.add_argument("--encode_method", type=str, default="ascii", choices=["unchange", "ascii", "caesar", "baseline", "unicode", "morse", "atbash", "utf", "gbk"], help="Encoding method to use for CipherChat attack")
    # cipherchat_parser.add_argument("--instruction_type", type=str, default="Crimes_And_Illegal_Activities",choices=["Crimes_And_Illegal_Activities", "Ethics_And_Morality", "Inquiry_With_Unsafe_Opinion", "Insult", "Mental_Health", "Physical_Harm", "Privacy_And_Property", "Reverse_Exposure", "Role_Play_Instruction","Unfairness_And_Discrimination", "Unsafe_Instruction_Topic"], help="Type of instruction for demonstrations")
    # cipherchat_parser.add_argument("--demonstration_toxicity", type=str, default=["toxic", "harmless"][0])
    # cipherchat_parser.add_argument("--language", type=str, default=["zh", "en"][-1])

    # # ArtPrompt
    # artprompt_parser = subparsers.add_parser("ArtPrompt", help="Run ArtPrompt attack")

    # # Jailbroken
    # jailbroken_parser = subparsers.add_parser("Jailbroken", help="Run Jailbroken attack")
    # jailbroken_parser.add_argument("--mutation_method", type=str, default="combination_3", choices=["disemvowel","leetspeak","artificial","auto_payload_splitting","auto_obfuscation", "base64_input_only","base64","base64_raw", "combination_1" ,"combination_2","combination_3","rot13"], help="Specify the mutation method to use")

    # # ReNeLLM
    # renellm_parser = subparsers.add_parser("ReNeLLM", help="Run ReNeLLM attack")
    # renellm_parser.add_argument('--rewrite_model', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4"], help='model uesd for rewriting the prompt')
    # renellm_parser.add_argument('--judge_model', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4"], help='model uesd for harmful classification')
    # renellm_parser.add_argument("--gpt_base_url",type=str,default="", help="baseurl for gpt api--attack and evaluator")
    # renellm_parser.add_argument('--iter_max', type=int, default=1, help='max iteration times')

    # # laa
    # laa_parser = subparsers.add_parser("Laa", help="Run laa attack")
    # ########### Target model parameters ##########
    # laa_parser.add_argument("--prompt-template", type=str, default="refined_best_simplified")
    # ############ Judge model parameters ##########
    # laa_parser.add_argument("--judge-model", default="gpt-4-0613", help="Name of judge model.",choices=["gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview", "no-judge"])
    # laa_parser.add_argument("--judge-max-n-tokens", type=int, default=10,help="Maximum number of tokens for the judge.")
    # laa_parser.add_argument("--judge-temperature", type=float, default=0, help="Temperature to use for judge.")
    # laa_parser.add_argument("--judge-top-p", type=float, default=1.0, help="Top-p to use for judge.")
    # laa_parser.add_argument("--judge-max-n-calls", type=int, default=1,help="Maximum number of calls to the judge inside the random search loop.")
    # ########### RS parameters ##########
    # laa_parser.add_argument("--n-iterations", type=int, default=5, help="Number of iterations to run the attack.")
    # laa_parser.add_argument("--n-chars-adv", type=int, default=100, help="Number of characters in the adversarial string.")
    # laa_parser.add_argument("--n-chars-change-max", type=int, default=0,help="Max number of characters to change per iteration.")

    # # Tap
    # tap_parser = subparsers.add_parser("Tap", help="Run Tap attack")
    # tap_parser.add_argument("--attack-model",default = "gpt-4",help = "Name of attacking model.",choices=["gpt-3.5-turbo",  "gpt-4", "gpt-4-turbo", "gpt-4-1106-preview"])
    # tap_parser.add_argument("--evaluator-model",default="gpt-4",help="Name of evaluator model.",choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4-1106-preview", "no-evaluator"])
    # tap_parser.add_argument("--gpt_base_url",type=str,default="", help="baseurl for gpt api--attack and evaluator")
    # ############### Tree of thought args ############ 
    # tap_parser.add_argument("--branching-factor",type = int,default = 1,help = "Branching factor") # 创建多个分支
    # tap_parser.add_argument("--width",type = int,default = 2,help = "Width") 
    # tap_parser.add_argument("--depth",type = int,default = 10,help = "Depth")
    # ########### Auxliary parameters ##########
    # tap_parser.add_argument("--n-streams",type = int, default = 1,help = "Number of root nodes in the tree of thoughts.")
    
    
    # ########### PiF parameters ##########
    # pif_parser = subparsers.add_parser("PiF", help="Run PiF attack")
    # pif_parser.add_argument("--gen_model_path", type=str, default='../Llama-2-7b-chat-hf')
    # pif_parser.add_argument("--tgt_model_path", type=str, default='../Mistral-7B-Instruct')
    # pif_parser.add_argument("--opt_objective", type=str, default='ASR')
    # pif_parser.add_argument("--prompt_template", type=str, default="Human: {}. Assistant: Sure, here is")
    # pif_parser.add_argument("--evaluation_template", type=str, default="this intent is [MASK]")
    # pif_parser.add_argument("--batch_size", type=int, default=16)
    # pif_parser.add_argument("--interation", type=int, default=50)
    # pif_parser.add_argument("--top_n", type=int, default=15)
    # pif_parser.add_argument("--top_m", type=int, default=15)
    # pif_parser.add_argument("--top_k", type=int, default=15)
    # pif_parser.add_argument("--warm_up", type=int, default=0)
    # pif_parser.add_argument("--threshold", type=float, default=0.85)
    # pif_parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    jailbreak_pipeline_black(args)



