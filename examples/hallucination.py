import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.hallucination.llm.factool.pipeline import factool_pipeline

from modules.hallucination.vlm.vl_uncertainty.pipeline import vl_uncertainty_pipeline
from modules.hallucination.vlm.vlm_autodetect.pipeline import vlm_autodetect_pipeline
from modules.hallucination.vlm.vlm_qa.pipeline import vlm_qa_pipeline


class Hallucination():
    def __init__(self, args):
        self.args = args
        if self.args.method == 'factool':
            self.pipeline = factool_pipeline(self.args)
        elif self.args.method == 'vl-uctt':
            self.pipeline = vl_uncertainty_pipeline(self.args)
        elif self.args.method == 'auto-detect':
            self.pipeline = vlm_autodetect_pipeline(self.args)
        elif self.args.method in ['hallusionbench', 'vh-test-oeq', 'vh-test-ynq']:
            self.pipeline = vlm_qa_pipeline(self.args)

    def llm_run(self):
        self.pipeline.run(self.args)
    
    def vlm_run(self):
        if args.method in ["vl-uctt", "auto-detect", "hallusionbench", "vh-test-oeq", "vh-test-ynq"]:
            # 统一使用 pipeline.run() 方法
            self.pipeline.run()
            return
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="cove", choices=["factool", "cove", "factscore", 'selfcheckgpt', 'hallusionbench', 'vh-test-oeq', 'vh-test-ynq', 'auto-detect', 'vl-uctt'], help="The detection method to use.")
    parser.add_argument("--model-type", type=str, required=False, default="llm", choices=["llm", "vlm"], help="Select model type: LLM (Large Language Model) or VLM (Visual Language Model)")
    args, remaining_argv = parser.parse_known_args()

    if args.method == 'factool':
        factool_group = parser.add_argument_group('factool')
        factool_group.add_argument("--model-source", type=str, default="api", choices=["local", "api"])
        factool_group.add_argument("--task", type=str, default="kbqa", choices=["sci", "code", "math", "kbqa"])
        # factool_group.add_argument("--model_path", type=str, default='/home/hub/model/Llama-2-7b-chat-hf')
        factool_group.add_argument("--data_path", type=str, default='/home/jinhong_chen/storage/AwesomeLLMSecurityPlatform/modules/hallucination/llm/factool/data/kbqa.jsonl')
        factool_group.add_argument("--api-model-name", type=str, default="gpt-3.5-turbo")
        factool_group.add_argument("--search-type", type=str, default="online", choices=["online", "local"])
        factool_group.add_argument("--wrapper", type=str, choices=["gpt-3.5-turbo", "gpt-4"], default="gpt-4o")
        factool_group.add_argument('--n_samples', type=int, default=2)

    elif args.method == 'hallusionbench' or args.method == 'vh-test-oeq' or args.method == 'vh-test-ynq':
        halluvh_group = parser.add_argument_group('halluvh')
        halluvh_group.add_argument('--model-name', type=str, default="llava-1.5-13b-hf", 
                                   help="Model name or path. Supports: llava-1.5-13b-hf, llava-1.5-7b-hf, llava-v1.6-vicuna-13b-hf, Qwen2-VL-2B-Instruct, etc.")
        halluvh_group.add_argument('--model_path_dir', type=str, default=None, help='Model path directory (optional)')
        halluvh_group.add_argument('--data_path_dir', type=str, default=None, help='Data path directory (optional)')
    
    elif args.method == 'auto-detect':
        auto_group = parser.add_argument_group('autodetect')
        auto_group.add_argument('--model-name', type=str, default="llava-1.5-13b-hf", 
                               help="Model name or path. Supports: llava-1.5-13b-hf, llava-1.5-7b-hf, llava-v1.6-vicuna-13b-hf, Qwen2-VL-2B-Instruct, etc.")
        auto_group.add_argument('--autodetect-type', type=str, default="d", 
                               choices=['a', 'g', 'd', 'de', 'da', 'dr'],
                               help="LLM-free detect type: 'a' (all), 'g' (generative), 'd' (discriminative), 'de' (existence), 'da' (attribute), 'dr' (relation)")
        auto_group.add_argument('--model_path_dir', type=str, default=None, help='Model path directory (optional)')
        auto_group.add_argument('--data_path_dir', type=str, default='modules/hallucination/vlm/vlm_autodetect/query', help='Data path directory (optional)')
        auto_group.add_argument('--word-association', type=str, default=None, help='Path to word association file')
        auto_group.add_argument('--safe-words', type=str, default=None, help='Path to safe words file')
        auto_group.add_argument('--annotation', type=str, default=None, help='Path to annotation file')
        auto_group.add_argument('--metrics', type=str, default=None, help='Path to metrics file')
        auto_group.add_argument('--similarity-score', type=float, default=0.8, help='Similarity score threshold')

    elif args.method == 'vl-uctt':
        vluctt_group = parser.add_argument_group('vl-uctt')
        vluctt_group.add_argument('--lvlm', type=str, default='Qwen2-VL-2B-Instruct')
        vluctt_group.add_argument('--benchmark', type=str, default='MMVet')
        vluctt_group.add_argument('--llm', type=str, default='Qwen2.5-3B-Instruct')
        vluctt_group.add_argument('--uncertainty', type=str, default='vl_uncertainty')
        vluctt_group.add_argument('--uncertainty_thres', type=float, default=1.0)
        vluctt_group.add_argument('--visual_perturbation', type=str, default='blurring')
        vluctt_group.add_argument('--blur_radius_list', type=float, nargs='+', default=[0.6, 0.8, 1.0, 1.2, 1.4])
        vluctt_group.add_argument('--textual_perturbation', type=str, default='llm_rephrasing')
        vluctt_group.add_argument('--textual_perturbation_temp_list', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5])
        vluctt_group.add_argument('--textual_perturbation_instruction_template', type=str, default="Given the input question: '{question}', generate a semantically equivalent variation by changing the wording, structure, grammar, or narrative. Ensure the perturbed question maintains the same meaning as the original. Provide only the rephrased question as the output.")
        vluctt_group.add_argument('--pair_order', type=str, default='progressively')
        vluctt_group.add_argument('--inference_temp', type=float, default=0.1)
        vluctt_group.add_argument('--sampling_temp', type=float, default=1.0)
        vluctt_group.add_argument('--sampling_time', type=int, default=5)
        vluctt_group.add_argument('--model_path_dir', type=str, default='~/models')
        vluctt_group.add_argument('--data_path_dir', type=str, default='~/datasets/ScienceQA')


    args = parser.parse_args()
    
    hallucination_instance = Hallucination(args)
    if(args.model_type == "llm"):
        hallucination_instance.llm_run()
    else:
        hallucination_instance.vlm_run()