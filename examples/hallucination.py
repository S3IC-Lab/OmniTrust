import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.hallucination.llm.factool.pipeline import factool_pipeline
from modules.hallucination.llm.cove.pipeline import cove_pipeline
from modules.hallucination.llm.factscore.pipeline import factscore_pipeline
from modules.hallucination.llm.selfcheckgpt.pipeline import selfcheckgpt_pipeline
# from modules.hallucination.vlm.vl_uncertainty.pipeline import vl_uncertainty_pipeline
# from modules.hallucination.vlm.vlm_qa.pipeline import vlm_qa_pipeline
# from modules.hallucination.vlm.vlm_autodetect.pipeline import vlm_autodetect_pipeline


class Hallucination():
    def __init__(self, args):
        self.args = args
        if self.args.method == 'factool':
            self.pipeline = factool_pipeline(self.args)
        elif self.args.method == 'cove':
            self.pipeline = cove_pipeline(self.args)
        elif self.args.method == 'factscore':
            self.pipeline = factscore_pipeline(self.args)
        elif self.args.method == 'selfcheckgpt':
            self.pipeline = selfcheckgpt_pipeline(self.args)
        elif self.args.method == 'vl-uctt':
            self.pipeline = vl_uncertainty_pipeline(self.args)

    def llm_run(self):
        self.pipeline.run(self.args)
    
    def vlm_run(self, args):
        if args.method == "hallusionbench":
            pipeline = vlm_qa_pipeline(
                domain="hallusionbench",
                foundation_model_path=args.model_name
            )
        elif args.method == "vh-test-oeq":
            pipeline = vlm_qa_pipeline(
                domain="vh-test-oeq",
                foundation_model_path=args.model_name
            )
        elif args.method == "vh-test-ynq":
            pipeline = vlm_qa_pipeline(
                domain="vh-test-ynq",
                foundation_model_path=args.model_name
            )
        elif args.method == "auto-detect":
            pipeline = vlm_autodetect_pipeline(
                domain="auto-detect",
                foundation_model_path=args.model_name,
                evaluation_type = args.autodetect_type
            )
        elif  args.method == "vl-uctt":
            self.pipeline.run()
            return

        pipeline.get_response()
        result = pipeline.evaluation()
        return result
        
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
        factool_group.add_argument("--wrapper", type=str, choices=["gpt-3.5-turbo", "gpt-4"], default="gpt-3.5-turbo")
        factool_group.add_argument('--n_samples', type=int, default=2)
        
    
    elif args.method == "cove":
        cove_group = parser.add_argument_group('cove')
        cove_group.add_argument("--setting", type=str, default="two_step", choices=["joint", "two_step", "factored"])
        cove_group.add_argument("-m","--model",type=str, help="LLM to use for predictions.", default="qwen3", choices=["llama2", "llama2_70b", "llama-65b", "gpt-3.5-turbo", "gpt-4", "deepseek-r1", "qwen3"],)
        cove_group.add_argument("-t", "--task", type=str, help="Task.", default="multispan_qa", choices=["wikidata", "wikidata_category", "multispan_qa"],)
        cove_group.add_argument("-temp", "--temperature", type=float, help="Temperature.", default=0.07)
        cove_group.add_argument("-p", "--top-p", type=float, help="Top-p.", default=0.9)
        cove_group.add_argument('--n_samples', type=int, default=10)
        cove_group.add_argument('--model_path', default='/home/model/Qwen_Qwen3-14B', type=str)


    elif args.method == 'factscore':
        factscore_group = parser.add_argument_group('factscore')
        factscore_group.add_argument('--model_name', type=str, default="qwen3", choices=["llama-7b", "gpt-3.5-turbo", "deepseek-r1", "qwen3"])
        factscore_group.add_argument('--setting', type=str, default="retrieval+ChatGPT", choices=["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm"])
        factscore_group.add_argument('--gamma', type=int, default=10, help="hyperparameter for length penalty")

        factscore_group.add_argument('--db_path', type=str, default="/home/model/enwiki-20230401.db")
        factscore_group.add_argument('--auxiliary_model', type=str, default="/home/hub/model/Llama-2-7b-chat-hf")
        factscore_group.add_argument('--model_path', type=str, default="/home/model/Qwen_Qwen3-14B")
        factscore_group.add_argument('--knowledge_source', type=str, default="enwiki-20230401")
        factscore_group.add_argument('--retrieval_path', type=str, default="/home/hub/model/gtr-t5-large")
        

        factscore_group.add_argument('--use_atomic_facts', action="store_true", default=True)
        factscore_group.add_argument('--verbose', action="store_true", help="for printing out the progress bar")    
        factscore_group.add_argument('--n_samples', type=int, default=100)


    elif args.method == 'selfcheckgpt':
        selfcheckgpt_group = parser.add_argument_group('selfcheckgpt')
        selfcheckgpt_group.add_argument("--setting", type=str, default="Ngram", choices=["MQAG", "BERTScore", "NLI", "LLMPrompt", "Ngram"])
        selfcheckgpt_group.add_argument('--model_path', type=str, default="/home/model/Qwen_Qwen3-14B")
        selfcheckgpt_group.add_argument('--baseline_path', type=str, default="/home/jinhong_chen/storage/AwesomeLLMSecurityPlatform/modules/hallucination/llm/selfcheckgpt/roberta-large.tsv")
        selfcheckgpt_group.add_argument('--n_samples', type=int, default=100)
        selfcheckgpt_group.add_argument('--num_questions_per_sent', type=int, default=5, help="Number of questions to generate per sentence.")
        selfcheckgpt_group.add_argument('--scoring_method', type=str, default='bayes_with_alpha', choices=['bayes_with_alpha', 'mean', 'max'], help="Scoring method for MQAG setting.")
        selfcheckgpt_group.add_argument('--api-model-name', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4"])
        selfcheckgpt_group.add_argument('--nli_model', type=str, default="/home/hub/model/deberta-v3-large")
        selfcheckgpt_group.add_argument('--n_repeats', type=int, default=5)
        selfcheckgpt_group.add_argument('--ngram_n', type=int, default=1)


    elif args.method == 'hallusionbench' or args.method == 'vh-test-oeq' or args.method == 'vh-test-ynq':
        halluvh_group = parser.add_argument_group('halluvh')
        halluvh_group.add_argument('--model-name', type=str, default="llama-7b")
    
    elif args.method == 'auto-detect':
        auto_group = parser.add_argument_group('autodetect')
        auto_group.add_argument('--model-name', type=str, default="llama-7b")
        auto_group.add_argument('--autodetect-type', type=str, default="g", help="LLM-free detect type, can use 'g' or 'd' or 'a'")

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
        vluctt_group.add_argument('--model_path_dir', type=str, default='/home/chunxue_xu/storage/model')
        vluctt_group.add_argument('--data_path_dir', type=str, default='/home/chunxue_xu/storage/dataset/ScienceQA')


    args = parser.parse_args()
    
    hallucination_instance = Hallucination(args)
    if(args.model_type == "llm"):
        hallucination_instance.llm_run()
    else:
        outputs = hallucination_instance.vlm_run(args)

    


# hallucination_instance = Hallucination("liuhaotian/llava-v1.5-13b")
# response_list = hallucination_instance.vlm_run()

# print(response_list)