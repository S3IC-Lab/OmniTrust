import pathlib
from model.model_adapter import load_model_demo
from transformers import AutoTokenizer, AutoModelForCausalLM
from .evaluate import evaluate
import torch
from .data.data_processor import (
    read_json,
    get_questions_from_list,
    get_questions_from_dict,
)

class cove_pipeline():
    def __init__(self, args):
        current_dir = pathlib.Path(__file__).parent
        self.file_path_mapping = {
            "wikidata": current_dir / "../../../../data/dataset/cove/wikidata_questions.json",
            "multispan_qa": current_dir / "../../../../data/dataset/cove/multispanqa_dataset.json",
            "wikidata_category": current_dir / "../../../../data/dataset/cove/wikidata_category_dataset.json",
        }


    def run(self, args):
        data = read_json(self.file_path_mapping[args.task])
        if args.task == "wikidata":
            questions = get_questions_from_dict(data)
        else:
            questions = get_questions_from_list(data)

        n_samples = min(args.n_samples, len(questions))
        questions = questions[:n_samples]

        if args.model[:3] == "gpt":
            from .cove_chains_openai import ChainOfVerificationOpenAI
            chain_openai = ChainOfVerificationOpenAI(
                model_id=args.model,
                temperature=args.temperature,
                task=args.task,
                setting=args.setting,
                questions=questions,
            )
            result_path = chain_openai.run_chain()
        else:
            from .cove_chains_hf import ChainOfVerificationHuggingFace
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            chain_hf = ChainOfVerificationHuggingFace(
                model_id=args.model,
                top_p=args.top_p,
                temperature=args.temperature,
                task=args.task,
                setting=args.setting,
                questions=questions,
                model=model, 
                tokenizer=tokenizer,
            )
            result_path = chain_hf.run_chain()
        evaluate(result_path, self.file_path_mapping[args.task], args.task)
