import argparse
from datasets import load_dataset
import os
import sys
import torch
import spacy
import numpy as np
from .modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore, SelfCheckNgram, SelfCheckNLI, SelfCheckLLMPrompt
from .utils import data_generation
from tqdm import tqdm
import json
import random

def convert_np_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_np_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_floats(item) for item in obj]
    elif isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj


class selfcheckgpt_pipeline():
    def __init__(self, args):
        self.setting = args.setting

    def run(self, args):
        # bios, multi_bios = data_generation(args)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(current_dir, "qwen_entity_bios.jsonl")

        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                records.append(data)

        m = min(args.n_samples, len(records))
        sampled = random.sample(records, k=m)

        bios = [item["bio"] for item in sampled]
        multi_bios = [item["extra_bios"] for item in sampled]

        nlp = spacy.load("en_core_web_sm")
        all_sent_scores = []
        all_doc_scores  = []
        logs = []
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        if args.setting == "Ngram":
            selfcheck_ngram = SelfCheckNgram(n=1)
            all_sent_scores = []
            all_doc_scores  = []
            logs = []

            for idx, bio in enumerate(tqdm(bios, desc="Running SelfCheck-1gram")):
                sentences = [sent.text.strip() for sent in nlp(bio).sents]
                sampled_passages = multi_bios[idx]

                if len(sentences) == 0 or not sampled_passages:
                    sent_scores = []
                    doc_score = float("nan")
                else:
                    sent_scores_np = selfcheck_ngram.predict(
                        sentences=sentences,
                        passage=bio,
                        sampled_passages=sampled_passages,
                    )
                    sent_scores = sent_scores_np.tolist()
                    doc_score = float(np.mean(sent_scores))

                all_sent_scores.append(sent_scores)
                all_doc_scores.append(doc_score)
                logs.append({
                    "bio_index": idx,
                    "bio_text": bio,
                    "num_sentences": len(sentences),
                    "num_samples": len(sampled_passages),
                    "sentences": sentences,
                    "sentence_scores": sent_scores,
                    "doc_score": doc_score,
                })

            valid_doc_scores = [s for s in all_doc_scores if not np.isnan(s)]
            overall_avg_score = float(np.mean(valid_doc_scores)) if valid_doc_scores else float("nan")

            print("\n=== Ngram SelfCheck Completed ===")
            print(f"Average document-level score: {overall_avg_score:.6f}")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(current_dir, "selfcheck_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "qwen_selfcheck_ngram.json")

            output = {
                "method": "Ngram",
                "n": 1,
                "num_bios": len(bios),
                "overall_avg_score": overall_avg_score,
                "doc_scores": all_doc_scores,
                "logs": logs,
            }

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            print(f"Log file saved to: {log_path}")


        elif args.setting == "MQAG":
            selfcheck_mqag = SelfCheckMQAG(g1_model = args.api_model_name, g2_model = args.api_model_name, answering_model = args.api_model_name, answerability_model = args.api_model_name, max_tokens = 512, temperature = 0.3)
            sent_scores_mqag = selfcheck_mqag.predict(sentences = sentences, passage = passage, sampled_passages = [sample1, sample2, sample3], num_questions_per_sent = args.num_questions_per_sent, scoring_method = args.scoring_method)
            print(sent_scores_mqag)
        elif args.setting == "BERTScore":
            
            selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True, baseline_path=args.baseline_path)
            for idx, bio in enumerate(tqdm(bios, desc="Running SelfCheck-BERTScore")):
                sentences = [sent.text.strip() for sent in nlp(bio).sents] # spacy sentence tokenization

                sampled_passages = multi_bios[idx]
                if len(sentences) == 0 or not sampled_passages:
                    sent_scores = []
                    doc_score = float("nan")
                else:
                    sent_scores = selfcheck_bertscore.predict(
                        sentences=sentences,
                        sampled_passages=sampled_passages
                    ).tolist()
                    doc_score = float(np.mean(sent_scores))

                all_sent_scores.append(sent_scores)
                all_doc_scores.append(doc_score)
                logs.append({
                    "bio_index": idx,
                    "bio_text": bio,
                    "num_sentences": len(sentences),
                    "num_samples": len(sampled_passages),
                    "sentences": sentences,
                    "sentence_scores": sent_scores,
                    "doc_score": doc_score
                })


            valid_doc_scores = [s for s in all_doc_scores if not np.isnan(s)]
            overall_avg_score = float(np.mean(valid_doc_scores))

            print("\n=== BERTScore SelfCheck Completed ===")
            print(f"Average document-level score: {overall_avg_score:.6f}")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(current_dir, "selfcheck_logs")
            log_path = os.path.join(log_dir, "selfcheck_bertscore.json")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            output = {
                "method": "BERTScore",
                "num_bios": len(bios),
                "overall_avg_score": overall_avg_score,
                "doc_scores": all_doc_scores,
                "logs": logs,
            }
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)


            print(f"Log file saved to: {log_path}")

        elif args.setting == "LLMPrompt":
            selfcheck_prompt = SelfCheckLLMPrompt(args.model_path, device)
            sent_scores_prompt = selfcheck_prompt.predict(sentences = sentences, sampled_passages = [sample1, sample2, sample3], verbose = True,)
            print(convert_np_floats(sent_scores_prompt))

        elif args.setting == "NLI":
            selfcheck_nli = SelfCheckNLI(
                nli_model=args.nli_model,
                device=device
            )
            all_sent_scores = []
            all_doc_scores  = []
            logs = []

            for idx, bio in enumerate(tqdm(bios, desc="Running SelfCheck-NLI")):
                sentences = [sent.text.strip() for sent in nlp(bio).sents]

                sampled_passages = multi_bios[idx]

                if len(sentences) == 0 or not sampled_passages:
                    sent_scores = []
                    doc_score = float("nan")
                else:
                    sent_scores = selfcheck_nli.predict(
                        sentences=sentences,
                        sampled_passages=sampled_passages,
                    ).tolist()
                    doc_score = float(np.mean(sent_scores))

                all_sent_scores.append(sent_scores)
                all_doc_scores.append(doc_score)

                logs.append({
                    "bio_index": idx,
                    "bio_text": bio,
                    "num_sentences": len(sentences),
                    "num_samples": len(sampled_passages),
                    "sentences": sentences,
                    "sentence_scores": sent_scores,
                    "doc_score": doc_score,
                })
            valid_doc_scores = [s for s in all_doc_scores if not np.isnan(s)]
            overall_avg_score = float(np.mean(valid_doc_scores)) if valid_doc_scores else float("nan")

            print("\n=== NLI SelfCheck Completed ===")
            print(f"Average document-level score: {overall_avg_score:.6f}")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(current_dir, "selfcheck_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "selfcheck_nli.json")

            output = {
                "method": "NLI",
                "num_bios": len(bios),
                "overall_avg_score": overall_avg_score,
                "doc_scores": all_doc_scores,
                "logs": logs,
            }

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            print(f"Log file saved to: {log_path}")
