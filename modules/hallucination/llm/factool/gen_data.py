import os
import argparse
import pathlib
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import random
from tqdm import tqdm
from torch.nn import DataParallel
from config import GET_API, LOAD_API_CONFIG

def load_dataset(task: str, n_samples: int = None):
    current_dir = pathlib.Path(__file__).parent
    dataset_path = current_dir / "../../../../data/dataset/factool" / f"{task}.jsonl"
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    inputs = []
    with jsonlines.open(dataset_path) as reader:
        for line in reader:
            inputs.append(line)
    if n_samples is None or n_samples >= len(inputs):
        return inputs

    return random.sample(inputs, n_samples)


def get_prompt_templates():
    return {
        'code': (
            "Generate Python code ONLY for this problem. Maintain exact indentation and syntax.\n"
            "Problem: {question}\n"
            "Respond ONLY with the code, no explanations."
        ),
        'math': (
            "Solve this math problem directly. Show key steps with '=' in single line.\n"
            "Problem: {question}\n"
            "Format: [Required Variable] = [Calculation]\nFinal Answer = [Value]\n"
            "Respond ONLY with calculations, no text."
        ),
        'kbqa': (
            "Provide a detailed explanation with comprehensive context.\n"
            "Question: {question}\n"
            "Keep statements concise but comprehensive.\n"
            "Respond ONLY with factual statements."
        ),
        'sci': (
            "Provide citations in the format: Model (Authors, Year). Include 3+ papers.\n"
            "Question: {question}\n"
            "Respond ONLY with citation statements."
        )
    }

def run_api_inference(args, inputs, prompt_templates):
    api_key, api_url = GET_API('gpt-4o')
    if not api_url:
        raise ValueError(f"api_url is missing for model: {'gpt-4o'}")
    if not api_key:
        raise ValueError(f"api_key is missing for model: {'gpt-4o'}")

    self.client = OpenAI(
        base_url=api_url,
        api_key=api_key,
    )

    for i, dic in enumerate(inputs):
        prompt = prompt_templates[args.task].format(question=dic['prompt'])
        response = client.chat.completions.create(
            model=args.api_model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        dic['response'] = response.choices[0].message.content.strip()
    return inputs


def run_local_inference(args, inputs, prompt_templates):
    model_name = "/home/model/Qwen_Qwen3-14B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    for i, dic in tqdm(enumerate(inputs), total=len(inputs), desc="Running inference"):
        current_prompt = prompt_templates[args.task].format(question=dic['prompt'])
        
        tokenized_input = tokenizer(
            current_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized_input,
                max_new_tokens=300,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text.replace(current_prompt, "", 1).strip()
        dic['response'] = response

    return inputs


def response_generation(args):
    inputs = load_dataset(args.task, args.n_samples)
    prompt_templates = get_prompt_templates()

    if args.model_source == "api":
        return run_api_inference(args, inputs, prompt_templates)
    else:
        return run_local_inference(args, inputs, prompt_templates)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sci", help="The task type: code | math | kbqa | sci")
    parser.add_argument("--model_source", type=str, default="local", help="Model source: api | local")
    parser.add_argument("--api_model_name", type=str, default="gpt-3.5-turbo", help="API model name if using api")
    parser.add_argument('--n_samples', type=int, default=1000)
    args = parser.parse_args()

    results = response_generation(args)
    current_dir = pathlib.Path(__file__).parent
    output_path = current_dir / "data" / f"{args.task}.jsonl"
    output_path = output_path.resolve()

    with jsonlines.open(output_path, mode="w") as writer:
        for dic in results:
            writer.write(dic)

    print(f"✅ Results saved to: {output_path}")
