import pathlib
import jsonlines
import argparse
import json
from model.model_adapter import load_model_demo
from openai import OpenAI
import random
from config import GET_API, LOAD_API_CONFIG

def load_data(args):
    inputs = list()
    dataset_path = pathlib.Path(args.data_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    with jsonlines.open(dataset_path) as reader:
        for line in reader:
            inputs.append(line)

    if hasattr(args, 'n_samples') and args.n_samples is not None:
        max_n = len(inputs)
        if args.n_samples > max_n:
            print(f"[Warning] Requested n_samples={args.n_samples}, "
                  f"but only {max_n} available. Using {max_n} instead.")
            args.n_samples = max_n

        inputs = random.sample(inputs, args.n_samples)
    return inputs



def calculate(outputs):
    # calculate average response_level_factuality
    total_response_factuality = sum(output['response_level_factuality'] == True for output in outputs)
    avg_response_level_factuality = total_response_factuality / len(outputs)

    # calculate average claim_level_factuality
    num_claims = 0
    total_claim_factuality = 0
    for output in outputs:
        if output['category'] == 'code':
            total_claim_factuality += (output['claim_level_factuality'] == True)
            num_claims += 1
        else:
            num_claims += len(output['claim_level_factuality'])
            if output['category'] == 'kbqa':
                if output['claim_level_factuality'] != []:
                    total_claim_factuality += sum(claim['factuality'] == True if claim != None else 0 for claim in output['claim_level_factuality'])
                else:
                    total_claim_factuality += 0
            elif output['category'] == 'math':
                total_claim_factuality += sum(claim_factuality == True for claim_factuality in output['claim_level_factuality'])
            elif output['category'] == 'scientific':
                total_claim_factuality += sum(claim['factuality'] == True for claim in output['claim_level_factuality'])



    avg_claim_level_factuality = total_claim_factuality / num_claims

    return {"average_claim_level_factuality": avg_claim_level_factuality, "average_response_level_factuality": avg_response_level_factuality, "detailed_information": outputs}


def save_result(args, results):
    current_dir = pathlib.Path(__file__).parent
    dataset_path = current_dir / "results" / f"{args.task}.jsonl"

    dataset_path = dataset_path.resolve()

    with open(dataset_path, 'w', encoding='utf-8') as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + '\n')


def response_generation(args, inputs):
    prompt_templates = {
        'code': (
            "Generate Python code ONLY for this problem. Maintain exact indentation and syntax.\n"
            "Problem: {question}\n"
            "Respond ONLY with the code, no explanations."
        ),
        'math': (
            "Solve this math problem directly. Show key steps with '=' in single line.\n"
            "Problem: {question}\n"
            "Format: [Required Variable] = [Calculation]\nFinal Answer = [Value]"
            "Respond ONLY with calculations, no text."
        ),
        'kbqa': (
            "Provide a detailed explanation with comprehensive context.\n"
            "Question: {question}\n"
            "Keep statements concise but comprehensive."
            "Respond ONLY with factual statements."
        ),
        'sci': (
            "Provide citations in the format: Model (Authors, Year). Include 3+ papers.\n"
            "Question: {question}\n"
            "Respond ONLY with citation statements."
        )
    }
    if args.model_source == "api":
        # openai.api_key = os.environ.get("OPENAI_API_KEY", None)
        api_key, api_url = GET_API(model_name)
        if not api_url:
            raise ValueError(f"api_url is missing for model: {model_name}")
        if not api_key:
            raise ValueError(f"api_key is missing for model: {model_name}")

        client = OpenAI(
            base_url=api_url,
            api_key=api_key,
        )

        for i, dic in enumerate(inputs):
            response = client.chat.completions.create(
                model=args.api_model_name,
                messages=[{
                    "role": "user",
                    "content": prompt_templates[args.task].format(question=dic['prompt'])
                }],
        )
            dic['response'] = response.choices[0].message.content.strip()
    else:
    
        model, tokenizer = load_model_demo(args)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = model.device

        for i, dic in enumerate(inputs):
            current_prompt = prompt_templates[args.task].format(question=dic['prompt'])
            tokenized_input = tokenizer(
                current_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            input_ids = tokenized_input.input_ids.to(device)
            attention_mask = tokenized_input.attention_mask.to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_text.replace(current_prompt, "", 1).strip()
            dic['response'] = response
    return inputs


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        type=str,
        default="kbqa",
        help="The detection method to use."
    )
    args = parser.parse_args()

    inputs = load_data(args)
    print(inputs[0])