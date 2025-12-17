import random
import argparse
import torch
import pandas as pd
import json
import collections
from attack.privacy.models.ft_clm import FinetunedCasualLM
from attack.privacy.attacks.DataExtraction.Analyze import get_enron_results, get_memrise_results, read_all_models
from tqdm import tqdm
from data.data_registry.agnews import agnewsDataset
from data.data_registry.xsum import xsumDataset
from data.data_registry.enron import EnronDataset
random.seed(0)


def Conditional_DEA(model, tokenizer, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading models
    llm = FinetunedCasualLM(model_path=args.model_path, arch=args.model_path, max_seq_len=args.max_seq_len, model=model, tokenizer=tokenizer)
    print("The model is loaded successfully")

    # loading datasets
    prompts = []
    labels = []
    if args.dataset == 'enron':
        args.method = 'enron'
        enron = EnronDataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/enron")
        format = f'prefix-{args.min_prompt_len}'
        # format in ['prefix-50','0-shot-known-domain-b','0-shot-unknown-domain-c', '3-shot-known-domain-c', '5-shot-unknown-domain-b'........]
        prompts, labels = enron.load_data(format=format, tokenizer_path=args.model_path)
    elif args.dataset == 'xsum':
        args.method = 'memrise'
        dataset = xsumDataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/xsum/default", datatype='train')
        prompts, labels = dataset.load_data()
    elif args.dataset == 'agnews':
        args.method = 'memrise'
        dataset = agnewsDataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/agnews/data", datatype='train')
        prompts, labels = dataset.load_data()


    num_samples = len(prompts)
    if args.num_sample != -1 and args.num_sample < num_samples:
        prompts = prompts[:args.num_sample]
        labels = labels[:args.num_sample]
    else:
        prompts = prompts[:num_samples]
        labels = labels[:num_samples]

    # attacks
    output_fname = "../attack/privacy/attacks/DataExtraction/result/test_output.jsonl"
    result = []
    print("Start attack")
    for i, prompt in enumerate(tqdm(prompts)):
        ground_truth = labels[i]
        if args.method == 'enron':
            res = llm.query(prompt)
            result.append({'idx': i, 'output': res, 'label': ground_truth, 'prompt': prompt})
        elif args.method == 'memrise':
            res = llm.query(prompt[:args.prefix_len])
            result.append({'idx': i, 'output': res, 'label': ground_truth, 'prompt': prompt})

        if i % 100 == 0:
            print(f'Finish {i} samples')
            with open(output_fname, 'w') as outfile:
                for entry in result:
                    json.dump(entry, outfile)
                    outfile.write('\n')

    with open(output_fname, 'w') as outfile:
        for entry in result:
            json.dump(entry, outfile)
            outfile.write('\n')

    # Analyze the saved results
    target_models, target_files = read_all_models(subfix=".jsonl", BASE_DIR="../attack/privacy/attacks/DataExtraction/result")
    models2files = {}
    for i, model in enumerate(target_models):
        models2files[model] = target_files[i]
    od_models2files = collections.OrderedDict(sorted(models2files.items()))

    result_list = []

    if args.method == 'enron':
        for model, filename in od_models2files.items():
            result = get_enron_results(filename)
            correct_count, local_correct_count, domain_correct_count, total_count, total_wo_reject_count = result

            correct_count_acc = correct_count / total_count * 100
            local_correct_count_acc = local_correct_count / total_count * 100
            domain_correct_count_acc = domain_correct_count / total_count * 100
            reject_rate = (1 - total_wo_reject_count / total_count) * 100
            leakage_rate_wo_reject = (correct_count + local_correct_count + domain_correct_count) / 3 / total_wo_reject_count * 100
            leakage_rate = (correct_count + local_correct_count + domain_correct_count) / 3 / total_count * 100

            model = model.replace("_", "-")
            cur_result = {"dataset": 'all', "model": model,
                          "correct": round(correct_count_acc, 2),
                          "correct_local": round(local_correct_count_acc, 2),
                          "correct_domain": round(domain_correct_count_acc, 2),
                          "leak_rate": round(leakage_rate, 2),
                          "reject_rate": round(reject_rate, 2),
                          "leak_rate_wo_reject": round(leakage_rate_wo_reject, 2),
                          }
            print(cur_result)
            result_list.append(cur_result)
    elif args.method == 'memrise':
        for model, filename in od_models2files.items():
            result = get_memrise_results(filename, llm, args.prefix_len, args.suffix_len, args.bert_path)
            model = model.replace("_", "-")
            total_count, reject_count, top_likelihood, average_similarity = result
            reject_rate = (reject_count / total_count) * 100

            cur_result = {"dataset": 'all', "model": model,
                          "reject_rate": round(reject_rate, 2),
                          "average_similarity": average_similarity,
                          "top_likelihood": top_likelihood
                          }
            print(cur_result)
            result_list.append(cur_result)

    df = pd.DataFrame.from_dict(result_list)
    df.to_csv('../attack/privacy/attacks/DataExtraction/result//dea_pythia.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sample', default=10, type=int, help='use -1 to include all samples')
    parser.add_argument('--model', default='gpt2', type=str, choices=['gpt2', 'llama2'])
    parser.add_argument('--method', default='enron', type=str, choices=['enron', 'memrise'])
    parser.add_argument('--min_prompt_len', default=200, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--data', default="xsum", type=str, choices=["enron", "xsum", "agnews"])
    parser.add_argument('--revision', default="main", type=str)
    parser.add_argument('--num_gpus_per_model', default=1, type=int)
    parser.add_argument('--max_gpu_memory', help="Maxmum GPU memory used for model weights per GPU.", type=int)
    parser.add_argument('--model_path', default=' ', type=str)
    parser.add_argument('--bert_path', default='../../download/bert', type=str)
    parser.add_argument('--prefix_len', default=100, type=int)
    parser.add_argument('--suffix_len', default=500, type=int)
    args = parser.parse_args()

    Conditional_DEA(args)