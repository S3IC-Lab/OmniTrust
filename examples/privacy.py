import argparse
import sys
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_adapter import load_model_demo
from modules.privacy.pipeline import privacy_pipeline_demo
import warnings
warnings.filterwarnings("ignore")


def llm_privacy(args):
    model, tokenizer = load_model_demo(args)
    privacy_pipeline_demo(model=model, tokenizer=tokenizer, args=args)


def diffusion_privacy(args):
    privacy_pipeline_demo(args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', default='C_DEA', type=str, choices=['MIA', 'C_DEA', 'SD_MIA'])
    parser.add_argument('--revision', default="main", type=str)
    parser.add_argument('--num_gpus_per_model', default=1, type=int)
    parser.add_argument('--max_gpu_memory', help="Maxmum GPU memory used for model weights per GPU.", type=int)
    parser.add_argument('--model_path', default='/home/hub/model/gpt2', type=str)

    args, remaining_argv = parser.parse_known_args()

    if args.attack == 'MIA':
        parser.add_argument('--metric', default='MIN_K_PROB', type=str,
                            choices=['LOSS', 'PPL', 'REFER', 'ZLIB', 'LOWER_CASE', 'WINDOW', 'LIRA', 'NEIGHBOR', 'MIN_K_PROB', 'MIN_K_PLUS_PROB', 'RECALL'])
        parser.add_argument('--num_sample', default=10, type=int, help='use -1 to include all samples')
        parser.add_argument('--dataset', default='wikitext-103-v1', type=str,
                            choices=['agnews', 'xsum', 'wikitext-103-v1', 'wikitext-2-v1', 'wikimia'])
        parser.add_argument('--ref_model_path', default='/home/model/gpt2', type=str, help='reference model path')
        parser.add_argument('--max_seq_len', default=1024, type=int)
        parser.add_argument('--n_neighbor', default=50, type=int, help='num of neighbors in neighbor attack')
        parser.add_argument('--k_ratio', default=0.1, type=float, help='K ratio for Min-K and Min-K++ methods (default: 0.1)')
        parser.add_argument('--recall_num_shots', default=5, type=int, help='Number of prefix shots for ReCaLL metric (default: 5)')
        parser.add_argument('--recall_prefix_path', default='', type=str, help='Optional file with prefixes for ReCaLL metric')
        parser.add_argument('--model_name', default='gpt2', type=str, choices=['gpt2', 'llama2'])

    elif args.attack == 'C_DEA':
        parser.add_argument('--num_sample', default=10, type=int, help='use -1 to include all samples')
        parser.add_argument('--method', default='enron', type=str, choices=['enron', 'memrise'])
        parser.add_argument('--min_prompt_len', default=200, type=int)
        parser.add_argument('--max_seq_len', default=512, type=int)
        parser.add_argument('--dataset', default="enron", type=str, choices=["enron", "xsum", "agnews"])
        parser.add_argument('--bert_path', default='/home/hub/model/bert', type=str)
        parser.add_argument('--prefix_len', default=100, type=int)
        parser.add_argument('--suffix_len', default=500, type=int)

    elif args.attack == 'SD_MIA':
        parser.add_argument('--method', default='naive', type=str, choices=['naive', 'pia', 'pian', 'sec', 'black_naive', 'black_score_t', 'black_score_c', 'black_score_d'])
        parser.add_argument('--num_sample', default=10, type=int)
        parser.add_argument('--dataset', default='coco', type=str, choices=['coco', 'pokemon', 'flickr', 'text-to-image-2m'])
        parser.add_argument('--filter', default=0, type=int, help='Improvements in naive/sec/pia/pian')  # https://arxiv.org/abs/2505.20955
        parser.add_argument('--device', default='cuda:2', type=str)

    args = parser.parse_args(remaining_argv, namespace=args)

    if args.attack == 'SD_MIA':
        diffusion_privacy(args)
    else:
        llm_privacy(args)
