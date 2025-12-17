import argparse
import os
from model.model_adapter import load_model_demo
from attack.privacy.models.ft_clm import FinetunedCasualLM
from attack.privacy.attacks.MIA.member_inference import MemberInferenceAttack, MIAMetric
from data.data_registry.agnews import agnewsDataset
from data.data_registry.xsum import xsumDataset
from data.data_registry.wikitext import wikitextDataset
from data.data_registry.WikiMIA import WikiMIADataset  # 添加 WikiMIA 导入


def make_if_not_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)


def load_recall_prefixes(prefix_path):
    if not prefix_path:
        return []
    if not os.path.exists(prefix_path):
        print(f"WARN: ReCaLL prefix file {prefix_path} not found. Falling back to dataset prefixes.")
        return []
    with open(prefix_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    candidates = [seg.strip() for seg in raw.split('\n\n') if seg.strip()]
    if not candidates:
        # fallback to newline splits if double-newline parsing yields nothing
        candidates = [line.strip() for line in raw.splitlines() if line.strip()]
    return candidates


def MIA(model, tokenizer, args):
    args.run_name = f"{args.metric}_{args.num_sample}"
    if args.max_seq_len != 1024:
        args.run_name += f"_len{args.max_seq_len}"
    if args.dataset != 'echr':
        args.run_name += f"_{args.dataset}"
    if args.n_neighbor != 50:
        args.run_name += f"_nn{args.n_neighbor}"
    # Add k_ratio to run_name for Min-K and Min-K++ methods
    if args.metric in ['MIN_K_PROB', 'MIN_K_PLUS_PROB'] and args.k_ratio != 0.1:
        args.run_name += f"_k{args.k_ratio}"
    
    args.result_dir = os.path.join("./results/", f"{args.model_name}")
    make_if_not_exist(args.result_dir)
    cache_file = os.path.join(args.result_dir, args.run_name)
    metric = MIAMetric[args.metric]

    # loading datasets
    if args.dataset == 'agnews':
        ds_1 = agnewsDataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/agnews/data", datatype='train')
        train_set, _ = ds_1.load_data()
        ds_2 = agnewsDataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/agnews/data", datatype='test')
        test_set, _ = ds_2.load_data()
    elif args.dataset == 'xsum':
        ds_1 = xsumDataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/xsum/default", datatype='train')
        train_set, _ = ds_1.load_data()
        ds_2 = xsumDataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/xsum/default", datatype='test')
        test_set, _ = ds_2.load_data()
    elif args.dataset == 'wikitext-103-v1':
        ds_1 = wikitextDataset("//home/puwei_lian/workspace/OmniTrust/data/dataset/wikitext/wikitext-103-raw-v1", datatype='train')
        train_set, _ = ds_1.load_data()
        ds_2 = wikitextDataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/wikitext/wikitext-103-raw-v1", datatype='test')
        test_set, _ = ds_2.load_data()
    elif args.dataset == 'wikitext-2-v1':
        ds_1 = wikitextDataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/wikitext/wikitext/wikitext-2-v1", datatype='train')
        train_set, _ = ds_1.load_data()
        ds_2 = wikitextDataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/wikitext/wikitext/wikitext-2-v1", datatype='test')
        test_set, _ = ds_2.load_data()
    elif args.dataset == 'wikimia':  # 添加 WikiMIA 数据集支持
        ds_1 = WikiMIADataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/wikimia/data", datatype='member')
        train_set, _ = ds_1.load_data()
        ds_2 = WikiMIADataset("/home/puwei_lian/workspace/OmniTrust/data/dataset/wikimia/data", datatype='non_member')
        test_set, _ = ds_2.load_data()
    else:
        raise NotImplementedError(f"dataset: {args.dataset}")

    if args.num_sample > 0:  # 修改条件，支持 -1 表示使用全部数据
        if args.num_sample < len(train_set):
            train_set = train_set[:args.num_sample]
        if args.num_sample < len(test_set):
            test_set = test_set[:args.num_sample]

    # loading models
    llm = FinetunedCasualLM(model_path=args.model_path, arch=args.model_path, max_seq_len=args.max_seq_len, model=model, tokenizer=tokenizer)

    if metric in (MIAMetric.REFER, MIAMetric.LIRA, MIAMetric.NEIGHBOR):
        # 保存原始模型路径
        original_model_path = args.model_path
        args.model_path = args.ref_model_path
        model_1, tokenizer_1 = load_model_demo(args)
        ref_llm = FinetunedCasualLM(model_path=args.ref_model_path, arch=args.ref_model_path, max_seq_len=args.max_seq_len, model=model_1, tokenizer=tokenizer_1)
        ref_llm._lm.eval()
        # 恢复原始模型路径
        args.model_path = original_model_path
    else:
        ref_llm = None

    recall_prefixes = load_recall_prefixes(getattr(args, 'recall_prefix_path', ''))
    # attacks
    print("Start attack")
    attack = MemberInferenceAttack(
        metric=metric, 
        ref_model=ref_llm, 
        n_neighbor=args.n_neighbor,
        k_ratio=args.k_ratio,
        recall_prefixes=recall_prefixes,
        recall_num_shots=args.recall_num_shots
    )
    results = attack.execute(llm, train_set, test_set, cache_file=cache_file, resume=False)
    score_dict = attack.evaluate(results)
    print("results:", score_dict)
    return score_dict  # 添加返回值


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='PPL', type=str, 
                        choices=['LOSS', 'PPL', 'REFER', 'ZLIB', 'LOWER_CASE', 'WINDOW', 'LIRA', 'NEIGHBOR', 'MIN_K_PROB', 'MIN_K_PLUS_PROB', 'RECALL'])
    parser.add_argument('--num_sample', default=1000, type=int, help='use -1 to include all samples')
    parser.add_argument('--dataset', default='xsum', type=str, 
                        choices=['agnews', 'xsum', 'wikitext-103-v1', 'wikitext-2-v1', 'wikimia'])  # 添加 wikimia
    parser.add_argument('--model', default='gpt2', type=str, choices=['gpt2', 'llama2'])
    parser.add_argument('--arch', default='gpt2', type=str, help='reference model', choices=['gpt2', 'llama2'])
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--n_neighbor', default=50, type=int, help='num of neighbors in neighbor attack')
    parser.add_argument('--k_ratio', default=0.1, type=float, help='K ratio for Min-K and Min-K++ methods (default: 0.1)')
    parser.add_argument('--recall_num_shots', default=5, type=int, help='Number of prefix shots for ReCaLL metric (default: 5)')
    parser.add_argument('--recall_prefix_path', default='', type=str, help='Optional file path with prefixes for ReCaLL metric')
    parser.add_argument('--revision', default="main", type=str)
    parser.add_argument('--num_gpus_per_model', default=1, type=int)
    parser.add_argument('--max_gpu_memory', help="Maximum GPU memory used for model weights per GPU.", type=int)
    parser.add_argument('--model_path', default=' ', type=str)
    parser.add_argument('--ref_model_path', default=' ', type=str, help='Reference model path for REFER, LIRA, NEIGHBOR attacks')
    parser.add_argument('--model_name', default='gpt2', type=str, choices=['gpt2', 'llama2'])  # 添加 model_name 参数
    args = parser.parse_args()

    model, tokenizer = load_model_demo(args)
    MIA(model, tokenizer, args)
