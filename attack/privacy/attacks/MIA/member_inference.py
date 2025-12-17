import torch
import os
import zlib
import random
import numpy as np
from attack.privacy.attacks.AttackBase import AttackBase
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from enum import Enum
from tqdm import tqdm
from collections import defaultdict
from attack.privacy.models.ft_clm import FinetunedCasualLM
import torch.nn.functional as F


class MIAMetric(Enum):
    LOSS = "loss"
    PPL = "perplexity"  # the perplexity of the model.
    REFER = "refer"  # the ratio of log-perplexities of the largest GPT-2 model and the reference model.
    ZLIB = "zlib"  # the ratio of the (log) of the GPT-2 perplexity and the zlib entropy (as computed by compressing the text).
    LOWER_CASE = "lowercase"  # the ratio of perplexities of the model on the original sample and on the lowercased sample
    WINDOW = "window"  # the minimum perplexity of the model across any sliding window of 50 tokens.
    LIRA = "lira"  # https://arxiv.org/pdf/2203.03929.pdf
    NEIGHBOR = "neighbor"  # https://aclanthology.org/2023.findings-acl.719.pdf
    MIN_K_PROB = "min_k_prob"  # https://arxiv.org/pdf/2310.16789.pdf
    MIN_K_PLUS_PROB = "min_k_plus_prob"  # Min-K++ method
    RECALL = "recall"  # ReCaLL membership inference metric


class MemberInferenceAttack(AttackBase):
    def __init__(self, metric: MIAMetric, ref_model=None, n_neighbor=50, k_ratio=0.1,
                 recall_prefixes=None, recall_num_shots=0):
        # self.extraction_prompt = ["Tell me about..."]  # TODO this is just an example to extract data.
        self.metric = metric
        self.ref_model = ref_model
        self.n_neighbor = n_neighbor
        self.k_ratio = k_ratio  # K ratio for Min-K and Min-K++ methods
        self.recall_prefixes = recall_prefixes or []
        self.recall_num_shots = recall_num_shots
        self._recall_context = None

    @torch.no_grad()
    def _get_score(self, model: FinetunedCasualLM, text: str):
        """Return score. Smaller value means membership."""
        if self.metric == MIAMetric.PPL:
            ppl = model.evaluate_ppl(text)
            score = ppl
        elif self.metric == MIAMetric.LOSS:
            loss = model.evaluate(text)
            score = loss
        elif self.metric == MIAMetric.LOWER_CASE:
            ppl = model.evaluate_ppl(text)
            ref_ppl = model.evaluate_ppl(text.lower())
            score = ppl / ref_ppl
        elif self.metric == MIAMetric.WINDOW:
            # ppl = model.evaluate_ppl(text)
            assert model.tokenizer is not None
            input_ids = model.tokenizer(text, return_tensors='pt',
                                        truncation=True,
                                        max_length=model.max_seq_len).input_ids
            win_size = 50
            if len(input_ids) > win_size:
                ppls = []
                for idx in range(len(input_ids) - win_size):
                    _ppl = model.evaluate_ppl(input_ids[idx, idx + win_size], tokenized=True)
                    ppls.append(_ppl.item())
                score = np.min(ppls)
            else:
                score = model.evaluate_ppl(input_ids, tokenized=True)
        elif self.metric == MIAMetric.REFER:
            ppl = model.evaluate_ppl(text)
            ref_ppl = self.ref_model.evaluate_ppl(text)
            score = np.log(ppl) / np.log(ref_ppl)
        elif self.metric == MIAMetric.LIRA:
            # https://arxiv.org/pdf/2203.03929.pdf
            ppl = model.evaluate_ppl(text)
            ref_ppl = self.ref_model.evaluate_ppl(text)
            # score = np.log(ref_ppl) - np.log(ppl)
            score = np.log(ppl) - np.log(ref_ppl)
        elif self.metric == MIAMetric.NEIGHBOR:
            assert self.ref_model is not None, 'Neighborhood MIA requires a reference model'
            neighbor_avg = 0
            neighbors = self.ref_model.generate_neighbors(text, n=self.n_neighbor)
            for neighbor in neighbors:
                neighbor_avg += model.evaluate(neighbor)
            neighbor_avg /= len(neighbors)
            score = model.evaluate(text) - neighbor_avg
        elif self.metric == MIAMetric.ZLIB:
            ppl = model.evaluate_ppl(text)
            num_bits = len(zlib.compress(text.encode())) * 8
            score = ppl / num_bits
        elif self.metric == MIAMetric.MIN_K_PROB:
            # Get logits from model
            input_ids = model.tokenizer.encode(text, return_tensors='pt', truncation=True,
                                               max_length=model.max_seq_len).cuda()
            with torch.no_grad():
                outputs = model._lm(input_ids, labels=input_ids)
            logits = outputs[1]

            # Apply softmax to the logits to get probabilities
            probabilities = torch.nn.functional.log_softmax(logits, dim=-1).cpu().data
            all_prob = []
            input_ids_processed = input_ids[0][1:]
            for i, token_id in enumerate(input_ids_processed):
                probability = probabilities[0, i, token_id].item()
                all_prob.append(probability)

            # Calculate Min-K% Probability using configurable k_ratio
            k_length = int(len(all_prob) * self.k_ratio)
            topk_prob = np.sort(all_prob)[:k_length]
            score = -np.mean(topk_prob).item()
            if np.isnan(score):
                score = random.uniform(0.0, 1000.0)
        elif self.metric == MIAMetric.MIN_K_PLUS_PROB:
            # Get logits from model
            input_ids = model.tokenizer.encode(text, return_tensors='pt', truncation=True,
                                               max_length=model.max_seq_len).cuda()
            with torch.no_grad():
                outputs = model._lm(input_ids, labels=input_ids)
            logits = outputs[1]

            # Calculate Min-K++ scores
            input_ids_processed = input_ids[0][1:].unsqueeze(-1)
            probs = F.softmax(logits[0, :-1], dim=-1)
            log_probs = F.log_softmax(logits[0, :-1], dim=-1)
            token_log_probs = log_probs.gather(dim=-1, index=input_ids_processed).squeeze(-1)
            
            # Calculate mean and variance for normalization
            mu = (probs * log_probs).sum(-1)
            sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
            
            # Normalize token log probabilities
            mink_plus = (token_log_probs - mu) / sigma.sqrt()
            
            # Calculate Min-K++ score
            k_length = int(len(mink_plus) * self.k_ratio)
            if k_length > 0:
                topk = torch.sort(mink_plus)[0][:k_length]
                score = -torch.mean(topk).item()
            else:
                score = -torch.mean(mink_plus).item()
            
            if np.isnan(score):
                score = random.uniform(0.0, 1000.0)
        elif self.metric == MIAMetric.RECALL:
            if self._recall_context is None:
                raise RuntimeError("ReCaLL metric requires prefix context initialization before scoring.")

            unconditional_ll = self._sequence_log_likelihood(model, text)
            conditional_ll = self._conditional_log_likelihood(model, self._recall_context, text)
            denominator = unconditional_ll if unconditional_ll != 0 else 1e-8
            score = conditional_ll / denominator
            if np.isnan(score):
                score = random.uniform(0.0, 1000.0)
        else:
            raise NotImplementedError(f"{self.metric}")
        return score

    def execute(self, model, train_set, test_set, cache_file=None, resume=False):
        model._lm.eval()
        if self.metric == MIAMetric.RECALL:
            self._initialize_recall_context(test_set)
        if resume:
            if os.path.exists(cache_file):
                print(f"resume from {cache_file}")
                loaded = torch.load(cache_file)
                results = loaded['results']
                print(f"resume: i={loaded['i']}, member={loaded['member']}")
            else:
                print(f"WARN: Cann't resume. Not found {cache_file}.")
                resume = False
                results = defaultdict(list)
        else:
            results = defaultdict(list)
        if resume:
            if loaded['member'] != 1:
                print(f"Train set has been evaluated.")
                resume_i = len(train_set)
            else:
                resume_i = loaded['i']
                print(f"Resume from {resume_i + 1}/{len(test_set)}")
        else:
            resume_i = -1
        member = 1
        for i, sample in enumerate(tqdm(train_set)):
            if i <= resume_i:
                continue
            text = self._extract_text(sample)
            score = self._get_score(model, text)
            results['score'].append(score)
            results['membership'].append(member)
            if (i + 1) % 100 == 0:
                torch.save({'results': results, 'i': i, 'member': member}, cache_file)
        print(f"Train avg score: {np.mean(np.array(results['score']))}")

        test_scores = []
        member = 0

        if resume and loaded['member'] == 0:
            resume_i = loaded['i']
            print(f"Resume from {resume_i + 1}/{len(test_set)}")
        else:
            resume_i = -1
        for i, sample in enumerate(tqdm(test_set)):
            if i <= resume_i:
                continue
            text = self._extract_text(sample)
            score = self._get_score(model, text)
            results['score'].append(score)
            test_scores.append(score)
            results['membership'].append(0)
            if (i + 1) % 30 == 0:
                torch.save({'results': results, 'i': i, 'member': member}, cache_file)
        print(f"Test avg score: {np.mean(np.array(test_scores))}")
        torch.save({'results': results, 'i': -1, 'member': -1}, cache_file)
        return results

    def evaluate(self, results):
        # results['score']
        score_dict = {}
        results['score'] = np.array(results['score'])
        results['membership'] = np.array(results['membership'])
        # # follow https://arxiv.org/pdf/2203.03929.pdf
        # threshold = np.quantile(results['score'][results['membership']==0], 0.9)
        threshold = np.mean(results['score'][results['membership'] == 0])
        score_dict['nonmember_score'] = np.mean(results['score'][results['membership'] == 0])
        score_dict['member_score'] = np.mean(results['score'][results['membership'] == 1])
        # for computing AUC, you can use any threshold.
        # threshold = np.quantile(results['score'], 0.5)
        results['score'] -= threshold
        # this is for the ease of using roc_auc_score, which is equivalent to varying threshold.
        # results['score'] = 1. - 1 / (1 + np.exp(- results['score']))
        # NOTE score has to be reversed such that lower score implies membership.
        score_dict['acc'] = accuracy_score(results['membership'], results['score'] < 0)
        score_dict['auc'] = roc_auc_score(results['membership'], - results['score'])
        fpr, tpr, thresholds = roc_curve(results['membership'], - results['score'])
        score_dict[r'TPR@0.1%FPR'] = None
        for fpr_, tpr_, thr_ in zip(fpr, tpr, thresholds):
            if fpr_ > 0.001:
                score_dict[r'TPR@0.1%FPR'] = tpr_
                break

        # *************************************************************************可视化
        # auc_0 = score_dict['auc']
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 10))
        #
        # plt.plot(fpr, tpr, color='#8074C8', lw=5, label=f'AUC = {auc_0:.2f}')
        #
        # # 绘制对角线
        # plt.plot([0, 1], [0, 1], color='#EC6E66', lw=5, linestyle='--')
        #
        # # 设置坐标轴限制
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        #
        # # 确定共同的横坐标点
        # common_x = np.linspace(0, 1, 10)  # 这里选择了10个均匀分布的点，可按需调整
        #
        # # 找到每条曲线在共同横坐标点上对应的纵坐标
        # def find_closest_y(x_vals, y_vals, common_x):
        #     closest_y = []
        #     for x in common_x:
        #         idx = np.argmin(np.abs(x_vals - x))
        #         closest_y.append(y_vals[idx])
        #     return np.array(closest_y)
        #
        # closest_y_0 = find_closest_y(fpr, tpr, common_x)
        #
        # # 绘制三角形标记
        # plt.scatter(common_x, closest_y_0, marker='s', color='#FFFFFF', s=100, edgecolor='#629C35', zorder=3)
        #
        # ax = plt.gca()
        # ax.set_facecolor('#f0f0f0')
        #
        # for spine in ax.spines.values():
        #     spine.set_linewidth(4)  # Set border width
        #
        # plt.xlabel('False Positive Rate', fontsize=40)
        # plt.ylabel('True Positive Rate', fontsize=40)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        #
        # plt.grid(True, color='white', linestyle='-', linewidth=4, zorder=1)
        # plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        # plt.legend(loc='lower right', fontsize=24)
        #
        # save_path = './Fig'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # filename = os.path.join(save_path, f"ROC.pdf")
        # plt.savefig(filename)
        #
        # plt.show()
        # plt.close()


        # def visualize_score_distribution(results):
        #     # 确保 membership 和 score 是 numpy 数组
        #     scores = np.array(results['score'])
        #     membership = np.array(results['membership'])
        #
        #     # 分离 member 和 nonmember 的分数
        #     member_scores = scores[membership == 1]
        #     nonmember_scores = scores[membership == 0]
        #
        #     member_mean = np.mean(member_scores)
        #     nonmember_mean = np.mean(nonmember_scores)
        #
        #     plt.figure(figsize=(10, 10))
        #
        #     # 设置背景颜色为浅浅灰色
        #     ax = plt.gca()
        #     ax.set_facecolor('#f0f0f0')
        #
        #     # Create histograms with white borders
        #     plt.hist(member_scores, bins=50, color='#7AB656', alpha=0.5, label='Member Scores',
        #              edgecolor='white', linewidth=2, zorder=2)
        #     plt.hist(nonmember_scores, bins=50, color='#DBB428', alpha=0.5, label='Hold-out Scores',
        #              edgecolor='white', linewidth=2, zorder=2)
        #
        #     plt.xlabel('Scores', fontsize=44)
        #     plt.ylabel('Number of Samples', fontsize=44)
        #
        #     # Bold the borders of the plot
        #     for spine in ax.spines.values():
        #         spine.set_linewidth(4)  # Set border width
        #
        #     # Mark the mean positions
        #     plt.axvline(member_mean, color='green', linestyle='dashed', linewidth=4, zorder=3)
        #     plt.axvline(nonmember_mean, color='orange', linestyle='dashed', linewidth=4, zorder=3)
        #
        #     # 添加白色网格
        #     plt.grid(True, color='white', linestyle='-', linewidth=3, zorder=1)
        #     plt.xticks(fontsize=24)
        #     plt.yticks(fontsize=24)
        #
        #     plt.legend(fontsize=32, frameon=True, borderpad=1, loc='upper right')
        #     plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
        #
        #     save_path = './Fig'
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     filename = os.path.join(save_path, f"Distribution.pdf")
        #     plt.savefig(filename)
        #     plt.show()
        #     plt.close()
        #
        # visualize_score_distribution(results)
        # # *************************************************************************可视化


        return score_dict

    def _extract_text(self, sample):
        if isinstance(sample, str):
            return sample
        if isinstance(sample, dict):
            for key in ('prompt', 'text', 'sentence', 'content'):
                value = sample.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        if isinstance(sample, (list, tuple)) and sample:
            first = sample[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                return self._extract_text(first)
        return str(sample)

    def _initialize_recall_context(self, fallback_samples):
        if self._recall_context is not None:
            return
        candidates = []
        if self.recall_prefixes:
            candidates = [c for c in self.recall_prefixes if isinstance(c, str) and c.strip()]
        else:
            for sample in fallback_samples:
                text = self._extract_text(sample)
                if text:
                    candidates.append(text)
                if self.recall_num_shots > 0 and len(candidates) >= self.recall_num_shots:
                    break
        if not candidates:
            raise RuntimeError("Unable to initialize ReCaLL prefix context. Please provide non-empty prefixes.")
        num_shots = self.recall_num_shots if self.recall_num_shots > 0 else min(len(candidates), 5)
        self._recall_context = "\n".join(candidates[:num_shots])

    def _sequence_log_likelihood(self, model: FinetunedCasualLM, text: str):
        input_ids = model.tokenizer.encode(text, return_tensors='pt', truncation=True,
                                           max_length=model.max_seq_len).to('cuda')
        with torch.no_grad():
            outputs = model._lm(input_ids, labels=input_ids)
        logits = outputs.logits
        token_log_probs = self._token_log_probability(logits, input_ids)
        return token_log_probs.sum().item()

    def _conditional_log_likelihood(self, model: FinetunedCasualLM, prefix: str, target: str):
        tokenizer = model.tokenizer
        prefix_ids = tokenizer.encode(prefix, return_tensors='pt', truncation=True,
                                      max_length=model.max_seq_len).to('cuda')
        target_ids = tokenizer.encode(target, return_tensors='pt', truncation=True,
                                      max_length=model.max_seq_len).to('cuda')
        if target_ids.shape[1] > 1:
            combined_ids = torch.cat([prefix_ids, target_ids[:, 1:]], dim=1)
        else:
            combined_ids = torch.cat([prefix_ids, target_ids], dim=1)
        prefix_len = prefix_ids.shape[1]
        overflow = max(0, combined_ids.shape[1] - model.max_seq_len)
        if overflow > 0:
            combined_ids = combined_ids[:, overflow:]
            prefix_len = max(0, prefix_len - overflow)
        labels = combined_ids.clone()
        labels[:, :prefix_len] = -100
        with torch.no_grad():
            outputs = model._lm(combined_ids, labels=labels)
        logits = outputs.logits
        token_log_probs = self._token_log_probability(logits, combined_ids)
        mask = labels[:, 1:] != -100
        masked = token_log_probs.masked_select(mask)
        return masked.sum().item()

    def _token_log_probability(self, logits, input_ids):
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        tokens = input_ids[:, 1:]
        gathered = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
        return gathered
