import spacy
import bert_score
import numpy as np
import torch
import re
from tqdm import tqdm
import os
from typing import Dict, List, Set, Tuple, Union, Any
from transformers import logging
from openai import OpenAI
logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LongformerTokenizer, LongformerForMultipleChoice, LongformerForSequenceClassification
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from .utils import MQAGConfig, expand_list1, expand_list2, NLIConfig, LLMPromptConfig
from .modeling_mqag import question_generation_sentence_level, answering
from .modeling_ngram import UnigramModel, NgramModel

# ---------------------------------------------------------------------------------------- #
# Functions for counting
def method_simple_counting(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    AT,
):
    """
    simple counting method score => count_mismatch / (count_match + count_mismatch)
    :return score: 'inconsistency' score
    """
    # bad questions, i.e. not answerable given the passage
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_good_sample, count_match = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            count_good_sample += 1
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
    if count_good_sample == 0:
        score = 0.5
    else:
        score = (count_good_sample-count_match) / count_good_sample
    return score

def method_vanilla_bayes(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    beta1, beta2, AT,
):
    """
    (vanilla) bayes method score: compute P(sentence is non-factual | count_match, count_mismatch)
    :return score: 'inconsistency' score
    """
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
            else:
                count_mismatch += 1
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def method_bayes_with_alpha(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    beta1, beta2,
):
    """
    bayes method (with answerability score, i.e. soft-counting) score
    :return score: 'inconsistency' score
    """
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        ans_score = u_score_s[s]
        a_S = np.argmax(prob_s[s])
        if a_DT == a_S:
            count_match += ans_score
        else:
            count_mismatch += ans_score
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def answerability_scoring(
    u_model,
    u_tokenizer,
    question,
    context,
    max_length,
    device,
):
    """
    :return prob: prob -> 0.0 means unanswerable, prob -> 1.0 means answerable
    """
    input_text = question + ' ' + u_tokenizer.sep_token + ' ' + context
    inputs = u_tokenizer(input_text, max_length=max_length, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)
    logits = u_model(**inputs).logits
    logits = logits.squeeze(-1)
    prob = torch.sigmoid(logits).item()
    return prob

class SelfCheckMQAG:
    """
    SelfCheckGPT (MQAG variant) using OpenAI API for all components
    """
    def __init__(
        self,
        g1_model: str = "gpt-3.5-turbo",
        g2_model: str = "gpt-3.5-turbo",
        answering_model: str = "gpt-3.5-turbo",
        answerability_model: str = "gpt-3.5-turbo",
        max_tokens: int = 512,
        temperature: float = 0.3
    ):
        """
        Initialize with OpenAI API settings
        
        :param g1_model: OpenAI model for first-stage question generation
        :param g2_model: OpenAI model for second-stage question generation
        :param answering_model: OpenAI model for answering questions
        :param answerability_model: OpenAI model for answerability scoring
        :param api_key: Your OpenAI API key
        :param max_tokens: Max tokens for API responses
        :param temperature: Creativity level (0-2)
        """
        self.client = OpenAI(
            base_url="https://zzzzapi.com/v1",
            api_key="sk-3vW4zNs9yLa69NK5QrSIRCc0l8fpdelO7nS3gMOkSItyY5jz"
        )
        self.g1_model = g1_model
        self.g2_model = g2_model
        self.answering_model = answering_model
        self.answerability_model = answerability_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        print("SelfCheck-MQAG initialized with OpenAI API")

    def _call_openai(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Dict[str, str] = None,
        is_json: bool = False
    ) -> Any:
        """Generic OpenAI API call handler"""
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        if is_json:
            params["response_format"] = {"type": "json_object"}
        
        try:
            response = self.client.chat.completions.create(**params)
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""

    def predict(
        self,
        sentences: List[str],
        passage: str,
        sampled_passages: List[str],
        num_questions_per_sent: int = 5,
        scoring_method: str = "bayes_with_alpha",
        **kwargs,
    ) -> np.ndarray:
        """
        Evaluate sentences against sampled passages using OpenAI API
        
        :param sentences: Sentences to evaluate (split from passage)
        :param passage: Original full passage
        :param sampled_passages: LLM-generated reference passages
        :param num_questions_per_sent: Questions to generate per sentence
        :param scoring_method: Scoring algorithm (counting/bayes/bayes_with_alpha)
        :return: Sentence-level inconsistency scores
        """
        assert scoring_method in ['counting', 'bayes', 'bayes_with_alpha']
        num_samples = len(sampled_passages)
        sent_scores = []
        
        for sentence in sentences:
            # Generate questions for this sentence
            questions = self.question_generation_sentence_level(
                sentence, passage, num_questions_per_sent
            )
            
            scores = []
            for question_item in questions:
                question = question_item['question']
                options = question_item['options']
                
                # Get probabilities for original passage
                prob = self.answering(question, options, passage)
                u_score = self.answerability_scoring(question, passage)
                
                # Get probabilities for sampled passages
                prob_s = np.zeros((num_samples, 4))
                u_score_s = np.zeros((num_samples,))
                
                for si, sampled_passage in enumerate(sampled_passages):
                    prob_s[si] = self.answering(question, options, sampled_passage)
                    u_score_s[si] = self.answerability_scoring(question, sampled_passage)
                
                # Calculate inconsistency score
                if scoring_method == 'counting':
                    score = self.method_simple_counting(
                        prob, u_score, prob_s, u_score_s, num_samples, AT=kwargs.get('AT', 0.5)
                    )
                elif scoring_method == 'bayes':
                    score = self.method_vanilla_bayes(
                        prob, u_score, prob_s, u_score_s, num_samples,
                        beta1=kwargs.get('beta1', 0.5), 
                        beta2=kwargs.get('beta2', 0.5),
                        AT=kwargs.get('AT', 0.5)
                    )
                elif scoring_method == 'bayes_with_alpha':
                    score = self.method_bayes_with_alpha(
                        prob, u_score, prob_s, u_score_s, num_samples,
                        beta1=kwargs.get('beta1', 0.5), 
                        beta2=kwargs.get('beta2', 0.5)
                    )
                scores.append(score)
            
            sent_scores.append(np.mean(scores))
        
        return np.array(sent_scores)

    def question_generation_sentence_level(
        self,
        sentence: str,
        passage: str,
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple-choice questions using OpenAI API
        
        :param sentence: Target sentence to generate questions about
        :param passage: Full context passage
        :param num_questions: Number of questions to generate
        :return: List of question dictionaries with options
        """
        prompt = f"""
        Generate {num_questions} multiple-choice questions based on the following sentence:
        "{sentence}"
        
        The full context is:
        "{passage}"
        
        For each question:
        1. Create a clear question about information specifically from the target sentence
        2. Provide 4 plausible options (A, B, C, D)
        3. Mark the correct answer
        4. Return in JSON format: {{"questions": [{{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "correct": "A"}}]}}
        """
        
        response = self._call_openai(
            self.g1_model,
            [{"role": "user", "content": prompt}],
            is_json=True
        )
        
        try:
            import json
            data = json.loads(response)
            return data.get('questions', [])
        except:
            print("Failed to parse question generation response")
            return []

    def answering(
        self,
        question: str,
        options: List[str],
        context: str
    ) -> np.ndarray:
        """
        Get answer probabilities using OpenAI API
        
        :param question: Question text
        :param options: List of 4 options
        :param context: Passage to search for answers
        :return: Probability distribution over options [pA, pB, pC, pD]
        """
        option_text = "\n".join(options)
        prompt = f"""
        Context: {context}
        
        Question: {question}
        Options:
        {option_text}
        
        Analyze the context and determine the most likely correct answer.
        Return ONLY the letter of the correct option (A, B, C, or D).
        """
        
        response = self._call_openai(
            self.answering_model,
            [{"role": "user", "content": prompt}]
        )
        
        # Parse response to get probability distribution
        prob = np.zeros(4)
        if response in ['A', 'B', 'C', 'D']:
            index = ord(response) - ord('A')
            prob[index] = 1.0
        else:  # Fallback if response format unexpected
            prob = np.ones(4) / 4
            
        return prob

    def answerability_scoring(
        self,
        question: str,
        context: str
    ) -> float:
        """
        Get answerability score using OpenAI API
        
        :param question: Question text
        :param context: Passage to evaluate
        :return: Answerability score between 0-1
        """
        prompt = f"""
        Context: {context}
        Question: {question}
        
        On a scale from 0 to 1, how answerable is the question based on the context?
        - 0: The context provides no information to answer the question
        - 0.5: The context provides partial but insufficient information
        - 1: The context contains clear information to answer the question
        
        Return ONLY a numerical score between 0 and 1 with one decimal place.
        """
        
        response = self._call_openai(
            self.answerability_model,
            [{"role": "user", "content": prompt}]
        )
        
        try:
            score = float(response)
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default score if parsing fails

    # ================== Scoring Methods ================== #
    def method_simple_counting(
        self,
        prob, u_score, prob_s, u_score_s, num_samples, AT=0.5
    ):
        """Simple counting-based scoring"""
        count = 0
        for si in range(num_samples):
            if u_score_s[si] >= AT:
                if np.argmax(prob) != np.argmax(prob_s[si]):
                    count += 1
        return count / num_samples

    def method_vanilla_bayes(
        self,
        prob, u_score, prob_s, u_score_s, num_samples, beta1=0.5, beta2=0.5, AT=0.5
    ):
        """Bayesian scoring with smoothing"""
        alpha = 1.0
        for si in range(num_samples):
            if u_score_s[si] >= AT:
                p_obs = prob_s[si][np.argmax(prob)] + beta1
                p_obs /= (1 + 4 * beta1)
                alpha *= p_obs
        return 1 - (alpha ** (1/num_samples))

    def method_bayes_with_alpha(
        self,
        prob, u_score, prob_s, u_score_s, num_samples, beta1=0.5, beta2=0.5
    ):
        """Advanced Bayesian scoring with alpha weighting"""
        alpha = 1.0
        for si in range(num_samples):
            p_obs = np.dot(prob, prob_s[si]) + beta1
            p_obs /= (1 + 4 * beta1)
            weight = (u_score * u_score_s[si]) ** beta2
            alpha *= (p_obs ** weight)
        return 1 - (alpha ** (1/num_samples))

class SelfCheckBERTScore:
    """
    SelfCheckGPT (BERTScore variant): Checking LLM's text against its own sampled texts via BERTScore (against best-matched sampled sentence)
    """
    def __init__(self, default_model="/home/hub/model/roberta-large", rescale_with_baseline=True, baseline_path='/home/jinhong_chen/AwesomeLLMSecurityPlatform/modules/hallucination/llm/selfcheckgpt/roberta-large.tsv'):
        """
        :default_model: model for BERTScore
        :rescale_with_baseline:
            - whether or not to rescale the score. If False, the values of BERTScore will be very high
            - this issue was observed and later added to the BERTScore package,
            - see https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.default_model = default_model # en => roberta-large
        self.rescale_with_baseline = rescale_with_baseline
        self.baseline_path = baseline_path
        print("SelfCheck-BERTScore initialized")

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :return sent_scores: sentence-level score which is 1.0 - bertscore
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        bertscore_array = np.zeros((num_sentences, num_samples))
        for s in range(num_samples):
            sample_passage = sampled_passages[s]
            sentences_sample = [sent for sent in self.nlp(sample_passage).sents] # List[spacy.tokens.span.Span]
            sentences_sample = [sent.text.strip() for sent in sentences_sample if len(sent) > 3]
            num_sentences_sample  = len(sentences_sample)

            refs  = expand_list1(sentences, num_sentences_sample) # r1,r1,r1,....
            cands = expand_list2(sentences_sample, num_sentences) # s1,s2,s3,...

            P, R, F1 = bert_score.score(
                    cands, refs,
                    model_type=self.default_model, verbose=False, lang='en', num_layers=17, baseline_path=self.baseline_path,
                    rescale_with_baseline=self.rescale_with_baseline,
            )
            F1_arr = F1.reshape(num_sentences, num_sentences_sample)
            F1_arr_max_axis1 = F1_arr.max(axis=1).values
            F1_arr_max_axis1 = F1_arr_max_axis1.numpy()

            bertscore_array[:,s] = F1_arr_max_axis1

        bertscore_mean_per_sent = bertscore_array.mean(axis=-1)
        one_minus_bertscore_mean_per_sent = 1.0 - bertscore_mean_per_sent
        return one_minus_bertscore_mean_per_sent


class SelfCheckNgram:
    def __init__(self, n: int, lowercase: bool = True):
        self.n = n
        self.lowercase = lowercase
        print(f"SelfCheck-{n}gram initialized")

    def predict(
        self,
        sentences: List[str],
        passage: str,
        sampled_passages: List[str],
    ):
        if self.n == 1:
            ngram_model = UnigramModel(lowercase=self.lowercase)
        elif self.n > 1:
            ngram_model = NgramModel(n=self.n, lowercase=self.lowercase)
        else:
            raise ValueError("n must be >= 1")

        ngram_model.add(passage)
        for sampled_passage in sampled_passages:
            ngram_model.add(sampled_passage)
        ngram_model.train(k=0)

        stats = ngram_model.evaluate(sentences)
        sent_scores = np.array(stats["sent_level"]["max_neg_logprob"], dtype=float)
        return sent_scores


class SelfCheckNLI:
    """
    SelfCheckGPT (NLI variant): Checking LLM's text against its own sampled texts via DeBERTa-v3 finetuned to Multi-NLI
    """
    def __init__(
        self,
        nli_model: str = None,
        device = None
    ):
        nli_model = nli_model if nli_model is not None else NLIConfig.nli_model
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(nli_model)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(nli_model)
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        print("SelfCheck-NLI initialized to device", device)

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :return sent_scores: sentence-level score which is P(condict|sentence, sample)
        note that we normalize the probability on "entailment" or "contradiction" classes only
        and the score is the probability of the "contradiction" class
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        for sent_i, sentence in enumerate(sentences):
            for sample_i, sample in enumerate(sampled_passages):
                inputs = self.tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=[(sentence, sample)],
                    add_special_tokens=True, padding="longest",
                    truncation=True, return_tensors="pt",
                    return_token_type_ids=True, return_attention_mask=True,
                )
                inputs = inputs.to(self.device)
                logits = self.model(**inputs).logits # neutral is already removed
                probs = torch.softmax(logits, dim=-1)
                prob_ = probs[0][1].item() # prob(contradiction)
                scores[sent_i, sample_i] = prob_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence


class SelfCheckLLMPrompt:
    """
    SelfCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via open-source LLM prompting
    """
    def __init__(
        self,
        model_path: str = None,
        device = None
    ):
        model_path = model_path if model_path is not None else LLMPromptConfig.model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
        self.model.eval()
        if device is None:
            device = torch.device("cpu")
        self.model.to(device)
        self.device = device
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"SelfCheck-LLMPrompt ({os.path.basename(model_path)}) initialized to device {device}")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ") 

                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                generate_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,
                    do_sample=False, # hf's default for Llama2 is True
                )
                output_text = self.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )[0]
                generate_text = output_text.replace(prompt, "")
                score_ = self.text_postprocessing(generate_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        # tested on Llama-2-chat (7B, 13B) --- this code has 100% coverage on wikibio gpt3 generated data
        # however it may not work with other datasets, or LLMs
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]