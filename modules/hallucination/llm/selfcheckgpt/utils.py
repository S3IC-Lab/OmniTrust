import numpy as np
import torch
import os
import jsonlines
import spacy
from openai import OpenAI
import random
from tqdm import tqdm
import json

ENTITY_LIST = [
    "Marie Curie", 
    "Albert Einstein", 
    "Nikola Tesla", 
    "Stephen Hawking", 
    "Rosalind Franklin", 
    "Charles Darwin", 
    "Isaac Newton", 
    "Jane Goodall", 
    "Richard Feynman", 
    "Ada Lovelace", 
    "Niels Bohr", 
    "Dmitri Mendeleev", 
    "Michael Faraday", 
    "Hedy Lamarr", 
    "Tu Youyou", 
    "Yoshinori Ohsumi", 
    "Yamanaka Shinya", 
    "Fei-Fei Li", 
    "Vera Rubin", 
    "James Watson", 
    "Emmanuelle Charpentier", 
    "Jennifer Doudna", 
    "Frances Arnold", 
    "Gregor Mendel", 
    "Barbara McClintock",
    "Nelson Mandela", 
    "Mahatma Gandhi", 
    "Angela Merkel", 
    "Franklin D. Roosevelt", 
    "Winston Churchill", 
    "Margaret Thatcher", 
    "Barack Obama", 
    "Aung San Suu Kyi", 
    "Julius Nyerere", 
    "Abraham Lincoln", 
    "Indira Gandhi", 
    "Golda Meir", 
    "John F. Kennedy", 
    "Vladimir Putin", 
    "Fidel Castro", 
    "Lee Kuan Yew", 
    "Otto von Bismarck", 
    "Theodore Roosevelt", 
    "Jiang Zemin", 
    "Benito Mussolini", 
    "Dilma Rousseff", 
    "Ellen Johnson Sirleaf", 
    "Kamala Harris",
    "Leonardo da Vinci", 
    "Vincent van Gogh", 
    "Pablo Picasso", 
    "Frida Kahlo", 
    "Maya Angelou", 
    "Claude Monet", 
    "Igor Stravinsky", 
    "Louis Armstrong", 
    "Billie Holiday", 
    "Banksy", 
    "Anton Chekhov", 
    "Shakespeare", 
    "Georgia O’Keeffe", 
    "Ai Weiwei", 
    "Hayao Miyazaki", 
    "Takashi Murakami", 
    "Yoko Ono", 
    "Zaha Hadid", 
    "Damien Hirst", 
    "Bob Dylan",
    "Steve Jobs", 
    "Bill Gates", 
    "Elon Musk", 
    "Mark Zuckerberg", 
    "Warren Buffett", 
    "Richard Branson", 
    "Larry Page", 
    "Sergey Brin", 
    "Jack Ma", 
    "Mukesh Ambani", 
    "Tim Berners-Lee", 
    "Jeff Bezos", 
    "Reed Hastings", 
    "Howard Schultz", 
    "Ratan Tata",
    "Malala Yousafzai", 
    "Greta Thunberg", 
    "Emma Watson", 
    "Dwayne Johnson", 
    "Rihanna", 
    "Beyoncé", 
    "Taylor Swift", 
    "Cristiano Ronaldo", 
    "Lionel Messi", 
    "Kim Kardashian", 
    "Oprah Winfrey", 
    "Ellen DeGeneres", 
    "Michelle Obama", 
    "BTS", 
    "Jackie Chan"
]

def data_generation(args):
    selected_entities = random.sample(
        ENTITY_LIST,
        k=min(args.n_samples, len(ENTITY_LIST))
    )

    n_repeats = getattr(args, "n_repeats", 5)
    bios = []
    multi_bios = []
    
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    try:
        for entity in tqdm(selected_entities, desc="Generating data", total=len(selected_entities)):
            entity_bios = []

            for _ in range(n_repeats):
                prompt = (
                    f"Tell me a detailed bio of {entity}. "
                    f"Include information about their early life, major achievements, and lasting impact."
                )
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                ).to(model.device)

                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

                if prompt in full_text:
                    text = full_text.split(prompt, 1)[1].strip()
                else:

                    text = full_text
                entity_bios.append(text)

            bios.append(entity_bios[0])
            multi_bios.append(entity_bios[1:])

    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, f"qwen_entity_bios_.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for bio, extra in zip(bios, multi_bios):
            record = {
                "bio": bio,
                "extra_bios": extra
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return bios, multi_bios


class MQAGConfig:
    generation1_squad: str = "potsawee/t5-large-generation-squad-QuestionAnswer"
    generation1_race: str = "potsawee/t5-large-generation-race-QuestionAnswer"
    generation2: str = "potsawee/t5-large-generation-race-Distractor"
    answering: str = "potsawee/longformer-large-4096-answering-race"
    answerability: str = "potsawee/longformer-large-4096-answerable-squad2"

class NLIConfig:
    nli_model: str = "potsawee/deberta-v3-large-mnli"

class LLMPromptConfig:
    model: str = "meta-llama/Llama-2-7b-chat-hf"

# Question Generation & Answering Input Processing
def prepare_qa_input(t5_tokenizer, context, device):
    """
    input: context
    output: question <sep> answer
    """
    encoding = t5_tokenizer(
        [context],
        return_tensors="pt",
    )
    input_ids = encoding.input_ids.to(device)
    return input_ids


def prepare_distractor_input(t5_tokenizer, context, question, answer, device, separator='<sep>'):
    """
    input: question <sep> answer <sep> article
    output: distractor1 <sep> distractor2 <sep> distractor3
    """
    input_text = question + ' ' + separator + ' ' + answer + ' ' + separator + ' ' + context
    encoding = t5_tokenizer(
        [input_text],
        return_tensors="pt",
    )
    input_ids = encoding.input_ids.to(device)
    return input_ids


def prepare_answering_input(
    tokenizer, # longformer_tokenizer
    question, options, context,
    device, max_seq_length=4096,
):
    c_plus_q = context + ' ' + tokenizer.bos_token + ' ' + question
    c_plus_q_4 = [c_plus_q] * len(options)

    tokenized_examples = tokenizer(
        c_plus_q_4, options,
        max_length=max_seq_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    tokenized_examples = tokenized_examples.to(device)
    input_ids = tokenized_examples['input_ids'].unsqueeze(0)
    attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)

    example_encoded = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    return example_encoded

# SelfCheck - BERTScore utils
def expand_list1(mylist, num):
    expanded = []
    for x in mylist:
        for _ in range(num):
            expanded.append(x)
    return expanded

def expand_list2(mylist, num):
    expanded = []
    for _ in range(num):
        for x in mylist:
            expanded.append(x)
    return expanded

# MQAG score utils
def smoothing(probs):
    probs = probs + 1e-12
    probs = probs / probs.sum()
    return probs

def kl_div(probs1, probs2):
    assert len(probs1) == len(probs2)
    probs1 = smoothing(probs1)
    probs2 = smoothing(probs2)
    xx = probs1 * np.log(probs1 / probs2)
    return xx.sum()

def onebest_argmax(probs1, probs2):
    answer1 = probs1.argmax()
    answer2 = probs2.argmax()
    if answer1 == answer2:
        count = 0
    else:
        count = 1
    return count

def hellinger_dist(probs1, probs2):
    # https://en.wikipedia.org/wiki/Hellinger_distance
    sqrt_p1 = np.sqrt(probs1)
    sqrt_p2 = np.sqrt(probs2)
    return ((sqrt_p1 - sqrt_p2)**2).sum(axis=-1) / 1.4142135

def total_variation(probs1, probs2):
    diff = np.abs(probs1 - probs2)
    return diff.max()

def get_prob_distances(probs1, probs2):
    kl = kl_div(probs1, probs2)
    ob = onebest_argmax(probs1, probs2)
    hl = hellinger_dist(probs1, probs2)
    tv = total_variation(probs1, probs2)
    return kl, ob, hl, tv
