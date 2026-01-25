import csv
import json
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
import os
import time
# import openai
from openai import OpenAI
import threading


api_key = 'sk-xxx替换为自己的key'
api_base = 'https://xiaoai.plus/v1'

def get_image_file_location(root, row):
    if int(row['visual_input']) == 0:
        return None
    img_file = row['set_id'] + "_" + row['figure_id'] + ".png"
    return os.path.join(root, row['category'], row['subcategory'], img_file)



def evaluate_by_chatgpt(data, output_entry, correctness_entry, gpt_model="gpt-4", load_json=False, save_json_path="./hallusion_result.jsons"):
    if load_json and os.path.exists(save_json_path):
        with open(save_json_path, 'r') as f:
            output = json.load(f)
    else:
        output = []
    for sample in tqdm(data[len(output):]):
        prompt = 'Imagine you are an intelligent teacher. Thoroughly read the question, reference answer and the prediction answer to ensure a clear understanding of the information provided. Assess the correctness of the predictions. '
        prompt += 'If the prediction answer does not conflict with the reference answer, please generate "correct". If the prediction answer conflict with the reference answer, please generate "incorrect". If the prediction answer is unclear about the answer, please generate "unclear". \n\n Question:'
        prompt += sample['question']
        prompt += '\nReference answer: '
        prompt += sample['gt_answer_details']
        prompt += '\nPrediction answer:'
        prompt += sample[output_entry]
        prompt += '\nOutput:'

        # https://github.com/openai/openai-python/issues/322#issuecomment-1767841683
        while True:
            try:
                client=OpenAI(
                    base_url=api_base, #·sk-xxx替换为自己的key
                    api_key=api_key
                )
                completion=client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                output_text = completion.choices[0].message.content
                break
            except:
                print("Timeout, retrying...")
                time.sleep(5)  # Wait for 5 seconds before retrying


        if 'incorrect' in output_text.lower(): 
            gpt_correctness = "0"

        elif 'correct' in output_text.lower():
            gpt_correctness = "1"
        else:
            gpt_correctness = "2"

        sample[correctness_entry] = gpt_correctness

        output.append(sample)

        with open(save_json_path, 'w') as f:
            json.dump(output, f)

    return output

def check_same_by_chatgpt(data, output_entry, gpt_model="gpt-4", load_json=False, save_json_path="./hallusion_result.json"):

    for sample in tqdm(data):
        if "same" not in sample.keys():
            prompt = 'Imagine you are an intelligent teacher. Thoroughly read the two responses to two different questions. Assess the consistency of the information provided within those two responses. '
            prompt += 'You do not know the specific questions, but you can asssess the consistency among the two responses by checking for logical conflicts if both responses are correct. '
            prompt += 'If response1 does not conflict with response2, please generate "same". Otherwise, generate "different". \n\n response1:'
            prompt += sample['gt_answer_details']
            prompt += '\nresponse2: '
            prompt += sample[output_entry]
            prompt += '\nOutput:'

            # https://github.com/openai/openai-python/issues/322#issuecomment-1767841683
            while True:
                try:
                    client=OpenAI(
                        base_url=api_base, #·sk-xxx替换为自己的key
                        api_key=api_key
                    )
                    completion=client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    output_text = completion.choices[0].message.content
                    break
                except:
                    print("Timeout, retrying...")
                    time.sleep(5)  # Wait for 5 seconds before retrying

            gpt_same = "0"

            if 'same' in output_text.lower(): 
                gpt_same = "1"

            elif 'different' in output_text.lower():
                gpt_same = "0"


            sample["same"] = gpt_same

            with open(save_json_path, 'w') as f:
                json.dump(data, f)

    return data

def get_eval_all(domain, data, model_correctness_entry): # per question

    eval_all_dict = dict()
    eval_all_stat = {}
    eval_all_stat["LH"] = 0
    eval_all_stat["VI"] = 0
    eval_all_stat["Mix"] = 0

    for r in data:
        if domain == "hallusionbench":
            name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"]), str(r["question_id"])])
            assert name not in eval_all_dict 
            
            eval_all_dict[name] = r["correct"]
            
            if str(r["category"]) == "VD": # VD
                if str(r["figure_id"]) == "0":
                    if str(r[model_correctness_entry]) == "0" or str(r[model_correctness_entry]) == "2":
                        eval_all_stat["VI"] += 1
                else:
                    if str(r[model_correctness_entry]) == "0":
                        eval_all_stat["Mix"] += 1
                    elif str(r[model_correctness_entry]) == "2":
                        eval_all_stat["VI"] += 1
            else: # VS
                if str(r["visual_input"]) == "0": # no visual
                    if str(r[model_correctness_entry]) == "0":
                        eval_all_stat["LH"] += 1
                else: # original visual or modified visual (isual_input == 1 or 2)
                    if str(r[model_correctness_entry]) == "0":
                        eval_all_stat["Mix"] += 1
                    elif str(r[model_correctness_entry]) == "2":
                        eval_all_stat["VI"] += 1
        
        else:
            print(r)
            name = str(r["question_id"])
            assert name not in eval_all_dict 
            
            eval_all_dict[name] = r["correct"]

    eval_all_stat["note"] = "all accuracy per question"
    eval_all_stat["total"] = len(eval_all_dict.keys())
    eval_all_stat["correct"] = np.count_nonzero(list(eval_all_dict.values()))
    return eval_all_stat

def assign_correctness(data_arr, correctness_entry):
    for r in data_arr:
        assert int(r[correctness_entry]) == 0 or int(r[correctness_entry]) == 1 or int(r[correctness_entry]) == 2
        r["correct"] = 1 if int(r[correctness_entry]) == 1 else 0

    return data_arr

