import argparse
import os
from typing import Dict, List
import numpy as np
from .data.data_processor import (
    read_json,
    get_cleaned_final_answer,
    get_answers_from_dict,
    get_answers_from_list,
)


def convert_np_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_np_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_floats(item) for item in obj]
    elif isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj


def compute_metrics_for_open_answer(
    answers: List[str], true_answers: List[str]
) -> Dict[str, float]:
    precision_list = []
    recall_list = []
    f1_score_list = []

    for answer, true_answer in zip(answers, true_answers):
        answer = set(answer.split(" "))
        true_answer = set(true_answer.split(" "))

        tp = len(answer.intersection(true_answer))
        fp = len(answer.difference(true_answer))
        fn = len(true_answer.difference(answer))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall) if tp > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

    return {
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1_score": np.mean(f1_score_list),
    }


# def compute_metrics_for_list_answer(
#     answers: List[List[str]], true_answers: List[List[str]]
# ) -> Dict[str, float]:
#     positive_percentages = []
#     negative_percentages = []
#     precision_numerator = 0
#     precision_denominator = 0
    
#     for answer, true_answer in zip(answers, true_answers):
#         total_items = len(answer)
#         if total_items == 0:
#             positive_percentages.append(0.0)
#             negative_percentages.append(100.0)
#             continue
            
#         positive = 0
#         for item in answer:
#             if item in true_answer:
#                 positive += 1
        
#         negative = total_items - positive
 
#         positive_pct = positive / total_items
#         negative_pct = negative / total_items
        
#         positive_percentages.append(positive_pct)
#         negative_percentages.append(negative_pct)

#         precision_numerator += positive
#         precision_denominator += total_items

#     precision_pct = 0.0
#     if precision_denominator > 0:
#         precision_pct = precision_numerator / precision_denominator

#     return {
#         "avg_correct_pct": np.mean(positive_percentages),
#         "avg_incorrect_pct": np.mean(negative_percentages),
#         "overall_precision_pct": precision_pct,
#     }

def compute_metrics_for_list_answer(
    answers: List[List[str]], true_answers: List[List[str]]
) -> Dict[str, float]:
    positive_answers = []
    negative_answers = []
    for answer, true_answer in zip(answers, true_answers):
        positive = 0
        negative = 0
        for item in answer:
            if item in true_answer:
                positive += 1
            else:
                negative += 1
        positive_answers.append(positive)
        negative_answers.append(negative)

    tp = np.sum(positive_answers)
    fp = np.sum(negative_answers)

    return {
        "positive_avg": np.mean(positive_answers),
        "negative_avg": np.mean(negative_answers),
        "precision": tp / (tp + fp),
    }


def evaluate(result_path: str, dataset_path: str, dataset_type: str):
    if not (os.path.exists(dataset_path) and os.path.exists(result_path)):
        raise ValueError("Dataset or results path does not exist.")
    dataset = read_json(dataset_path)
    results = read_json(result_path)

    if dataset_type == "wikidata":
        true_answers = get_answers_from_dict(dataset)
        true_answers = [element[1] for element in true_answers]

    elif dataset_type == "wikidata_category" or dataset_type == "multispan_qa":
        true_answers = get_answers_from_list(dataset)
    
    if dataset_type == "multispan_qa":
        answers = [result["Final Refined Answer"] for result in results]
        baseline_answers = [result["Baseline Answer"] for result in results]
        baseline_metrics = compute_metrics_for_open_answer(
            baseline_answers, true_answers
        )
        metrics = compute_metrics_for_open_answer(answers, true_answers)
    else:
        answers = get_cleaned_final_answer(results, "Final Refined Answer")
        baseline_answers = get_cleaned_final_answer(results, "Baseline Answer")
        baseline_metrics = compute_metrics_for_list_answer(
            baseline_answers, true_answers
        )
        metrics = compute_metrics_for_list_answer(answers, true_answers)

    print(f"Baseline metrics: {convert_np_floats(baseline_metrics)}")
    print(f"CoVe metrics: {convert_np_floats(metrics)}")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "-r", "--result-path", type=str, help="Path to the result file."
    )
    argParser.add_argument(
        "-d", "--dataset-path", type=str, help="Path to the original dataet."
    )
    argParser.add_argument(
        "-t",
        "--dataset-type",
        type=str,
        help="Type of the dataet.",
        choices=["wikidata", "wikidata_category", "multispan_qa"],
    )

    args = argParser.parse_args()

    evaluate(args.result_path, args.dataset_path, args.dataset_type)
