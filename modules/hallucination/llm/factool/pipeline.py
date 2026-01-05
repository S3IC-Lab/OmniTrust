import sys
import yaml
from .kb_qa.pipeline import knowledge_qa_pipeline
from .code.pipeline import code_pipeline
from .math_problem.pipeline import math_pipeline
from .scientific.pipeline import scientific_pipeline
from .med_doc_qa.pipeline import med_doc_qa_pipeline
from .utils import *
import os
import pathlib
import copy
import random
import asyncio


class factool_pipeline():
    def __init__(self, args):
        self.pipelines = {
                            "kbqa_online": knowledge_qa_pipeline(
                                args.wrapper, 10, "online"
                            ),
                            "code": code_pipeline(
                                args.wrapper, 3, 3
                            ),
                            "math": math_pipeline(
                                args.wrapper
                            ),
                            "scientific": scientific_pipeline(
                                args.wrapper
                            ),
                            "med_doc_qa": med_doc_qa_pipeline(
                                args.wrapper
                            )
                        }

    def run(self, args):
        inputs = load_data(args)
        # inputs = inputs[:args.n_samples]
        inputs = random.sample(inputs, args.n_samples)

        outputs = copy.deepcopy(inputs)
        

        if args.task == 'kbqa':
            if args.search_type is None or args.search_type == "online":
                # online
                results = asyncio.run(
                    self.pipelines[args.task + "_online"].run_with_tool_api_call(
                        [sample['prompt'] for sample in inputs],
                        [sample['response'] for sample in inputs],
                    )
                )
            else:
                # local
                results = asyncio.run(
                    knowledge_qa_pipeline(
                        self.foundation_model, 2, "local", args.data_link, args.embedding_link
                    ).run_with_tool_api_call(
                        [sample['prompt'] for sample in inputs],
                        [sample['response'] for sample in inputs],
                    )
                )
        elif args.task == 'code':
                results = asyncio.run(
                    self.pipelines[args.task].run_with_tool_api_call(
                        [sample['prompt'] for sample in inputs],
                        [sample['response'] for sample in inputs],
                        [sample['entry_point'] for sample in inputs]
                    )
                )
        elif args.task == 'math':
                results = asyncio.run(
                    self.pipelines[args.task].run_with_tool_api_call(
                    [sample['prompt'] for sample in inputs],
                    [sample['response'] for sample in inputs]
                    )
                )
        else:
            results = asyncio.run(
                self.pipelines["scientific"].run_with_tool_api_call(
                    [sample['prompt'] for sample in inputs],
                    [sample['response'] for sample in inputs]
                )
            )
        for i, result in enumerate(results):
            outputs[i].update(result)
            
        results = calculate(outputs)
        # print(results)
        print("average_claim_level_factuality:", results["average_claim_level_factuality"], "average_response_level_factuality", results["average_response_level_factuality"])
        save_result(args, outputs)
