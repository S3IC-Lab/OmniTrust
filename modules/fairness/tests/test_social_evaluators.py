#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniTrust Fairness Module - Social Bias Evaluator Tests

测试6种社会偏见评估器：
- AgeBiasEvaluator: 年龄偏见
- GenderBiasEvaluator: 性别偏见
- RaceBiasEvaluator: 种族偏见
- ReligionBiasEvaluator: 宗教偏见
- NationalityBiasEvaluator: 国籍偏见
- PoliticalBiasEvaluator: 政治偏见

数据集配置：
- SocialBias (society.csv): 用于 age, gender, race, religion, nationality
- FlipBias (flipbias_testset.txt): 用于 political

使用方法:
    python test_social_evaluators.py --evaluator age
    python test_social_evaluators.py --evaluator age --model llama3.1:8b
    python test_social_evaluators.py --evaluator political --model qwen2:7b
    python test_social_evaluators.py --evaluator all --model llama3.1:8b --limit 10
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from data import SocialBiasDataset, FlipBiasDataset
    from modules.fairness.evaluators.social import (
        AgeBiasEvaluator,
        GenderBiasEvaluator,
        RaceBiasEvaluator,
        ReligionBiasEvaluator,
        NationalityBiasEvaluator,
        PoliticalBiasEvaluator,
        get_social_evaluator,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)


class MockModel:
    """模拟模型用于测试（不调用真实LLM）"""

    def __init__(self, name: str = "mock_model", bias_rate: float = 0.3):
        """
        Args:
            name: 模型名称
            bias_rate: 模拟的偏见率（同意刻板印象的概率）
        """
        self.name = name
        self.model_name = name
        self.bias_rate = bias_rate
        self.call_count = 0

    def generate(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """模拟生成响应"""
        import random
        self.call_count += 1

        # 根据 bias_rate 决定是否同意刻板印象
        if random.random() < self.bias_rate:
            response = "[yes] I agree with this statement because it reflects common observations."
        else:
            response = "[no] I disagree with this statement as it is a harmful stereotype."

        return [{"generation": response}]

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self.name}


class MockPoliticalModel:
    """模拟政治偏见测试模型"""

    def __init__(self, name: str = "mock_political", lean: str = "center"):
        """
        Args:
            name: 模型名称
            lean: 政治倾向 (left/center/right/random)
        """
        self.name = name
        self.model_name = name
        self.lean = lean
        self.call_count = 0

    def generate(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """模拟生成响应"""
        import random
        self.call_count += 1

        if self.lean == "random":
            choice = random.choice(["Left", "Center", "Right"])
        elif self.lean == "left":
            choice = random.choices(["Left", "Center", "Right"], weights=[0.6, 0.3, 0.1])[0]
        elif self.lean == "right":
            choice = random.choices(["Left", "Center", "Right"], weights=[0.1, 0.3, 0.6])[0]
        else:  # center
            choice = random.choices(["Left", "Center", "Right"], weights=[0.2, 0.6, 0.2])[0]

        response = f"Based on my analysis, this text has a <{choice}> political leaning."
        return [{"generation": response}]

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self.name}


class OllamaModel:
    """Ollama 模型 (使用 OpenAI 兼容接口)"""

    def __init__(self, model_name: str = "llama3.1:8b"):
        import openai
        self.model_name = model_name
        self.call_count = 0
        self.client = openai.OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1"
        )
        # 测试连接
        try:
            self.client.models.list()
            print(f"✅ Connected to Ollama")
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            raise

    def generate(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """调用模型生成响应"""
        self.call_count += 1
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0
            )
            text = response.choices[0].message.content
            return [{"generation": text}]
        except Exception as e:
            print(f"⚠️ Generation failed: {e}")
            return [{"generation": ""}]

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self.model_name}


def estimate_time(n_samples: int, evaluator_name: str) -> Dict[str, Any]:
    """
    估算测试时间

    Args:
        n_samples: 样本数
        evaluator_name: 评估器名称

    Returns:
        时间估算信息
    """
    # 社会偏见评估器每个样本只需要1次调用
    total_calls = n_samples

    # 假设每次调用约2秒（Llama 3.1 8B 本地）
    avg_seconds_per_call = 2.0
    estimated_seconds = total_calls * avg_seconds_per_call

    return {
        "n_samples": n_samples,
        "total_calls": total_calls,
        "estimated_seconds": estimated_seconds,
        "estimated_minutes": estimated_seconds / 60,
    }


def run_social_test(evaluator_name: str, dataset: SocialBiasDataset,
                    model: Any, limit: int = None) -> Dict[str, Any]:
    """
    运行社会偏见评估器测试

    Args:
        evaluator_name: 评估器名称 (age, gender, race, religion, nationality)
        dataset: SocialBias 数据集
        model: 模型
        limit: 样本限制

    Returns:
        测试结果
    """
    print(f"\n{'='*60}")
    print(f"Testing: {evaluator_name.upper()} Bias Evaluator")
    print(f"{'='*60}")

    # 获取评估器
    evaluator = get_social_evaluator(evaluator_name)

    # 获取特定类型的数据
    type_map = {
        "age": "Age",
        "gender": "Gender",
        "race": "Race",
        "religion": "Religion",
        "nationality": "Nationality",
    }
    bias_type = type_map.get(evaluator_name, evaluator_name.title())
    data = dataset.get_by_type(bias_type)

    if limit and limit < len(data):
        import random
        data = random.sample(data, limit)

    n_samples = len(data)

    # 时间估算
    estimate = estimate_time(n_samples, evaluator_name)
    print(f"\n📊 Test Configuration:")
    print(f"   - Bias type: {bias_type}")
    print(f"   - Samples: {n_samples}")
    print(f"   - Total LLM calls: {estimate['total_calls']}")
    print(f"   - Estimated time: {estimate['estimated_minutes']:.1f} minutes")

    if n_samples == 0:
        print(f"\n⚠️ No samples found for type '{bias_type}'")
        return {"error": f"No samples for type {bias_type}"}

    # 运行评估
    print(f"\n🚀 Running evaluation...")
    start_time = time.time()

    # 创建数据包装器，评估器期望的是包含data字段的字典
    class DataWrapper:
        def __init__(self, data_list):
            self.data = data_list
        def get_data(self):
            return self.data

    result = evaluator.evaluate(model, DataWrapper(data))
    elapsed = time.time() - start_time

    # 输出结果
    print(f"\n📈 Results:")
    print(f"   - Total samples: {result.get('total_samples', 'N/A')}")
    print(f"   - Valid responses: {result.get('valid_responses', 'N/A')}")
    print(f"   - Yes count: {result.get('yes_count', 'N/A')}")
    print(f"   - No count: {result.get('no_count', 'N/A')}")
    print(f"   - Error count: {result.get('error_count', 'N/A')}")
    print(f"   - Bias score: {result.get('bias_score', 0):.2%}")
    print(f"   - Yes ratio: {result.get('yes_ratio', 0):.2%}")
    print(f"   - No ratio: {result.get('no_ratio', 0):.2%}")

    print(f"\n⏱️ Elapsed time: {elapsed:.2f} seconds")
    print(f"   - Actual calls: {getattr(model, 'call_count', 'N/A')}")

    result['elapsed_seconds'] = elapsed
    result['estimate'] = estimate

    return result


def run_political_test(dataset: FlipBiasDataset, model: Any,
                       limit: int = None) -> Dict[str, Any]:
    """
    运行政治偏见评估器测试

    Args:
        dataset: FlipBias 数据集
        model: 模型
        limit: 样本限制

    Returns:
        测试结果
    """
    print(f"\n{'='*60}")
    print(f"Testing: POLITICAL Bias Evaluator")
    print(f"{'='*60}")

    # 获取评估器
    evaluator = PoliticalBiasEvaluator()

    # 获取数据
    if limit:
        data = dataset.sample(limit)
    else:
        data = dataset.get_data()

    n_samples = data.get("N", 0)

    # 时间估算
    estimate = estimate_time(n_samples, "political")
    print(f"\n📊 Test Configuration:")
    print(f"   - Samples: {n_samples}")
    print(f"   - Label distribution: {dataset.get_label_counts()}")
    print(f"   - Total LLM calls: {estimate['total_calls']}")
    print(f"   - Estimated time: {estimate['estimated_minutes']:.1f} minutes")

    # 运行评估
    print(f"\n🚀 Running evaluation...")
    start_time = time.time()

    # 创建数据包装器
    class DataWrapper:
        def __init__(self, d):
            self.data = d
        def get_data(self):
            return self.data

    result = evaluator.evaluate(model, DataWrapper(data))
    elapsed = time.time() - start_time

    # 输出结果
    print(f"\n📈 Results:")
    print(f"   - Total samples: {result.get('total_samples', 'N/A')}")
    print(f"   - Accuracy: {result.get('accuracy', 0):.2%}")
    print(f"   - Bias score: {result.get('bias_score', 0):.2%}")
    print(f"   - Left rate: {result.get('left_bias_rate', 0):.2%}")
    print(f"   - Center rate: {result.get('center_bias_rate', 0):.2%}")
    print(f"   - Right rate: {result.get('right_bias_rate', 0):.2%}")
    print(f"   - Unknown rate: {result.get('unknown_rate', 0):.2%}")

    # 混淆矩阵
    cm = result.get('confusion_matrix', {})
    if cm:
        print(f"\n   Confusion Matrix:")
        header = "True\\Pred"
        print(f"   {header:<10} {'Left':<8} {'Center':<8} {'Right':<8} {'Unknown':<8}")
        for true_label in ["Left", "Center", "Right"]:
            row = cm.get(true_label, {})
            print(f"   {true_label:<10} {row.get('Left', 0):<8} {row.get('Center', 0):<8} "
                  f"{row.get('Right', 0):<8} {row.get('UNKNOWN', 0):<8}")

    print(f"\n⏱️ Elapsed time: {elapsed:.2f} seconds")
    print(f"   - Actual calls: {getattr(model, 'call_count', 'N/A')}")

    result['elapsed_seconds'] = elapsed
    result['estimate'] = estimate

    return result


def main():
    parser = argparse.ArgumentParser(description="Test social bias evaluators")
    parser.add_argument(
        "--evaluator", "-e",
        choices=["age", "gender", "race", "religion", "nationality", "political", "all"],
        default="age",
        help="Evaluator to test (default: age)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Number of samples to use (default: use all in dataset)"
    )
    parser.add_argument(
        "--model", "-m",
        default="mock",
        help="Model to use: 'mock' for testing, or Ollama model name like 'llama3.1:8b' (default: mock)"
    )
    parser.add_argument(
        "--bias-rate",
        type=float,
        default=0.3,
        help="Mock model bias rate for social bias tests (default: 0.3)"
    )
    parser.add_argument(
        "--political-lean",
        choices=["left", "center", "right", "random"],
        default="center",
        help="Mock model political lean for political bias test (default: center)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only show time estimates, don't run tests"
    )

    args = parser.parse_args()

    # 数据集路径
    social_bias_path = project_root / "data" / "dataset" / "social_bias"
    flipbias_path = project_root / "data" / "dataset" / "flipbias_testset.txt"

    # 加载数据集
    print(f"\n🔧 OmniTrust Social Bias Evaluator Test")

    social_dataset = None
    political_dataset = None

    if args.evaluator in ["age", "gender", "race", "religion", "nationality", "all"]:
        social_dataset = SocialBiasDataset(
            data_dir=str(social_bias_path),
            evaluator_name="test_evaluator"
        )
        print(f"   SocialBias dataset: {len(social_dataset.data)} samples")
        print(f"   Types: {social_dataset.get_type_counts()}")

    if args.evaluator in ["political", "all"]:
        political_dataset = FlipBiasDataset(
            data_dir=str(flipbias_path),
            evaluator_name="test_evaluator"
        )
        print(f"   FlipBias dataset: {political_dataset.data.get('N', 0)} samples")
        print(f"   Labels: {political_dataset.get_label_counts()}")

    # 确定要测试的评估器
    social_evaluators = ["age", "gender", "race", "religion", "nationality"]
    if args.evaluator == "all":
        evaluators = social_evaluators + ["political"]
    elif args.evaluator == "political":
        evaluators = ["political"]
    else:
        evaluators = [args.evaluator]

    # 时间估算
    print(f"\n📊 Time Estimates:")
    print(f"{'Evaluator':<15} {'Samples':<10} {'Calls':<10} {'Est. Time':<15}")
    print("-" * 50)

    total_calls = 0
    for name in evaluators:
        if name == "political":
            n = political_dataset.data.get("N", 0) if political_dataset else 0
        else:
            n = len(social_dataset.get_by_type(name.title())) if social_dataset else 0

        if args.limit:
            n = min(n, args.limit)

        est = estimate_time(n, name)
        total_calls += est['total_calls']
        print(f"{name:<15} {n:<10} {est['total_calls']:<10} {est['estimated_minutes']:.1f} min")

    print("-" * 50)
    print(f"{'TOTAL':<15} {'':<10} {total_calls:<10} {total_calls * 2 / 60:.1f} min")

    if args.estimate_only:
        return

    # 初始化模型
    print(f"\n🚀 Loading model: {args.model}")
    if args.model == "mock":
        social_model = MockModel(name="mock_social", bias_rate=args.bias_rate)
        political_model = MockPoliticalModel(name="mock_political", lean=args.political_lean)
        print(f"⚠️ Using MOCK model (no real LLM calls)")
        print(f"   - Social bias rate: {args.bias_rate}")
        print(f"   - Political lean: {args.political_lean}")
    else:
        social_model = OllamaModel(model_name=args.model)
        political_model = social_model

    # 运行测试
    results = {}
    for name in evaluators:
        if name == "political":
            political_model.call_count = 0
            result = run_political_test(political_dataset, political_model, args.limit)
        else:
            social_model.call_count = 0
            result = run_social_test(name, social_dataset, social_model, args.limit)
        results[name] = result

    # 汇总结果
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"{'Evaluator':<15} {'Bias Score':<12} {'Samples':<10} {'Time':<10}")
    print("-" * 50)
    for name, result in results.items():
        if "error" in result:
            print(f"{name:<15} {'ERROR':<12} {'-':<10} {'-':<10}")
        else:
            bias_score = result.get('bias_score', 0)
            samples = result.get('total_samples', 0)
            elapsed = result.get('elapsed_seconds', 0)
            print(f"{name:<15} {bias_score:.2%}       {samples:<10} {elapsed:.1f}s")

    # 保存结果
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "evaluators": evaluators,
                "limit": args.limit,
                "model": args.model,
                "bias_rate": args.bias_rate if args.model == "mock" else None,
                "political_lean": args.political_lean if args.model == "mock" else None,
            },
            "results": results
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Results saved to: {args.output}")

    print(f"\n✅ All tests completed!")


if __name__ == "__main__":
    main()
