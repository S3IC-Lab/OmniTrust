#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniTrust Fairness Module - Cognitive Bias Evaluator Tests

测试6种认知偏见评估器：
- OrderBiasEvaluator: 顺序偏见
- CompassionBiasEvaluator: 同情心偏见
- BandwagonBiasEvaluator: 从众偏见
- DistractionBiasEvaluator: 注意力分散偏见
- SelectiveBiasEvaluator: 选择性注意偏见
- FrequencyBiasEvaluator: 频率偏见

数据集配置：
- 10个样本, 4个模型 (vicuna, chatgpt, gpt4, alpaca)
- 模型对数: C(4,2) = 6对
- 每个评估器调用次数: 10 × 6 × 2 = 120次

使用方法:
    python test_cognitive_evaluators.py --evaluator order
    python test_cognitive_evaluators.py --evaluator order --model llama3.1:8b
    python test_cognitive_evaluators.py --evaluator all --model qwen2:7b
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


from data import CognitiveBiasDataset
from modules.fairness.evaluators.cognitive import (
    OrderBiasEvaluator,
    CompassionBiasEvaluator,
    BandwagonBiasEvaluator,
    DistractionBiasEvaluator,
    SelectiveBiasEvaluator,
    FrequencyBiasEvaluator,
    get_cognitive_evaluator,
)


class MockModel:
    """模拟模型用于测试（不调用真实LLM）"""

    def __init__(self, name: str = "mock_model", prefer_first: bool = False):
        self.name = name
        self.prefer_first = prefer_first
        self.call_count = 0

    def generate(self, prompts: List[str]) -> List[Dict[str, str]]:
        """模拟生成响应"""
        results = []
        for prompt in prompts:
            self.call_count += 1
            # 模拟响应：随机选择或固定偏好
            if self.prefer_first:
                response = "System Star is better"
            else:
                import random
                response = random.choice([
                    "System Star is better",
                    "System Square is better"
                ])
            results.append({"generation": response})
        return results

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

    def generate(self, prompts: List[str]) -> List[Dict[str, str]]:
        """调用模型生成响应"""
        results = []
        for prompt in prompts:
            self.call_count += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                text = response.choices[0].message.content
                results.append({"generation": text})
            except Exception as e:
                print(f"⚠️ Generation failed: {e}")
                results.append({"generation": ""})
        return results

    def get_model_info(self) -> Dict[str, Any]:
        return {"name": self.model_name}


def estimate_time(n_samples: int, n_models: int, evaluator_name: str) -> Dict[str, Any]:
    """
    估算测试时间

    Args:
        n_samples: 样本数
        n_models: 模型数
        evaluator_name: 评估器名称

    Returns:
        时间估算信息
    """
    from math import comb

    n_pairs = comb(n_models, 2)

    # 不同评估器的调用次数不同
    if evaluator_name in ["order", "compassion", "frequency"]:
        calls_per_pair = 2  # 正向 + 反向验证
    elif evaluator_name in ["bandwagon", "distraction"]:
        calls_per_pair = 2  # 两种条件
    elif evaluator_name == "selective":
        calls_per_pair = 1  # 只需一次调用
    else:
        calls_per_pair = 2

    total_calls = n_samples * n_pairs * calls_per_pair

    # 假设每次调用约2秒（Llama 3.1 8B 本地）
    avg_seconds_per_call = 2.0
    estimated_seconds = total_calls * avg_seconds_per_call

    return {
        "n_samples": n_samples,
        "n_models": n_models,
        "n_pairs": n_pairs,
        "calls_per_pair": calls_per_pair,
        "total_calls": total_calls,
        "estimated_seconds": estimated_seconds,
        "estimated_minutes": estimated_seconds / 60,
        "estimated_hours": estimated_seconds / 3600,
    }


def run_test(evaluator_name: str, dataset: CognitiveBiasDataset,
             model: Any, limit: int = None) -> Dict[str, Any]:
    """
    运行单个评估器测试

    Args:
        evaluator_name: 评估器名称
        dataset: 数据集
        model: 模型
        limit: 样本限制

    Returns:
        测试结果
    """
    print(f"\n{'='*60}")
    print(f"Testing: {evaluator_name.upper()} Bias Evaluator")
    print(f"{'='*60}")

    # 获取评估器
    evaluator = get_cognitive_evaluator(evaluator_name)

    # 准备数据
    if limit:
        data = dataset.sample(limit)
    else:
        data = dataset.get_data()

    n_samples = data["N"]
    n_models = len(data["responses"])

    # 时间估算
    estimate = estimate_time(n_samples, n_models, evaluator_name)
    print(f"\n📊 Test Configuration:")
    print(f"   - Samples: {n_samples}")
    print(f"   - Models: {n_models}")
    print(f"   - Model pairs: {estimate['n_pairs']}")
    print(f"   - Total LLM calls: {estimate['total_calls']}")
    print(f"   - Estimated time: {estimate['estimated_minutes']:.1f} minutes")

    # 运行评估
    print(f"\n🚀 Running evaluation...")
    start_time = time.time()

    # 创建临时数据包装器
    class DataWrapper:
        def __init__(self, d):
            self.data = d

    result = evaluator.evaluate(model, DataWrapper(data))
    elapsed = time.time() - start_time

    # 输出结果
    print(f"\n📈 Results:")
    print(f"   - Bias count: {result.get('bias_count', 'N/A')}")
    print(f"   - Bias rate: {result.get('bias_rate', 0):.2%}")
    print(f"   - Valid responses: {result.get('valid_responses', 'N/A')}")
    print(f"   - Valid rate: {result.get('valid_response_rate', 0):.2%}")
    print(f"   - Consistency: {result.get('consistency', 'N/A')}")
    print(f"   - Consistency rate: {result.get('consistency_rate', 0):.2%}")

    if result.get('first_order_bias'):
        print(f"   - First order bias: {result['first_order_bias']}")
    if result.get('last_order_bias'):
        print(f"   - Last order bias: {result['last_order_bias']}")
    if result.get('me_bias'):
        print(f"   - Me bias: {result['me_bias']}")

    print(f"\n⏱️ Elapsed time: {elapsed:.2f} seconds")
    print(f"   - Actual calls: {getattr(model, 'call_count', 'N/A')}")

    result['elapsed_seconds'] = elapsed
    result['estimate'] = estimate

    return result


def main():
    parser = argparse.ArgumentParser(description="Test cognitive bias evaluators")
    parser.add_argument(
        "--evaluator", "-e",
        choices=["order", "compassion", "bandwagon", "distraction", "selective", "frequency", "all"],
        default="order",
        help="Evaluator to test (default: order)"
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
        "--output", "-o",
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only show time estimates, don't run tests"
    )

    args = parser.parse_args()

    # 加载数据集 (10 samples, 4 models = 6 pairs)
    data_path = project_root / "data" / "dataset" / "cognitive_bias.json"
    dataset = CognitiveBiasDataset(
        data_dir=str(data_path),
        evaluator_name="test_evaluator"
    )

    data = dataset.get_data()
    n_samples = args.limit if args.limit else data["N"]
    n_models = len(data["responses"])

    print(f"\n🔧 OmniTrust Cognitive Bias Evaluator Test")
    print(f"   Dataset: {data['N']} samples, {n_models} models")
    print(f"   Using: {n_samples} samples")

    # 时间估算
    evaluators = ["order", "compassion", "bandwagon", "distraction", "selective", "frequency"]
    if args.evaluator != "all":
        evaluators = [args.evaluator]

    print(f"\n📊 Time Estimates (with {n_samples} samples, {n_models} models):")
    print(f"{'Evaluator':<15} {'Pairs':<8} {'Calls':<10} {'Est. Time':<15}")
    print("-" * 50)

    total_calls = 0
    for name in evaluators:
        est = estimate_time(n_samples, n_models, name)
        total_calls += est['total_calls']
        print(f"{name:<15} {est['n_pairs']:<8} {est['total_calls']:<10} {est['estimated_minutes']:.1f} min")

    print("-" * 50)
    print(f"{'TOTAL':<15} {'':<8} {total_calls:<10} {total_calls * 2 / 60:.1f} min")

    if args.estimate_only:
        return

    # 初始化模型
    print(f"\n🚀 Loading model: {args.model}")
    if args.model == "mock":
        model = MockModel(name="mock_model")
        print(f"⚠️ Using MOCK model (no real LLM calls)")
    else:
        model = OllamaModel(model_name=args.model)

    # 运行测试
    results = {}
    for name in evaluators:
        model.call_count = 0  # 重置计数
        result = run_test(name, dataset, model, args.limit)
        results[name] = result

    # 保存结果
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "evaluators": evaluators,
                "limit": args.limit,
                "model": args.model,
            },
            "results": results
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Results saved to: {args.output}")

    print(f"\n✅ All tests completed!")


if __name__ == "__main__":
    main()
