# VLM-QA 视觉语言模型问答评估框架

## 简介

VLM-QA 是一个基于 GPT-4 评估的视觉语言模型（VLM）问答性能评估框架。该框架通过视觉语言模型生成答案，然后使用 GPT-4 对答案的正确性进行评估，从而全面评估模型在视觉问答任务上的表现和幻觉情况。

## 功能特性

- **多种评估基准**：支持 `hallusionbench`、`vh-test-oeq`、`vh-test-ynq` 三种评估基准
- **GPT-4 自动评估**：使用 GPT-4 对模型答案进行自动评估，判断答案的正确性
- **多维度统计**：提供 LH（语言幻觉）、VI（视觉幻觉）、Mix（混合幻觉）等多维度统计信息
- **多模型支持**：支持 Qwen2-VL、LLaVA、LLaVANeXT 等多种视觉语言模型
- **结果持久化**：自动保存模型响应、评估结果和统计信息

## 安装要求

确保已安装以下依赖：

```bash
pip install torch transformers pillow tqdm numpy prettytable openai
```

对于 Qwen2-VL，还需要：

```bash
pip install qwen-vl-utils
```

## 数据集准备

### 数据集文件位置

位于：https://huggingface.co/datasets/S3IC/hallusion_vlm_qa

各评估基准对应的数据集文件如下：

| 基准测试 | 默认数据路径 | 文件名 |
|---------|------------|--------|
| **HallusionBench** | `data/dataset/hallusion_bench/` | `HallusionBench.json` |
| **VH-Test-OEQ** | `data/dataset/vh_test/` | `OEQ_Benchmark.json` |
| **VH-Test-YNQ** | `data/dataset/vh_test/` | `YNQ_Benchmark.json` |

### 数据集格式

数据集应为 JSON 格式，每个样本包含以下字段：

- `filename`: 图像文件路径（相对于项目根目录或绝对路径）
- `question`: 问题文本
- `gt_answer_details`: 标准答案详情（用于 GPT-4 评估）
- `category`: 类别（HallusionBench 专用）
- `subcategory`: 子类别（HallusionBench 专用）
- `set_id`: 集合 ID（HallusionBench 专用）
- `figure_id`: 图像 ID（HallusionBench 专用）
- `question_id`: 问题 ID
- `visual_input`: 视觉输入标识（HallusionBench 专用）

### 自定义数据路径

可以通过 `--data_path_dir` 参数指定自定义数据路径：

```bash
--data_path_dir /path/to/your/data/directory
```

## 使用方法

### 基本调用

通过 `OmniTrust/examples/hallucination.py` 调用：

```bash
python OmniTrust/examples/hallucination.py \
    --method hallusionbench \
    --model-type vlm \
    --model-name llava-1.5-13b-hf \
    --model_path_dir ~/models \
    --data_path_dir ~/datasets/hallusion_bench
```

### HallusionBench 评估

```bash
python OmniTrust/examples/hallucination.py \
    --method hallusionbench \
    --model-type vlm \
    --model-name Qwen2-VL-2B-Instruct \
    --model_path_dir ~/models
```

### VH-Test-OEQ 评估

```bash
python OmniTrust/examples/hallucination.py \
    --method vh-test-oeq \
    --model-type vlm \
    --model-name llava-1.5-13b-hf \
    --model_path_dir ~/models
```

### VH-Test-YNQ 评估

```bash
python OmniTrust/examples/hallucination.py \
    --method vh-test-ynq \
    --model-type vlm \
    --model-name llava-v1.6-vicuna-13b-hf \
    --model_path_dir ~/models
```

## 参数说明

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-name` | str | `llava-1.5-13b-hf` | 视觉语言模型名称或路径。支持：`Qwen2-VL-72B-Instruct`, `Qwen2-VL-7B-Instruct`, `Qwen2-VL-2B-Instruct`, `llava-v1.6-vicuna-13b-hf`, `llava-v1.6-mistral-7b-hf`, `llava-1.5-13b-hf`, `llava-1.5-7b-hf`，或本地模型路径 |
| `--model_path_dir` | str | `None` | 模型存储目录。如果提供，将从本地路径加载模型；否则从 HuggingFace 加载 |

### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_path_dir` | str | `None` | 数据集存储目录。如果提供，将从指定路径加载数据；否则使用默认路径 |

### 方法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--method` | str | - | 评估方法。必须为以下之一：`hallusionbench`, `vh-test-oeq`, `vh-test-ynq` |
| `--model-type` | str | `llm` | 模型类型。必须为 `vlm` |

## 输出结果

### 结果存储位置

运行完成后，评估结果将自动保存在以下目录：

```
modules/hallucination/vlm/vlm_qa/exp/
```

如果该目录不存在，程序会自动创建。

### 结果文件

评估过程会生成以下文件：

1. **模型响应文件**：`{method}_output.json`
   - 包含所有样本的模型预测答案
   - 字段：`model_prediction`（模型预测答案）

2. **评估结果文件**：`{method}_result.json`
   - 包含 GPT-4 评估结果
   - 字段：`gpt4v_output_gpt_check`（GPT-4 评估结果：0=错误，1=正确，2=不明确）

3. **评估统计文件**：`{method}_eval_result.json`
   - 包含评估统计信息
   - 字段：
     - `domain`: 评估基准名称
     - `model`: 使用的模型名称
     - `overall_accuracy`: 总体准确率（百分比）
     - `total`: 总样本数
     - `correct`: 正确样本数
     - `statistics`: 统计信息
       - `LH`: 语言幻觉数量
       - `VI`: 视觉幻觉数量
       - `Mix`: 混合幻觉数量
     - `timestamp`: 评估时间戳

### 结果文件示例

`hallusionbench_eval_result.json`:

```json
{
  "domain": "hallusionbench",
  "model": "llava-1.5-13b-hf",
  "overall_accuracy": 75.5,
  "total": 1000,
  "correct": 755,
  "statistics": {
    "LH": 120,
    "VI": 80,
    "Mix": 45
  },
  "timestamp": "2025-01-15 14:30:25"
}
```

## 使用示例

### 示例 1：使用 LLaVA 1.5 评估 HallusionBench

```bash
python OmniTrust/examples/hallucination.py \
    --method hallusionbench \
    --model-type vlm \
    --model-name llava-1.5-13b-hf \
    --model_path_dir ~/models
```

### 示例 2：使用 Qwen2-VL 评估 VH-Test-OEQ

```bash
python OmniTrust/examples/hallucination.py \
    --method vh-test-oeq \
    --model-type vlm \
    --model-name Qwen2-VL-7B-Instruct \
    --model_path_dir ~/models \
    --data_path_dir ~/datasets/vh_test
```

### 示例 3：使用 LLaVA-NeXT 评估 VH-Test-YNQ

```bash
python OmniTrust/examples/hallucination.py \
    --method vh-test-ynq \
    --model-type vlm \
    --model-name llava-v1.6-vicuna-13b-hf \
    --model_path_dir ~/models
```

### 示例 4：使用本地模型路径

```bash
python OmniTrust/examples/hallucination.py \
    --method hallusionbench \
    --model-type vlm \
    --model-name /path/to/your/model/llava-1.5-13b-hf \
    --data_path_dir /path/to/your/data
```

## 工作流程

1. **数据加载**：从指定路径加载评估数据集
2. **模型响应生成**：使用视觉语言模型对每个样本生成答案
3. **GPT-4 评估**：使用 GPT-4 对模型答案进行评估
   - 比较模型答案与标准答案
   - 判断答案正确性（correct/incorrect/unclear）
   - 检查答案一致性
4. **结果统计**：计算总体准确率和各类幻觉统计
5. **结果保存**：将评估结果保存到文件

## 注意事项

1. **GPT-4 API 配置**：评估过程需要使用 GPT-4 API。请确保在 `vlm_utils.py` 中正确配置了 API 密钥和端点：
   ```python
   api_key = 'your-api-key'
   api_base = 'your-api-base-url'
   ```

2. **模型路径**：如果使用本地模型，确保 `--model_path_dir` 指向正确的模型目录。模型应按照 `{model_path_dir}/{model_name}` 的结构组织。

3. **数据集路径**：确保数据集文件路径正确。图像路径可以是相对路径（相对于项目根目录）或绝对路径。

4. **GPU 内存**：较大的模型（如 Qwen2-VL-72B）需要足够的 GPU 内存。建议根据硬件配置选择合适的模型。

5. **API 限制**：GPT-4 评估需要调用 API，请注意 API 调用频率限制和成本。

6. **结果文件**：如果结果文件已存在，程序会自动删除旧文件以避免冲突。

## 支持的模型和数据集

### 支持的视觉语言模型

- **Qwen2-VL**: `Qwen2-VL-72B-Instruct`, `Qwen2-VL-7B-Instruct`, `Qwen2-VL-2B-Instruct`
- **LLaVA**: `llava-1.5-13b-hf`, `llava-1.5-7b-hf`
- **LLaVA-NeXT**: `llava-v1.6-vicuna-13b-hf`, `llava-v1.6-mistral-7b-hf`

### 支持的评估基准

- **HallusionBench**: 综合视觉语言模型幻觉评估基准
- **VH-Test-OEQ**: 开放性问题评估基准
- **VH-Test-YNQ**: 是/否问题评估基准

## 文件结构

```
vlm_qa/
├── README.md                    # 本文档
├── pipeline.py                  # 主流程文件
├── vlm_utils.py                 # 工具函数（GPT-4 评估等）
├── __init__.py
└── exp/                         # 结果输出目录（自动创建）
    ├── {method}_output.json     # 模型响应
    ├── {method}_result.json     # 评估结果
    └── {method}_eval_result.json # 评估统计
```

## 评估指标说明

### 总体准确率（Overall Accuracy）

总体准确率 = (正确样本数 / 总样本数) × 100%

### 幻觉类型统计

- **LH (Language Hallucination)**: 语言幻觉，模型在无视觉输入或视觉输入不足时产生的错误
- **VI (Visual Illusion)**: 视觉幻觉，模型对视觉信息的错误理解
- **Mix (Mixed Hallucination)**: 混合幻觉，同时涉及语言和视觉的错误

### GPT-4 评估结果

- `0`: 错误（incorrect）- 模型答案与标准答案冲突
- `1`: 正确（correct）- 模型答案与标准答案一致
- `2`: 不明确（unclear）- 模型答案不明确或无法判断

## 故障排除

### 问题 1：找不到数据集文件

**解决方案**：
- 检查数据集文件路径是否正确
- 使用 `--data_path_dir` 参数指定正确的数据目录
- 确保数据集文件名为 `HallusionBench.json`、`OEQ_Benchmark.json` 或 `YNQ_Benchmark.json`

### 问题 2：GPT-4 API 调用失败

**解决方案**：
- 检查 `vlm_utils.py` 中的 API 配置是否正确
- 确认 API 密钥有效且有足够的配额
- 检查网络连接和 API 端点是否可访问

### 问题 3：模型加载失败

**解决方案**：
- 检查模型路径是否正确
- 确认模型文件完整
- 如果从 HuggingFace 加载，检查网络连接

### 问题 4：GPU 内存不足

**解决方案**：
- 使用较小的模型（如 Qwen2-VL-2B 或 LLaVA-7B）
- 减少批处理大小
- 使用 CPU 模式（不推荐，速度较慢）

