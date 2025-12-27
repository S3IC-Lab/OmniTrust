# VL-Uncertainty 视觉语言模型幻觉检测框架

## 简介

VL-Uncertainty 是一个基于不确定性估计的视觉语言模型（VLM）幻觉检测框架。该框架通过视觉和文本扰动来评估模型输出的不确定性，从而检测潜在的幻觉。

## 功能特性

- **多种不确定性估计方法**：支持 `vl_uncertainty` 和 `semantic_entropy` 两种方法
- **丰富的视觉扰动**：支持模糊、旋转、翻转、裁剪等多种视觉扰动方式
- **多样的文本扰动**：支持 LLM 重述、词交换、删除、插入等多种文本扰动方式
- **多模型支持**：支持 Qwen2-VL、InternVL、LLaVA 等多种视觉语言模型
- **多基准测试**：支持 MMVet、LLaVABench、MMMU、ScienceQA 等基准数据集

## 安装要求

确保已安装以下依赖：

```bash
pip install torch transformers pillow tqdm datasets numpy
```

对于 Qwen2-VL，还需要：

```bash
pip install qwen-vl-utils
```

还需要安装 HuggingFace CLI 工具：

```bash
pip install huggingface_hub[cli]
```

## 数据集准备

在使用框架之前，需要先从 HuggingFace 下载数据集到本地，并转换为框架所需的格式（本框架支持load_from_disk模式加载数据集）。

### 数据集 HuggingFace 路径

各基准测试数据集对应的 HuggingFace 路径如下：

| 数据集 | HuggingFace 路径 |
|--------|-----------------|
| **MMVet** | `whyu/mm-vet` |
| **LLaVABench** | `lmms-lab/llava-bench-in-the-wild` |
| **MMMU** | `MMMU/MMMU` |
| **ScienceQA** | `derek-thomas/ScienceQA` |

### 下载数据集

#### 1. MMVet

```bash
# 下载数据集
huggingface-cli download --repo-type dataset --resume-download whyu/mm-vet --local-dir ~/datasets/mm-vet
```

#### 2. LLaVABench

```bash
# 下载数据集
huggingface-cli download --repo-type dataset --resume-download lmms-lab/llava-bench-in-the-wild --local-dir ~/datasets/llava-bench-in-the-wild

# 如果下载的是 parquet 格式，需要转换为 load_from_disk 格式
# 创建转换脚本或使用以下 Python 代码：
python -c "
from datasets import load_dataset
dataset = load_dataset('parquet', data_files='~/datasets/llava-bench-in-the-wild/data/train-00000-of-00001.parquet')
dataset.save_to_disk('modules/hallucination/vlm/vl_uncertainty/datasets/llava-bench-in-the-wild')
"
```

#### 3. MMMU

MMMU 数据集需要为每个类别单独下载：

```bash
# 下载完整数据集
huggingface-cli download --repo-type dataset --resume-download MMMU/MMMU --local-dir ~/datasets/MMMU/MMMU

# MMMU 包含 30 个类别，每个类别需要单独处理
# 类别列表：Accounting, Agriculture, Architecture_and_Engineering, Art, Art_Theory, 
# Basic_Medical_Science, Biology, Chemistry, Clinical_Medicine, Computer_Science, 
# Design, Diagnostics_and_Laboratory_Medicine, Economics, Electronics, Energy_and_Power, 
# Finance, Geography, History, Literature, Manage, Marketing, Materials, Math, 
# Mechanical_Engineering, Music, Pharmacy, Physics, Psychology, Public_Health, Sociology

# 为每个类别转换数据格式（示例：Accounting）
python -c "
from datasets import load_dataset
dataset = load_dataset('parquet', data_files='~/datasets/MMMU/MMMU/Accounting/train-*.parquet')
dataset.save_to_disk('modules/hallucination/vlm/vl_uncertainty/datasets/MMMU/Accounting')
"
# 对其他 29 个类别重复上述操作
```

#### 4. ScienceQA

```bash
# 下载数据集
huggingface-cli download --repo-type dataset --resume-download derek-thomas/ScienceQA --local-dir modules/hallucination/vlm/vl_uncertainty/datasets/ScienceQA

# 如果下载的是原始格式，需要转换为 load_from_disk 格式
python -c "
from datasets import load_dataset
dataset = load_dataset('derek-thomas/ScienceQA')
dataset.save_to_disk('modules/hallucination/vlm/vl_uncertainty/datasets/ScienceQA')
"
```

### 数据集目录结构

下载并转换后，数据集应按照以下结构组织：

```
~/datasets/
├── mm-vet/                      # MMVet 数据集
│   └── (load_from_disk 格式)
├── llava-bench-in-the-wild/     # LLaVABench 数据集
│   └── (load_from_disk 格式)
├── MMMU/                        # MMMU 数据集
│   ├── Accounting/
│   ├── Agriculture/
│   ├── Architecture_and_Engineering/
│   ├── ... (其他 27 个类别)
│   └── Sociology/
└── ScienceQA/                   # ScienceQA 数据集
    └── (load_from_disk 格式)
```

### 数据格式转换脚本示例

如果需要批量转换数据格式，可以使用以下脚本：

```python
# convert_dataset.py
from datasets import load_dataset
import os

# 转换 LLaVABench
def convert_llava_bench():
    dataset = load_dataset('parquet', data_files='~/datasets/llava-bench-in-the-wild/data/train-00000-of-00001.parquet')
    dataset.save_to_disk('~/datasets/llava-bench-in-the-wild')

# 转换 MMMU（需要为每个类别单独转换）
def convert_mmmu_category(category_name):
    dataset = load_dataset('parquet', data_files=f'~/datasets/MMMU/MMMU/{category_name}/train-*.parquet')
    dataset.save_to_disk(f'~/datasets/MMMU/{category_name}')

# MMMU 所有类别
mmmu_categories = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
    'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power',
    'Finance', 'Geography', 'History', 'Literature', 'Manage',
    'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music',
    'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'
]

for category in mmmu_categories:
    convert_mmmu_category(category)
    print(f"Converted {category}")
```

### 验证数据集

下载并转换完成后，可以使用以下代码验证数据集是否正确：

```python
from datasets import load_from_disk

# 验证 MMVet
ds = load_from_disk('~/datasets/mm-vet')
print(f"MMVet: {len(ds['train'])} samples")

# 验证 LLaVABench
ds = load_from_disk('~/datasets/llava-bench-in-the-wild')
print(f"LLaVABench: {len(ds['train'])} samples")

# 验证 ScienceQA
ds = load_from_disk('~/datasets/ScienceQA')
print(f"ScienceQA: {len(ds['test'])} samples")

# 验证 MMMU（检查一个类别）
ds = load_from_disk('~/datasets/MMMU/Accounting')
print(f"MMMU Accounting: {len(ds['train'])} samples")
```

## 使用方法

### 基本调用

通过 `OmniTrust/examples/hallucination.py` 调用：

```bash
python OmniTrust/examples/hallucination.py \
    --method vl-uctt \
    --model-type vlm \
    --lvlm Qwen2-VL-2B-Instruct \
    --benchmark LLaVABench \
    --llm Qwen2.5-0.5B-Instruct \
    --uncertainty vl_uncertainty \
    --uncertainty_thres 1.0 \
    --visual_perturbation blurring \
    --blur_radius_list 0.1 0.3 0.5 0.7 1.0 \
    --textual_perturbation llm_rephrasing \
    --textual_perturbation_temp_list 0.1 0.2 0.3 0.4 0.5 \
    --textual_perturbation_instruction_template "Given the input question: '{question}', generate a semantically equivalent variation by changing the wording, structure, grammar, or narrative. Ensure the perturbed question maintains the same meaning as the original. Provide only the rephrased question as the output." \
    --pair_order progressively \
    --inference_temp 0.1 \
    --sampling_temp 0.5 \
    --sampling_time 5 \
    --model_path_dir ~/models \
    --data_path_dir ~/datasets/ScienceQA
```


## 参数说明

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lvlm` | str | `Qwen2-VL-2B-Instruct` | 视觉语言模型名称。支持：`Qwen2-VL-72B-Instruct`, `Qwen2-VL-7B-Instruct`, `Qwen2-VL-2B-Instruct`, `InternVL2-26B`, `InternVL2-8B`, `InternVL2-1B`, `llava-v1.6-vicuna-13b-hf`, `llava-v1.6-mistral-7b-hf`, `llava-1.5-13b-hf`, `llava-1.5-7b-hf` |
| `--llm` | str | `Qwen2.5-3B-Instruct` | 用于文本重述和答案验证的 LLM。支持：`Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B-Instruct`, `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct` |
| `--model_path_dir` | str | `~/models` | 模型存储目录。如果提供，将从本地路径加载模型；否则从 HuggingFace 加载 |

### 基准测试参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--benchmark` | str | `MMVet` | 基准测试数据集。支持：`MMVet`, `LLaVABench`, `MMMU`, `ScienceQA` |
| `--data_path_dir` | str | `~/dataset/ScienceQA` | 数据集存储目录 |

### 不确定性估计参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--uncertainty` | str | `vl_uncertainty` | 不确定性估计方法。可选：`vl_uncertainty`（视觉-语言不确定性）或 `semantic_entropy`（语义熵） |
| `--uncertainty_thres` | float | `1.0` | 不确定性阈值，超过此值将被判定为幻觉 |
| `--sampling_time` | int | `5` | 采样次数，用于不确定性估计 |
| `--inference_temp` | float | `0.1` | 推理时的温度参数（用于确定性推理） |
| `--sampling_temp` | float | `1.0` | 采样时的温度参数（用于不确定性估计） |

### 视觉扰动参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--visual_perturbation` | str | `blurring` | 视觉扰动类型。支持：`blurring`（模糊）, `rotation`（旋转）, `flipping`（翻转）, `shifting`（平移）, `cropping`（裁剪）, `erasing`（擦除）, `gaussian_noise`（高斯噪声）, `dropout`（随机丢弃）, `salt_and_pepper`（椒盐噪声）, `sharpen`（锐化）, `adjust_brightness`（亮度调整）, `adjust_contrast`（对比度调整）, `rotate_shift`（旋转+平移）, `crop_flip`（裁剪+翻转）, `rotate_blur`（旋转+模糊）, `crop_blur`（裁剪+模糊） |
| `--blur_radius_list` | float list | `[0.6, 0.8, 1.0, 1.2, 1.4]` | 模糊半径列表（仅当 `visual_perturbation=blurring` 时使用） |

### 文本扰动参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--textual_perturbation` | str | `llm_rephrasing` | 文本扰动类型。支持：`llm_rephrasing`（LLM 重述）, `swapping`（词交换）, `deleting`（词删除）, `inserting`（词插入）, `replacing`（词替换）, `text_shuffle`（文本打乱）, `noise_injection`（噪声注入）, `word_dropout`（词丢弃）, `character_dropout`（字符丢弃） |
| `--textual_perturbation_temp_list` | float list | `[0.1, 0.2, 0.3, 0.4, 0.5]` | 文本扰动的温度参数列表（仅当 `textual_perturbation=llm_rephrasing` 时使用） |
| `--textual_perturbation_instruction_template` | str | `"Given the input question: '{question}', generate a semantically equivalent variation..."` | LLM 重述的指令模板（仅当 `textual_perturbation=llm_rephrasing` 时使用） |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--pair_order` | str | `progressively` | 扰动对组合顺序。可选：`progressively`（渐进式）, `shift_N`（偏移 N 位）, `random_pair`（随机配对） |

## 输出结果

### 结果存储位置

运行完成后，幻觉检测结果将自动保存在以下目录：

```
modules/hallucination/vlm/vl_uncertainty/exp/
```

如果该目录不存在，程序会自动创建。结果文件命名格式为：`log_YYYY_MM_DD_HH_MM_SS.json`（例如：`log_2025_01_15_14_30_25.json`）

### 结果文件内容

结果文件包含：

- `args`: 运行参数
- `begin_time_str` / `end_time_str`: 开始和结束时间
- 每个样本的详细信息：
  - `question`: 问题
  - `gt_ans`: 正确答案
  - `ans`: 模型答案
  - `flag_ans_correct`: 答案是否正确
  - `ans_sampling_list`: 采样答案列表
  - `uncertainty`: 不确定性值
  - `flag_predict_hallucination`: 是否预测为幻觉
  - `flag_detection_correct`: 检测是否正确
- `Hallucination detection accuracy`: 幻觉检测准确率
- `Total samples`: 总样本数

## 使用示例

### 示例 1：使用模糊扰动和 LLM 重述

```bash
python OmniTrust/examples/hallucination.py \
    --method vl-uctt \
    --model-type vlm \
    --lvlm Qwen2-VL-2B-Instruct \
    --benchmark LLaVABench \
    --llm Qwen2-0.5B-Instruct \
    --uncertainty vl_uncertainty \
    --visual_perturbation blurring \
    --blur_radius_list 0.1 0.3 0.5 0.7 1.0 \
    --textual_perturbation llm_rephrasing \
    --sampling_time 5 \
    --data_path_dir ~/datasets/ScienceQA \
    --model_path_dir ~/model
```

### 示例 2：使用语义熵方法

```bash
python OmniTrust/examples/hallucination.py \
    --method vl-uctt \
    --model-type vlm \
    --lvlm Qwen2-VL-7B-Instruct \
    --benchmark ScienceQA \
    --llm Qwen2.5-3B-Instruct \
    --uncertainty semantic_entropy \
    --sampling_temp 0.8 \
    --sampling_time 10 \
    --data_path_dir ~/datasets/ScienceQA \
    --model_path_dir ~/model
```

### 示例 3：使用旋转和词交换扰动

```bash
python OmniTrust/examples/hallucination.py \
    --method vl-uctt \
    --model-type vlm \
    --lvlm llava-1.5-13b-hf \
    --benchmark LLaVABench \
    --uncertainty vl_uncertainty \
    --visual_perturbation rotation \
    --textual_perturbation swapping \
    --sampling_time 5 \
    --data_path_dir ~/datasets/ScienceQA \
    --model_path_dir ~/model
```

## 注意事项

1. **模型路径**：如果使用本地模型，确保 `--model_path_dir` 指向正确的模型目录。模型应按照 `{model_path_dir}/{model_name}` 的结构组织。

2. **数据集路径**：确保 `--data_path_dir` 指向正确的数据集目录。数据集应使用 HuggingFace `datasets` 库的 `load_from_disk` 格式。

3. **GPU 内存**：较大的模型（如 Qwen2-VL-72B）需要足够的 GPU 内存。建议根据硬件配置选择合适的模型。

4. **LLM 依赖**：`llm_rephrasing` 文本扰动方法需要 LLM 支持。如果不想使用 LLM，可以选择其他文本扰动方法。

5. **采样次数**：`--sampling_time` 越大，不确定性估计越准确，但计算时间也会相应增加。

## 支持的模型和数据集

### 支持的视觉语言模型

- **Qwen2-VL**: `Qwen2-VL-72B-Instruct`, `Qwen2-VL-7B-Instruct`, `Qwen2-VL-2B-Instruct`
- **InternVL**: `InternVL2-26B`, `InternVL2-8B`, `InternVL2-1B`
- **LLaVA**: `llava-1.5-13b-hf`, `llava-1.5-7b-hf`
- **LLaVA-NeXT**: `llava-v1.6-vicuna-13b-hf`, `llava-v1.6-mistral-7b-hf`

### 支持的基准测试数据集

- **MMVet**: 多模态视觉理解基准测试
- **LLaVABench**: LLaVA 基准测试（In-the-Wild）
- **MMMU**: 多模态多任务理解基准测试
- **ScienceQA**: 科学问答基准测试

## 文件结构

```
vl_uncertainty/
├── README.md                    # 本文档
├── pipeline.py                  # 主流程文件
├── __init__.py
├── benchmark/                   # 基准测试数据集接口
│   ├── MMVet.py
│   ├── LLaVABench.py
│   ├── MMMU.py
│   └── ScienceQA.py
├── llm/                         # LLM 接口
│   └── Qwen.py
└── util/                        # 工具函数
    ├── visual_perturbation.py   # 视觉扰动函数
    ├── textual_perturbation.py  # 文本扰动函数
    └── misc.py                  # 其他工具函数
```

