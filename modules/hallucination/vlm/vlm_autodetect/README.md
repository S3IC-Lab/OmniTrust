# VLM Auto-Detect 幻觉检测框架

## 简介

VLM Auto-Detect 是一个多维度视觉语言模型（VLM）幻觉检测框架。该框架通过分析模型生成的响应，从多个维度（生成式任务、判别式任务等）评估和检测潜在的幻觉。

## 功能特性

- **多维度检测**：支持生成式任务（Generative）和判别式任务（Discriminative）的幻觉检测
- **LLM-free 检测**：无需额外的语言模型，直接基于响应内容进行检测
- **多模型支持**：支持 Qwen2-VL、LLaVA、LLaVA-NeXT 等多种视觉语言模型
- **灵活的评估类型**：支持多种评估维度组合

## 安装要求

确保已安装以下依赖：

```bash
pip install torch transformers pillow tqdm nltk spacy
```

安装 spaCy 英文模型：

```bash
python -m spacy download en_core_web_lg
```

下载 NLTK 数据包（如果需要）：

```python
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
```

## 数据集准备

### 1. 下载数据集文件

在使用框架之前，需要先准备以下数据集文件：

#### 1.1 下载 `query_all.json`

将 `query_all.json` 文件下载到以下目录：

```
OmniTrust/modules/hallucination/vlm/vlm_autodetect/query/query_all.json
```

#### 1.2 下载图像文件

根据 `query_all.json` 中的图像路径信息，将图像文件下载到对应的目录。默认情况下，图像文件应存放在：

```
OmniTrust/data/dataset/halu_autodetect/image/
```

**注意**：`query_all.json` 中的 `filename` 字段指向的是相对于项目根目录的路径，例如：
```json
{
    "id": 1,
    "filename": "data/dataset/halu_autodetect/image/AMBER_1.jpg",
    "question": "Describe this image."
}
```

因此，图像文件应按照此路径结构组织。

### 2. 数据集目录结构

准备完成后，目录结构应如下：

```
OmniTrust/
├── modules/
│   └── hallucination/
│       └── vlm/
│           └── vlm_autodetect/
│               ├── query/
│               │   └── query_all.json          # 数据集文件
│               ├── utils/
│               │   ├── relation.json           # 词关联文件
│               │   ├── safe_words.txt         # 安全词列表
│               │   ├── annotations.json        # 标注文件
│               │   └── metrics.txt             # 指标定义
│               └── exp/                        # 结果输出目录（自动创建）
└── data/
    └── dataset/
        └── halu_autodetect/
            └── image/                          # 图像文件目录
                ├── AMBER_1.jpg
                ├── AMBER_2.jpg
                └── ...
```

## 使用方法

### 基本调用

通过 `OmniTrust/examples/hallucination.py` 调用：

```bash
python OmniTrust/examples/hallucination.py \
    --method auto-detect \
    --model-type vlm \
    --model-name llava-1.5-13b-hf \
    --autodetect-type d \
    --data_path_dir modules/hallucination/vlm/vlm_autodetect/query \
    --model_path_dir ~/models
```

### 参数说明

#### 必需参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--method` | str | - | 必须设置为 `auto-detect` |
| `--model-type` | str | - | 必须设置为 `vlm` |
| `--model-name` | str | `llava-1.5-13b-hf` | 模型名称或路径。支持：`llava-1.5-13b-hf`, `llava-1.5-7b-hf`, `llava-v1.6-vicuna-13b-hf`, `Qwen2-VL-2B-Instruct`, `Qwen2-VL-7B-Instruct`, `Qwen2-VL-72B-Instruct` 等 |

#### 检测类型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--autodetect-type` | str | `d` | LLM-free 检测类型。可选值：<br>- `a`: 所有维度（all）<br>- `g`: 生成式任务（generative）<br>- `d`: 判别式任务（discriminative，包含 de、da、dr）<br>- `de`: 存在性检测（existence）<br>- `da`: 属性检测（attribute）<br>- `dr`: 关系检测（relation） |

#### 路径参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path_dir` | str | `None` | 模型路径目录（可选）。如果提供，将从本地路径加载模型；否则从 HuggingFace 加载 |
| `--data_path_dir` | str | `modules/hallucination/vlm/vlm_autodetect/query` | 数据集路径目录。指向包含 `query_all.json` 的目录 |

#### 配置文件参数（可选）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--word-association` | str | `None` | 词关联文件路径。默认使用 `utils/relation.json` |
| `--safe-words` | str | `None` | 安全词列表文件路径。默认使用 `utils/safe_words.txt` |
| `--annotation` | str | `None` | 标注文件路径。默认使用 `utils/annotations.json` |
| `--metrics` | str | `None` | 指标定义文件路径。默认使用 `utils/metrics.txt` |
| `--similarity-score` | float | `0.8` | 相似度分数阈值，用于同义词检测 |

## 使用示例

### 示例 1：使用默认参数检测判别式任务

```bash
python OmniTrust/examples/hallucination.py \
    --method auto-detect \
    --model-type vlm \
    --model-name llava-1.5-13b-hf \
    --autodetect-type d
```

### 示例 2：使用本地模型路径

```bash
python OmniTrust/examples/hallucination.py \
    --method auto-detect \
    --model-type vlm \
    --model-name Qwen2-VL-2B-Instruct \
    --model_path_dir ~/models \
    --autodetect-type d
```

### 示例 3：检测生成式任务

```bash
python OmniTrust/examples/hallucination.py \
    --method auto-detect \
    --model-type vlm \
    --model-name llava-v1.6-vicuna-13b-hf \
    --autodetect-type g \
    --data_path_dir /path/to/your/query/directory
```

### 示例 4：检测所有维度

```bash
python OmniTrust/examples/hallucination.py \
    --method auto-detect \
    --model-type vlm \
    --model-name llava-1.5-13b-hf \
    --autodetect-type a \
    --similarity-score 0.7
```

### 示例 5：使用自定义配置文件路径

```bash
python OmniTrust/examples/hallucination.py \
    --method auto-detect \
    --model-type vlm \
    --model-name llava-1.5-13b-hf \
    --autodetect-type d \
    --word-association /path/to/custom/relation.json \
    --annotation /path/to/custom/annotations.json
```

## 输出结果

### 结果存储位置

运行完成后，检测结果将自动保存在以下目录：

```
modules/hallucination/vlm/vlm_autodetect/exp/
```

如果该目录不存在，程序会自动创建。结果文件命名格式为：`log_YYYY_MM_DD_HH_MM_SS.json`（例如：`log_2025_01_15_14_30_25.json`）

### 结果文件内容

结果文件包含评估指标，根据 `--autodetect-type` 的不同，可能包含：

#### 生成式任务指标（`g` 或 `a`）

- `CHAIR`: CHAIR 分数（越低越好）
- `Cover`: 覆盖率（越高越好）
- `Hal`: 幻觉率（越低越好）
- `Cog`: 认知一致性（越低越好）

#### 判别式任务指标（`d`、`de`、`da`、`dr` 或 `a`）

- `Accuracy`: 准确率
- `Precision`: 精确率
- `Recall`: 召回率
- `F1`: F1 分数

### 控制台输出示例

```
Generative Task:
CHAIR:		15.2
Cover:		85.3
Hal:		12.5
Cog:		8.7

Descriminative Task:
Accuracy:	78.5
Precision:	82.1
Recall:		75.3
F1:		78.6
```

## 注意事项

1. **数据集路径**：
   - 确保 `query_all.json` 文件位于正确的目录
   - 确保图像文件路径与 `query_all.json` 中的 `filename` 字段匹配
   - 如果使用自定义 `--data_path_dir`，确保该目录下包含 `query_all.json`

2. **模型路径**：
   - 如果使用本地模型，确保 `--model_path_dir` 指向正确的模型目录
   - 模型应按照 `{model_path_dir}/{model_name}` 的结构组织
   - 如果不提供 `--model_path_dir`，将从 HuggingFace 自动下载模型

3. **GPU 内存**：
   - 较大的模型（如 Qwen2-VL-72B）需要足够的 GPU 内存
   - 建议根据硬件配置选择合适的模型

4. **依赖环境**：
   - 确保已安装 spaCy 英文模型：`python -m spacy download en_core_web_lg`
   - 确保 NLTK 数据包已下载（如果需要）

5. **配置文件**：
   - 默认配置文件位于 `utils/` 目录
   - 如需自定义，可通过相应参数指定路径
   - 如果配置文件不存在，程序会使用默认指标值

## 支持的模型

### 视觉语言模型

- **Qwen2-VL**: `Qwen2-VL-72B-Instruct`, `Qwen2-VL-7B-Instruct`, `Qwen2-VL-2B-Instruct`
- **LLaVA**: `llava-1.5-13b-hf`, `llava-1.5-7b-hf`
- **LLaVA-NeXT**: `llava-v1.6-vicuna-13b-hf`, `llava-v1.6-mistral-7b-hf`

## 文件结构

```
vlm_autodetect/
├── README.md                    # 本文档
├── pipeline.py                  # 主流程文件
├── query/                       # 数据集目录
│   └── query_all.json          # 数据集文件（需下载）
├── utils/                       # 工具文件目录
│   ├── relation.json           # 词关联文件
│   ├── safe_words.txt          # 安全词列表
│   ├── annotations.json        # 标注文件
│   └── metrics.txt             # 指标定义文件
└── exp/                        # 结果输出目录（自动创建）
    └── log_*.json              # 检测结果文件
```

## 故障排除

### 问题 1：找不到数据集文件

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: '.../query_all.json'
```

**解决方法**：
- 检查 `--data_path_dir` 参数是否正确
- 确保 `query_all.json` 文件存在于指定目录
- 检查文件路径是否正确

### 问题 2：找不到图像文件

**错误信息**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/dataset/halu_autodetect/image/...'
```

**解决方法**：
- 检查图像文件是否已下载到 `data/dataset/halu_autodetect/image/` 目录
- 确保图像路径与 `query_all.json` 中的 `filename` 字段匹配
- 检查图像文件权限

### 问题 3：spaCy 模型未找到

**错误信息**：
```
OSError: Can't find model 'en_core_web_lg'
```

**解决方法**：
```bash
python -m spacy download en_core_web_lg
```

### 问题 4：配置文件缺失

**错误信息**：
```
Warning: Required data files not found.
```

**解决方法**：
- 检查 `utils/` 目录下的配置文件是否存在
- 或通过参数指定自定义配置文件路径

