# OmniTrust Fairness (Bias) 模块开源开发计划

## 项目概述

**目标**: 基于 OmniTrust 架构独立开发 Fairness (Bias) 评估模块

**开发策略**: **完全独立重写**，不依赖原框架代码，参考原实现逻辑

**时间规划**:
- **第一周**: 开发阶段 (3人)
- **第二周**: 测试阶段 (2人)

**核心功能**:
- **12种偏见评估器**:
  - 6种认知偏见: order, compassion, bandwagon, distraction, selective, frequency
  - 6种社会偏见: age, gender, race, religion, nationality, political (政治归类为社会偏见)
- **2种量化器**: cognitive, social
- **2种可视化器**: single, combined
- **3种数据集**: FlipBias, CognitiveBias(基于原bias50更改), SocietyBias（基于原bias_society更改）
- **报告系统**: 支持 Markdown、PDF 格式导出

---

## 第一周：开发阶段 (3人)

### 人员分工

| 开发者        | 负责模块 | 工作量占比 |
|------------|----------|-----------|
| **A: 甘佳灵** | 核心架构 + 认知偏见评估器 | 40% |
| **B: 陈浩**  | 社会偏见评估器 + 可视化器 | 35% |
| **C: 徐菡艺** | 数据集 + 报告 + CLI集成 | 25% |

---

### Day 1-2: 核心架构搭建

#### 开发者A: 核心接口与注册表
**文件**: `modules/fairness/core/`

| 任务 | 输出文件 | 优先级 |
|------|---------|--------|
| 统一注册表实现 | `registry.py` | P0 |
| 配置加载器 | `config_loader.py` | P0 |
| 模型包装器 | `model_wrapper.py` | P0 |

#### 开发者B: 量化器基础架构
**文件**: `modules/fairness/quantifiers/`

| 任务 | 输出文件 | 优先级 |
|------|---------|--------|
| 量化器基类完善 | `base.py` | P0 |
| 统计工具函数 | `stats_utils.py` | P0 |

#### 开发者C: 数据集基础架构
**文件**: `modules/fairness/datasets/`

| 任务 | 输出文件 | 优先级 |
|------|---------|--------|
| 数据集基类 | `base.py` | P0 |
| 数据加载工具 | `loader.py` | P0 |

---

### Day 3-4: 评估器实现

#### 开发者A: 认知偏见评估器 (6种)
**文件**: `modules/fairness/evaluators/cognitive/`

| 评估器 | 输出文件 | 评估逻辑 |
|--------|---------|---------|
| OrderBiasEvaluator | `order.py` | 选项位置偏好检测 |
| CompassionBiasEvaluator | `compassion.py` | 同情心衰减效应检测 |
| BandwagonBiasEvaluator | `bandwagon.py` | 从众效应检测 |
| DistractionBiasEvaluator | `distraction.py` | 注意力分散偏见检测 |
| SelectiveBiasEvaluator | `selective.py` | 选择性注意偏见检测 |
| FrequencyBiasEvaluator | `frequency.py` | 频率/重复偏见检测 |

**共享基类**: `CognitiveBiasEvaluator(BaseBiasEvaluator)`
- 支持成对比较评估模式
- 标准化的 prompt 模板
- 响应解析与分数计算

#### 开发者B: 社会偏见评估器 (6种，含政治偏见)
**文件**: `modules/fairness/evaluators/social/`

| 评估器 | 输出文件 | 评估逻辑 |
|--------|---------|---------|
| AgeBiasEvaluator | `age.py` | 年龄刻板印象检测 |
| GenderBiasEvaluator | `gender.py` | 性别刻板印象检测 |
| RaceBiasEvaluator | `race.py` | 种族刻板印象检测 |
| ReligionBiasEvaluator | `religion.py` | 宗教刻板印象检测 |
| NationalityBiasEvaluator | `nationality.py` | 国籍刻板印象检测 |
| PoliticalBiasEvaluator | `political.py` | 政治立场偏见检测 |

**共享基类**: `SocialBiasEvaluator(BaseBiasEvaluator)`
- 刻板印象同意率计算
- 多人口群体对比分析
- 情感倾向识别

#### 开发者C: 数据集实现
**文件**: `modules/fairness/datasets/`

| 数据集 | 输出文件 | 用途 |
|--------|---------|------|
| FlipBiasDataset | `flipbias.py` | 政治偏见评估 |
| CognitiveBiasDataset | `cognitive.py` | 认知偏见评估 (bias_demo, bias50) |
| SocialBiasDataset | `social.py` | 社会偏见评估 (bias_society) |
| BaseDatasetLoader | `loader.py` | 通用数据加载工具 |

---

### Day 5: 量化器实现

#### 开发者A: 认知偏见量化器
**文件**: `modules/fairness/quantifiers/cognitive.py`

**核心功能**:
- 二项检验 (binomial test) 统计显著性
- Wilson score 置信区间计算
- 首位/末位偏好率 (first_order_bias_rate, last_order_bias_rate)
- 总偏见率、ME偏见率、一致性率
- 效应量 (Cohen's d) 计算

#### 开发者B: 社会偏见量化器
**文件**: `modules/fairness/quantifiers/social.py`

**核心功能**:
- 刻板印象同意率计算
- 政治倾向分布 (Left/Center/Right)
- 多群体公平性指标
- 偏见分数等级解释 (0-20% 非常低 → 80-100% 非常高)

#### 开发者C: 配置文件与统计工具
**文件**: `modules/fairness/`

| 任务 | 输出文件 |
|------|---------|
| 统计工具函数 | `quantifiers/stats_utils.py` |
| 评估器配置 | `config/evaluators.yaml` |
| 量化器配置 | `config/quantifiers.yaml` |
| 数据集配置 | `config/datasets.yaml` |
| 示例数据 | `examples/sample_data/` |

---

### Day 6: 可视化器与报告

#### 开发者A: 可视化器实现
**文件**: `modules/fairness/visualizers/`

| 可视化器 | 输出文件 | 功能 |
|---------|---------|------|
| SingleVisualizer | `single.py` | 单评估结果可视化 (条形图、饼图) |
| CombinedVisualizer | `combined.py` | 多评估结果对比可视化 |
| ChartUtils | `chart_utils.py` | 通用图表工具函数 (matplotlib封装) |

**图表类型**:
- 偏见分数条形图
- 响应分布饼图
- 群体对比热力图
- 雷达图 (多维偏见分布)

#### 开发者B: 报告生成器 (Markdown + PDF)
**文件**: `modules/fairness/report/`

| 报告生成器 | 输出文件 | 功能 |
|-----------|---------|------|
| MarkdownReportGenerator | `markdown.py` | Markdown格式报告 |
| PDFReportGenerator | `pdf.py` | PDF导出 (weasyprint/pdfkit) |
| UnifiedReportGenerator | `unified.py` | 多类型综合报告 |

**报告内容**:
1. 执行概要 (模型、数据集、偏见类型)
2. 详细评估结果表格
3. 可视化图表嵌入
4. 偏见分数解释
5. 改进建议

#### 开发者C: CLI入口与示例
**文件**: `modules/fairness/cli/`

| 任务 | 输出文件 |
|------|---------|
| 命令行入口 | `run_evaluation.py` |
| 批量评估 | `batch_evaluate.py` |
| 快速测试脚本 | `examples/quick_test.py` |
| 完整示例 | `examples/full_evaluation.py` |

**CLI 参数支持**:
```bash
python -m modules.fairness.cli.run_evaluation \
  --model llama3.2:1b \
  --evaluator order,gender \
  --dataset bias_demo \
  --output-dir results/ \
  --limit 10 \
  --pdf
```

---

### Day 7: 集成与文档

#### 开发者A: 模块集成
| 任务 | 说明 |
|------|------|
| `__init__.py` 导出 | 确保所有组件可导入 |
| 依赖检查 | 验证外部依赖 |
| 向后兼容 | 确保与现有基础架构兼容 |

#### 开发者B: API文档
| 任务 | 输出文件 |
|------|---------|
| 评估器API文档 | `docs/fairness/evaluators.md` |
| 量化器API文档 | `docs/fairness/quantifiers.md` |
| 可视化API文档 | `docs/fairness/visualizers.md` |

#### 开发者C: 用户指南
| 任务 | 输出文件 |
|------|---------|
| 快速开始 | `docs/fairness/quickstart.md` |
| 配置指南 | `docs/fairness/configuration.md` |
| 扩展指南 | `docs/fairness/extending.md` |

---

## 第一周交付物清单

### 目录结构
```
modules/fairness/
├── __init__.py                      # 模块导出
├── bias_types.py                    # 偏见类型定义 (已有)
├── base_bias_evaluator.py           # 评估器基类 (已有)
├── base_bias_quantifier.py          # 量化器基类 (已有)
├── base_bias_visualizer.py          # 可视化器基类 (已有)
├── bias_pipeline.py                 # 评估流水线 (已有)
│
├── core/                            # 核心组件 (新增)
│   ├── __init__.py
│   ├── registry.py                  # 组件注册表
│   ├── config_loader.py             # 配置加载器
│   └── model_wrapper.py             # 模型包装器
│
├── evaluators/                      # 评估器 (新增)
│   ├── __init__.py
│   ├── cognitive/                   # 认知偏见 (6种)
│   │   ├── __init__.py
│   │   ├── base.py                  # CognitiveBiasEvaluator
│   │   ├── order.py
│   │   ├── compassion.py
│   │   ├── bandwagon.py
│   │   ├── distraction.py
│   │   ├── selective.py
│   │   └── frequency.py
│   └── social/                      # 社会偏见 (6种，含政治)
│       ├── __init__.py
│       ├── base.py                  # SocialBiasEvaluator
│       ├── age.py
│       ├── gender.py
│       ├── race.py
│       ├── religion.py
│       ├── nationality.py
│       └── political.py
│
├── quantifiers/                     # 量化器 (新增)
│   ├── __init__.py
│   ├── stats_utils.py               # 统计工具 (二项检验、CI等)
│   ├── cognitive.py                 # 认知偏见量化器
│   └── social.py                    # 社会偏见量化器
│
├── visualizers/                     # 可视化器 (新增)
│   ├── __init__.py
│   ├── chart_utils.py               # 图表工具函数
│   ├── single.py                    # 单结果可视化
│   └── combined.py                  # 多结果对比可视化
│
├── datasets/                        # 数据集 (新增)
│   ├── __init__.py
│   ├── loader.py                    # 通用加载器
│   ├── cognitive.py                 # 认知偏见数据集
│   ├── social.py                    # 社会偏见数据集
│   └── flipbias.py                  # FlipBias政治数据集
│
├── report/                          # 报告生成 (新增)
│   ├── __init__.py
│   ├── markdown.py                  # Markdown报告
│   ├── pdf.py                       # PDF导出
│   └── unified.py                   # 统一报告生成器
│
├── cli/                             # 命令行工具 (新增)
│   ├── __init__.py
│   ├── run_evaluation.py            # 单次评估
│   └── batch_evaluate.py            # 批量评估
│
├── config/                          # 配置文件 (新增)
│   ├── evaluators.yaml
│   ├── quantifiers.yaml
│   └── datasets.yaml
│
└── examples/                        # 示例代码 (新增)
    ├── quick_test.py
    ├── full_evaluation.py
    └── sample_data/
```

### 功能矩阵

| 功能 | 数量 | 状态 |
|------|---|------|
| 认知偏见评估器 | 6 | 待开发 |
| 社会偏见评估器 | 6 | 待开发 |
| 量化器 | 2 (cognitive + social) | 待开发 |
| 可视化器 | 2 (single + combined) | 待开发 |
| 数据集 | 3 (cognitive + social + flipbias) | 待开发 |
| 报告生成器 | 3 (md + pdf + unified) | 待开发 |
| CLI工具 | 2 | 待开发 |

---

## 第二周：测试阶段 (2人)

### 人员分工

| 测试人员          | 负责内容 | 工作量占比 |
|---------------|---------|-----------|
| **测试者A: 谢晨宇** | 单元测试 + 集成测试 | 60% |
| **测试者B: 武春阳**    | 端到端测试 + 文档验证 | 40% |

---

### Day 8-9: 单元测试

#### 测试者A: 评估器单元测试
**文件**: `tests/fairness/test_evaluators/`

| 测试模块 | 测试文件 | 测试用例数 |
|---------|---------|-----------|
| 认知偏见评估器 | `test_cognitive.py` | ~30 |
| 社会偏见评估器 | `test_social.py` | ~25 |
| 政治偏见评估器 | `test_political.py` | ~10 |

**测试覆盖**:
- 正常输入测试
- 边界条件测试
- 异常输入测试
- Mock模型响应测试

#### 测试者B: 量化器+可视化器单元测试
**文件**: `tests/fairness/`

| 测试模块 | 测试文件 | 测试用例数 |
|---------|---------|-----------|
| 量化器 | `test_quantifiers.py` | ~20 |
| 可视化器 | `test_visualizers.py` | ~15 |
| 数据集 | `test_datasets.py` | ~15 |

---

### Day 10-11: 集成测试

#### 测试者A: 流水线集成测试
**文件**: `tests/fairness/test_integration/`

| 测试场景 | 测试文件 |
|---------|---------|
| 单评估器完整流程 | `test_single_evaluator.py` |
| 多评估器批量流程 | `test_batch_evaluation.py` |
| 评估器-量化器组合 | `test_evaluator_quantifier.py` |

#### 测试者B: 报告生成集成测试
**文件**: `tests/fairness/test_integration/`

| 测试场景 | 测试文件 |
|---------|---------|
| Markdown报告生成 | `test_report_markdown.py` |
| PDF报告生成 | `test_report_pdf.py` |
| 图表生成 | `test_chart_generation.py` |

---

### Day 12-13: 端到端测试

#### 测试者A: 真实模型测试
| 测试场景       | 模型          | 数据集 |
|------------|-------------|--------|
| Ollama本地模型 | llama3.2:1b | bias_demo |
| 阿里 API      | qwen_plus   | bias50 |
| 批量评估       | 多模型         | 多数据集 |

#### 测试者B: CLI测试 + 文档验证
| 测试内容 | 说明 |
|---------|------|
| CLI参数测试 | 所有命令行参数组合 |
| 帮助信息 | --help 输出验证 |
| 示例代码运行 | 文档中所有示例 |
| API文档验证 | 接口签名一致性 |

---

### Day 14: 回归测试与发布准备

#### 测试者A: 回归测试
| 任务 | 说明 |
|------|------|
| 完整测试套件运行 | pytest 全量运行 |
| 性能基准测试 | 执行时间、内存使用 |
| Bug修复验证 | 确认所有发现的问题已修复 |

#### 测试者B: 发布准备
| 任务 | 说明 |
|------|------|
| CHANGELOG更新 | 版本变更记录 |
| README更新 | 安装和使用说明 |
| 版本号标记 | 语义化版本 |

---

## 第二周交付物清单

### 测试目录结构
```
tests/fairness/
├── __init__.py
├── conftest.py                  # pytest fixtures
├── test_evaluators/
│   ├── test_cognitive.py
│   ├── test_social.py
│   └── test_political.py
├── test_quantifiers.py
├── test_visualizers.py
├── test_datasets.py
├── test_report.py
├── test_integration/
│   ├── test_single_evaluator.py
│   ├── test_batch_evaluation.py
│   ├── test_evaluator_quantifier.py
│   ├── test_report_markdown.py
│   ├── test_report_pdf.py
│   └── test_chart_generation.py
└── test_e2e/
    ├── test_ollama_evaluation.py
    ├── test_openai_evaluation.py
    └── test_cli.py
```

### 测试覆盖率目标

| 模块 | 目标覆盖率 |
|------|-----------|
| evaluators | ≥ 85% |
| quantifiers | ≥ 90% |
| visualizers | ≥ 80% |
| datasets | ≥ 85% |
| report | ≥ 80% |
| cli | ≥ 75% |
| **整体** | **≥ 80%** |

---

## 风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 原始框架接口变更 | 高 | 独立重写，不依赖原框架 |
| 模型API不稳定 | 中 | Mock测试 + 重试机制 |
| 中文字体问题 | 低 | 自动检测 + 降级方案 |
| PDF生成依赖 | 低 | 多种转换方案备选 (weasyprint/pdfkit/pandoc) |

---

## 里程碑

| 里程碑 | 日期 | 交付物 |
|--------|------|--------|
| M1: 核心架构完成 | Day 2 | 注册表、流水线、基类 |
| M2: 评估器完成 | Day 4 | 12种评估器 |
| M3: 量化+可视化完成 | Day 6 | 量化器、可视化器、报告 |
| M4: 开发完成 | Day 7 | 完整模块、文档 |
| M5: 单元测试完成 | Day 9 | 单元测试覆盖 |
| M6: 集成测试完成 | Day 11 | 集成测试通过 |
| M7: 端到端测试完成 | Day 13 | E2E测试通过 |
| M8: 发布就绪 | Day 14 | 版本发布 |

---

## 参考实现 (仅作逻辑参考，独立重写)

| 参考源文件 (AwesomeLLMSecurityPlatform) | 参考内容 |
|----------------------------------------|---------|
| `examples/llm_bias_rf/core/unified_registry.py` | 注册表设计模式 |
| `examples/llm_bias_rf/adapters/evaluators/bias_standard_adapter.py` | 认知偏见评估逻辑 |
| `examples/llm_bias_rf/adapters/evaluators/society_adapter.py` | 社会偏见评估逻辑 |
| `examples/llm_bias_rf/adapters/evaluators/political_adapter.py` | 政治偏见评估逻辑 |
| `examples/llm_bias_rf/plugins/quantifiers.py` | 统计检验和量化逻辑 |
| `examples/llm_bias_rf/plugins/visualizers.py` | 图表生成逻辑 |
| `utils/bias/report/unified_bias_report.py` | 报告生成模板 |
| `utils/bias/chart.py` | 可视化工具函数 |

---

## 依赖要求

### Python 依赖 (新增)
```
# requirements_fairness.txt
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
weasyprint>=60.0        # PDF导出
pdfkit>=1.0.0           # PDF备选方案
pyyaml>=6.0
```

### 可选依赖
```
# 中文字体支持
fonts-noto-cjk
# PDF工具
wkhtmltopdf             # pdfkit依赖
```
