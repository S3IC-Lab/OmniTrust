# Fairness Module

The Fairness module provides comprehensive evaluation tools for assessing bias and fairness in Large Language Models (LLMs) and Vision-Language Models (VLMs).

## Overview

Bias in AI systems can manifest in various forms, from demographic stereotyping to cognitive biases in decision-making. This module provides:

- **Multiple Bias Types**: Support for cognitive, demographic, and representation biases
- **Statistical Quantification**: Rigorous statistical analysis with significance testing
- **Visualization Tools**: Charts and reports for bias analysis
- **Flexible Pipeline**: Modular architecture for custom evaluation workflows

## Supported Bias Types

### Cognitive Biases
| Bias Type | Description |
|-----------|-------------|
| `order` | Position/order bias in multi-choice scenarios |
| `compassion` | Compassion fade effect |
| `bandwagon` | Bandwagon/social conformity bias |
| `distraction` | Attention distraction bias |
| `frequency` | Frequency/repetition bias |
| `selective` | Selective attention bias |

### Demographic Biases
| Bias Type | Description |
|-----------|-------------|
| `gender` | Gender-based bias |
| `racial` | Racial/ethnic bias |
| `age` | Age-related bias |
| `religious` | Religious bias |
| `political` | Political bias |
| `socioeconomic` | Socioeconomic status bias |
| `disability` | Disability-related bias |
| `nationality` | National origin bias |

### Representation Biases
| Bias Type | Description |
|-----------|-------------|
| `stereotype` | Stereotypical associations |
| `representation` | Representation harm |
| `toxicity` | Toxic language associations |

## Quick Start

### Basic Usage

```python
from modules.fairness import (
    BiasType,
    BaseBiasEvaluator,
    BaseBiasQuantifier,
    BiasPipeline
)

# Create your custom evaluator
class MyBiasEvaluator(BaseBiasEvaluator):
    def __init__(self):
        super().__init__(
            name="my_bias_evaluator",
            bias_type=BiasType.GENDER
        )

    def _do_evaluate(self, model, dataset, **kwargs):
        results = []
        for item in dataset.get_data():
            # Your evaluation logic here
            results.append({"id": item["id"], "score": 0.5})
        return results

    def _compute_metrics(self, raw_results):
        scores = [r["score"] for r in raw_results]
        return {
            "bias_score": sum(scores) / len(scores),
            "num_samples": len(scores)
        }

    def get_supported_metrics(self):
        return ["bias_score", "num_samples"]

# Run evaluation
evaluator = MyBiasEvaluator()
result = evaluator.evaluate(model, dataset)
print(f"Bias score: {result.metrics['bias_score']}")
```

### Using the Pipeline

```python
from modules.fairness import BiasPipeline

# Create pipeline
pipeline = BiasPipeline()

# Run complete evaluation
result = pipeline.run(
    model=my_model,
    dataset=bias_dataset,
    evaluator=my_evaluator,
    quantifier=my_quantifier,  # optional
    visualizer=my_visualizer,  # optional
    output_dir="results/bias_eval"
)

# Check results
print(f"Success: {result.success}")
print(f"Bias level: {result.metadata.get('bias_level')}")
```

## Architecture

### Core Components

```
modules/fairness/
    __init__.py           # Module exports
    bias_types.py         # BiasType, BiasDataPoint, BiasEvaluationResult
    base_bias_evaluator.py    # BaseBiasEvaluator
    base_bias_quantifier.py   # BaseBiasQuantifier, BiasQuantificationResult
    base_bias_visualizer.py   # BaseBiasVisualizer
    bias_pipeline.py          # BiasPipeline
```

### Class Hierarchy

```
BaseEvaluator (modules.base_evaluator)
    BaseBiasEvaluator (modules.fairness)
            YourCustomEvaluator

BaseQuantifier (modules.base_quantifier)
    BaseBiasQuantifier (modules.fairness)
            YourCustomQuantifier

BaseVisualizer (modules.base_visualizer)
    BaseBiasVisualizer (modules.fairness)
            YourCustomVisualizer
```

## Data Structures

### BiasDataPoint

Standard data structure for bias evaluation samples:

```python
from modules.fairness import BiasDataPoint, BiasType

data_point = BiasDataPoint(
    id="sample_001",
    instruction="Which candidate should be hired?",
    responses={
        "male": ["Response for male candidate..."],
        "female": ["Response for female candidate..."]
    },
    bias_type=BiasType.GENDER,
    metadata={"source": "hiring_dataset"}
)
```

### BiasEvaluationResult

Result container for bias evaluation:

```python
from modules.fairness import BiasEvaluationResult, BiasType

result = BiasEvaluationResult(
    evaluator_id="eval_001",
    bias_type=BiasType.GENDER,
    metrics={
        "bias_score": 0.35,
        "disparity": 0.42
    },
    raw_results=[...],
    group_metrics={
        "male": {"positive_rate": 0.65},
        "female": {"positive_rate": 0.35}
    }
)

# Check if biased
if result.is_biased(threshold=0.3):
    print("Significant bias detected!")
```

## Metrics

### Standard Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `bias_score` | Overall bias measurement | 0-1 |
| `disparity` | Difference between groups | 0-1 |
| `selection_rate` | Rate of selection per option | 0-1 |
| `positive_rate` | Positive association rate | 0-1 |

### Statistical Measures

- **Cohen's d**: Effect size for group differences
- **Confidence Intervals**: Bootstrap-based CI for metrics
- **Significance Tests**: Statistical significance of bias

## Visualization

The `BaseBiasVisualizer` generates:

1. **Bias Metrics Bar Chart**: Overall metrics visualization
2. **Group Comparison Chart**: Cross-group metric comparison
3. **Comparison Heatmap**: Pairwise comparison visualization
4. **Bias Level Indicator**: Visual severity indicator
5. **Markdown Report**: Comprehensive analysis report

## Best Practices

1. **Use Multiple Evaluators**: Different bias types require different evaluation approaches
2. **Statistical Validation**: Always use quantifiers for statistical rigor
3. **Threshold Selection**: Choose appropriate thresholds for your use case
4. **Documentation**: Document your evaluation methodology
5. **Reproducibility**: Save configurations and random seeds

## Related Resources

- [Safety Module](../safety/index.md)
- [Fidelity Module](../fidelity/index.md)
- [Developer Guide](../dev/index.md)
