from dataclasses import dataclass, field
from typing import Optional, Dict

# Task Configuration: Task-related settings like task type, GPUs, etc.
@dataclass
class TaskArgs:
    taskType: Optional[str] = "judge"  # Default task type
    num_choices: Optional[int] = 1
    num_gpus_per_model: Optional[int] = 8
    num_gpus_total: Optional[int] = 1
    max_gpu_memory: Optional[str] = None
    dtype: Optional[str] = "float16"  # Default is float16
    revision: Optional[str] = "main"  # Default revision is "main"

# Task Configuration: Task-related settings like task type, GPUs, etc.
@dataclass
class WtTaskArgs:
    taskType: Optional[str] = "watermark"  # Default task type
    num_choices: Optional[int] = 1
    num_gpus_per_model: Optional[int] = 8
    num_gpus_total: Optional[int] = 1
    max_gpu_memory: Optional[str] = None
    dtype: Optional[str] = "float16"  # Default is float16
    revision: Optional[str] = "main"  # Default revision is "main"

# LLM Model Configuration: Parameters related to the model
@dataclass
class LLMArgs:
    model_path: Optional[str] = None
    model_id: Optional[str] = None
    max_new_tokens: Optional[int] = 1024
    judge_model: Optional[str] = None
    baseline_model: Optional[str] = None
    num_choices: Optional[int] = 1
    num_gpus_per_model: Optional[int] = 8
    num_gpus_total: Optional[int] = 1
    max_gpu_memory: Optional[str] = None
    dtype: Optional[str] = "float16"
    revision: Optional[str] = "main"

# Data Configuration: Parameters related to the input/output data files
@dataclass
class DataArgs:
    input_file: Optional[str] = None
    outputs_file: Optional[str] = None
    ref_file: Optional[str] = None
    test_file: Optional[str] = None
    prompt_file: Optional[str] = None

# Debug Configuration: Parameters for debugging and testing
@dataclass
class DebugArgs:
    question_begin: Optional[int] = None
    question_end: Optional[int] = None

# Base configuration: Aggregate all other configurations
@dataclass
class BaseConfig:
    taskArgs: TaskArgs = field(default_factory=TaskArgs)
    llmArgs: LLMArgs = field(default_factory=LLMArgs)
    dataArgs: DataArgs = field(default_factory=DataArgs)
    debugArgs: DebugArgs = field(default_factory=DebugArgs)
