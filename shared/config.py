"""Pydantic configuration schemas for all MCP Tuna pipelines."""

from typing import Dict, List, Optional
from pydantic import BaseModel


class PipelineConfig(BaseModel):
    model: str = "gpt-4o"
    debug: bool = False


class GeneratorConfig(PipelineConfig):
    technique: str = "sft"
    output_dir: str = "./output"
    custom_template: Optional[str] = None


class EvaluatorConfig(PipelineConfig):
    weights: Dict[str, float] = {"complexity": 0.3, "ifd": -0.2, "quality": 0.9}
    threshold: float = 0.7
    language: str = "en"


class CleaningConfig(BaseModel):
    remove_duplicates: bool = True
    min_instruction_length: int = 10
    min_output_length: int = 20
    remove_empty_fields: bool = True


class NormalizationConfig(BaseModel):
    target_format: str = "sft"  # sft | dpo | grpo
    merge_instruction_input: bool = True
    strip_whitespace: bool = True


class FinetuningConfig(PipelineConfig):
    base_model: str = "Qwen/Qwen3-1.7B"
    num_epochs: int = 3
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16


class OrchestrationConfig(PipelineConfig):
    num_problems: int = 50
    n_per_problem: int = 4
    temperatures: List[float] = [0.3, 0.5, 0.7, 1.0]
    cost_budget: float = 1.0          # USD per task
    latency_budget: float = 60.0       # seconds per task
    reward_weights: Dict[str, float] = {"accuracy": 0.5, "cost": 0.25, "latency": 0.25}
    output_format: str = "sft"         # sft | dpo | grpo
    base_model: str = "Qwen/Qwen3-1.7B"


class ModelEvaluationConfig(PipelineConfig):
    metrics: List[str] = ["rouge", "bertscore", "llm_judge"]
    max_new_tokens: int = 1024
    temperature: float = 0.1
    judge_model: str = "gpt-4o"
    bertscore_model: str = "roberta-large"
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class AdvancedJudgeConfig(PipelineConfig):
    """Configuration for the advanced LLM-as-a-judge evaluation system."""
    default_judge_type: str = "pointwise"
    default_judge_model: str = "gpt-4o"
    default_aggregation: str = "mean"
    timeout_s: float = 30.0


class FTEvaluatorConfig(PipelineConfig):
    """Configuration for domain knowledge fine-tune evaluation."""
    judge_models: List[str] = ["gpt-4o"]
    temperature: float = 0.0
    max_tokens: int = 2048
    system_prompt_path: Optional[str] = None


class HostingConfig(BaseModel):
    model_path: str
    adapter_path: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8001
    transport: str = "http"  # http | stdio
    quantization: Optional[str] = None  # None | "4bit" | "8bit"


class ChatConfig(BaseModel):
    endpoint: Optional[str] = None
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    streaming: bool = True
