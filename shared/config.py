"""Pydantic configuration schemas for all AgentY pipelines."""

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
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
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
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"


class HostingConfig(BaseModel):
    model_path: str
    adapter_path: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8001
    transport: str = "http"  # http | stdio
