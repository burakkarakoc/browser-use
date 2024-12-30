from dataclasses import dataclass
from typing import Literal, Optional, Union
import torch

@dataclass
class LLMConfig:
    """Configuration for local LLMs"""
    model_name: str
    model_type: Literal["hf_pipeline", "vllm"]
    device_map: str = "auto"
    torch_dtype: Union[str, torch.dtype] = "auto"
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None