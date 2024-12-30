from typing import Union
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_community.llms.vllm import VLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .config import LLMConfig

class LLMFactory:
    """Factory class for creating LLM instances"""
    
    @staticmethod
    def create_llm(config: LLMConfig) -> Union[HuggingFacePipeline, VLLM]:
        """Create an LLM instance based on configuration"""
        if config.model_type == "hf_pipeline":
            return LLMFactory._create_hf_pipeline(config)
        elif config.model_type == "vllm":
            if not torch.cuda.is_available():
                raise RuntimeError("vLLM requires CUDA to be available")
            return LLMFactory._create_vllm(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    @staticmethod
    def _create_hf_pipeline(config: LLMConfig) -> HuggingFacePipeline:
        """Create a HuggingFace pipeline LLM"""
        # Convert dtype if string
        dtype = config.torch_dtype
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                cache_dir=config.cache_dir,
                trust_remote_code=config.trust_remote_code
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                device_map=config.device_map,
                torch_dtype=dtype,
                trust_remote_code=config.trust_remote_code,
                cache_dir=config.cache_dir
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                return_full_text=True
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace pipeline: {str(e)}")

    @staticmethod
    def _create_vllm(config: LLMConfig) -> VLLM:
        """Create a vLLM instance"""
        if not torch.cuda.is_available():
            raise RuntimeError("vLLM requires CUDA GPU to be available")
            
        try:
            return VLLM(
                model=config.model_name,
                trust_remote_code=config.trust_remote_code,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                client=None
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM: {str(e)}") 