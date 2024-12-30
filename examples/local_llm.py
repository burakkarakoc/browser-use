"""
Local LLM integration for browser-use.
Example usage:

from browser_use.llm import create_local_agent

agent = create_local_agent(
    task="Search for 'Python programming' and get the first result",
    model_name="Qwen/Qwen1.5-7B-Chat",
    model_type="hf_pipeline"
)
await agent.run()
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Literal, Optional
from browser_use import Agent, Browser, BrowserConfig
from browser_use.llm.config import LLMConfig
from browser_use.llm.factory import LLMFactory

def create_local_agent(
    task: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    model_type: Literal["hf_pipeline", "vllm"] = "hf_pipeline",
    device_map: str = "auto",
    torch_dtype: str = "float16",
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    cache_dir: Optional[str] = "./models",
    headless: bool = True,
) -> Agent:
    """
    Create an agent using a local LLM.
    
    Args:
        task: The task for the agent to perform
        model_name: Name of the model from HuggingFace
        model_type: Type of model loading ("hf_pipeline" or "vllm")
        device_map: Device to load model on ("auto", "cpu", "cuda:0", etc.)
        torch_dtype: Model precision ("float16", "float32", etc.)
        temperature: Sampling temperature (0.0 to 1.0)
        max_new_tokens: Maximum new tokens to generate
        cache_dir: Directory to cache models
        headless: Whether to run browser in headless mode
    
    Returns:
        Agent: Configured browser-use agent
    """
    # Configure browser
    browser = Browser(
        config=BrowserConfig(
            headless=headless,
            disable_security=True
        )
    )
    
    # Configure LLM
    config = LLMConfig(
        model_name=model_name,
        model_type=model_type,
        device_map=device_map,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    
    # Create LLM instance
    try:
        llm = LLMFactory.create_llm(config)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM: {e}")
    
    # Create and return agent
    return Agent(
        task=task,
        llm=llm,
        browser=browser
    )

# Example usage
async def main():
    agent = create_local_agent(
        task="Search Google for 'OpenAI' and tell me the title of the first result",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        headless=True
    )
    
    try:
        await agent.run()
    except Exception as e:
        print(f"Agent execution failed: {e}")
    finally:
        await agent.browser.close()

if __name__ == "__main__":
    asyncio.run(main()) 