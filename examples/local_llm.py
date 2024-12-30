import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from browser_use import Agent
from browser_use.llm.config import LLMConfig
from browser_use.llm.factory import LLMFactory

async def main():
    # Configure your LLM
    config = LLMConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        model_type="hf_pipeline",  # or "vllm" if you have CUDA
        device_map="auto",         # Will use GPU if available
        torch_dtype="float16",     # Use half precision to save memory
        cache_dir="./models"
    )
    
    # Create LLM instance
    try:
        llm = LLMFactory.create_llm(config)
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return
    
    # Initialize agent
    agent = Agent(
        task="Search Google for 'OpenAI' and tell me the title of the first result",
        llm=llm
    )
    
    try:
        await agent.run()
    except Exception as e:
        print(f"Agent execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 