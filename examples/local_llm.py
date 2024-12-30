import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from browser_use import Agent, Browser, BrowserConfig
from browser_use.llm.config import LLMConfig
from browser_use.llm.factory import LLMFactory

async def main():
    # Configure browser to run headless
    browser = Browser(
        config=BrowserConfig(
            headless=True,  # Force headless mode
            disable_security=True  # Helpful for some websites
        )
    )
    
    # Configure your LLM
    config = LLMConfig(
        model_name="facebook/opt-125m",  # Use smaller model for testing
        model_type="hf_pipeline",
        device_map="auto",
        torch_dtype="float16",
        cache_dir="./models",
        temperature=0.7  # Increase temperature for more varied responses
    )
    
    # Create LLM instance
    try:
        llm = LLMFactory.create_llm(config)
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return
    
    # Initialize agent with headless browser
    agent = Agent(
        task="Search Google for 'OpenAI' and tell me the title of the first result",
        llm=llm,
        browser=browser  # Pass the configured browser
    )
    
    try:
        await agent.run()
    except Exception as e:
        print(f"Agent execution failed: {e}")
    finally:
        await browser.close()  # Ensure browser is closed

if __name__ == "__main__":
    asyncio.run(main()) 