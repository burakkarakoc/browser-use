import pytest
import torch
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.llm.config import LLMConfig
from browser_use.llm.factory import LLMFactory

@pytest.fixture
def browser():
    return Browser(config=BrowserConfig(headless=True))

@pytest.fixture
async def context(browser):
    async with await browser.new_context() as context:
        yield context

@pytest.fixture
def test_config():
    return LLMConfig(
        model_name="facebook/opt-125m",  # Small model for testing
        model_type="hf_pipeline",
        max_new_tokens=512,
        cache_dir="./.test_models"
    )

@pytest.mark.asyncio
async def test_hf_pipeline_basic_task(test_config, context):
    """Test basic task with HuggingFace pipeline"""
    llm = LLMFactory.create_llm(test_config)
    
    agent = Agent(
        task="Go to example.com and extract the title",
        llm=llm,
        browser_context=context,
    )
    
    history = await agent.run(max_steps=3)
    action_names = history.action_names()
    
    assert 'go_to_url' in action_names
    assert 'extract_content' in action_names

@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason="VLLM requires CUDA")
async def test_vllm_basic_task(context):
    """Test basic task with vLLM"""
    config = LLMConfig(
        model_name="facebook/opt-125m",
        model_type="vllm",
        max_new_tokens=512
    )
    
    llm = LLMFactory.create_llm(config)
    
    agent = Agent(
        task="Go to example.com and extract the title",
        llm=llm,
        browser_context=context,
    )
    
    history = await agent.run(max_steps=3)
    action_names = history.action_names()
    
    assert 'go_to_url' in action_names
    assert 'extract_content' in action_names 