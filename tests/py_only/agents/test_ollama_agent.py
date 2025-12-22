import asyncio
import pytest
from unittest.mock import MagicMock, patch

from codur.config import CodurConfig
from codur.agents.ollama_agent import OllamaAgent

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.agents.configs.get.return_value = {}
    config.mcp_servers.get.return_value = {}
    config.providers.get.return_value = None
    config.llm_temperature = 0.7
    return config

@pytest.fixture
def mock_ollama_client():
    with patch("codur.agents.ollama_agent.OllamaClient") as MockClient:
        client_instance = MockClient.return_value
        client_instance.generate.return_value = "Mocked response"
        client_instance.stream_generate.return_value = iter(["Mocked", " ", "stream"])
        client_instance.chat.return_value = "Mocked chat response"
        client_instance.list_models.return_value = [{"name": "model1"}, {"name": "model2"}]
        yield client_instance

def test_init(mock_config, mock_ollama_client):
    agent = OllamaAgent(mock_config)
    assert agent.model == "ministral-3:14b"  # Default
    assert agent.base_url == "http://localhost:11434" # Default

def test_execute(mock_config, mock_ollama_client):
    agent = OllamaAgent(mock_config)
    response = agent.execute("Test task")
    assert response == "Mocked response"
    mock_ollama_client.generate.assert_called_with("Test task", stream=False)

def test_aexecute(mock_config, mock_ollama_client):
    agent = OllamaAgent(mock_config)
    
    async def run_test():
        response = await agent.aexecute("Async task")
        assert response == "Mocked response"
        # Note: aexecute calls generate in executor, so we check generate call
        mock_ollama_client.generate.assert_called_with("Async task", stream=False)
        
    asyncio.run(run_test())

def test_astream(mock_config, mock_ollama_client):
    agent = OllamaAgent(mock_config)
    
    async def run_test():
        chunks = []
        async for chunk in agent.astream("Stream task"):
            chunks.append(chunk)
        assert chunks == ["Mocked", " ", "stream"]
        mock_ollama_client.stream_generate.assert_called_with("Stream task")
        
    asyncio.run(run_test())

def test_chat(mock_config, mock_ollama_client):
    agent = OllamaAgent(mock_config)
    messages = [{"role": "user", "content": "Hello"}]
    response = agent.chat(messages)
    assert response == "Mocked chat response"
    mock_ollama_client.chat.assert_called_with(messages, stream=False)

def test_switch_model(mock_config, mock_ollama_client):
    agent = OllamaAgent(mock_config)
    agent.switch_model("new-model")
    assert agent.model == "new-model"
    mock_ollama_client.switch_model.assert_called_with("new-model")
