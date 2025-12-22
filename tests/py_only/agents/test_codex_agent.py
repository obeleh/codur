import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from codur.config import CodurConfig
from codur.agents.codex_agent import CodexAgent

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.agents.configs.get.return_value = {}
    config.agent_execution.default_cli_timeout = 300
    return config

def test_init(mock_config):
    agent = CodexAgent(mock_config)
    assert agent.command == "codex"
    assert agent.model == "gpt-5-codex"
    assert agent.reasoning_effort == "medium"

def test_build_command(mock_config):
    agent = CodexAgent(mock_config)
    task = "Write a function"
    cmd = agent._build_command(task)
    expected_cmd = [
        "codex", "exec", "--skip-git-repo-check",
        "-m", "gpt-5-codex",
        "--config", "model_reasoning_effort=medium",
        "--sandbox", "workspace-write",
        "--full-auto",
        task
    ]
    assert cmd == expected_cmd

def test_execute(mock_config):
    agent = CodexAgent(mock_config)
    # Mock _execute_cli to verify it's called with correct args
    with patch.object(agent, '_execute_cli', return_value="Executed") as mock_exec:
        result = agent.execute("Task", sandbox="read-only", timeout=60)
        assert result == "Executed"
        
        expected_cmd = agent._build_command("Task", sandbox="read-only")
        mock_exec.assert_called_with(expected_cmd, timeout=60, suppress_stderr=True)

def test_aexecute(mock_config):
    agent = CodexAgent(mock_config)
    # Mock _aexecute_cli
    with patch.object(agent, '_aexecute_cli', new_callable=AsyncMock) as mock_aexec:
        mock_aexec.return_value = "Async Executed"
        
        async def run_test():
            result = await agent.aexecute("Async Task")
            assert result == "Async Executed"
            
            expected_cmd = agent._build_command("Async Task")
            mock_aexec.assert_called_with(expected_cmd, timeout=None, suppress_stderr=True)
            
        asyncio.run(run_test())

def test_astream_raises(mock_config):
    agent = CodexAgent(mock_config)
    
    async def run_test():
        with pytest.raises(NotImplementedError):
            await agent.astream("Task")
            
    asyncio.run(run_test())

def test_resume_last(mock_config):
    agent = CodexAgent(mock_config)
    with patch("codur.agents.codex_agent.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Resumed output"
        
        result = agent.resume_last(additional_prompt="More info")
        assert result == "Resumed output"
        
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["codex", "exec", "--skip-git-repo-check", "resume", "--last"]
        assert kwargs["input"] == "More info"

def test_resume_last_failure(mock_config):
    agent = CodexAgent(mock_config)
    with patch("codur.agents.codex_agent.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        
        with pytest.raises(Exception, match="Codex resume failed with code 1"):
            agent.resume_last()
