import asyncio
import subprocess

import pytest

from codur.agents.cli_agent_base import BaseCLIAgent
from codur.config import CodurConfig, LLMSettings


class DummyCLIAgent(BaseCLIAgent):
    def __init__(self) -> None:
        config = CodurConfig(llm=LLMSettings(default_profile="test"))
        super().__init__(config)
        self.command = "dummy"

    def _build_command(self, task: str, **kwargs: str) -> list[str]:
        return [self.command, task]

    def execute(self, task: str, **kwargs: str) -> str:
        return self._execute_cli(self._build_command(task), **kwargs)

    async def aexecute(self, task: str, **kwargs: str) -> str:
        return await self._aexecute_cli(self._build_command(task), **kwargs)

    def _cli_name(self) -> str:
        return "Dummy"

    def __repr__(self) -> str:
        return "DummyCLIAgent()"

    @property
    def name(self) -> str:
        return "dummy"

    @classmethod
    def get_description(cls) -> str:
        return "A dummy agent for testing."


def test_execute_cli_success(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = DummyCLIAgent()

    class Result:
        returncode = 0
        stdout = " ok "
        stderr = ""

    def fake_run(*args, **kwargs):
        return Result()

    monkeypatch.setattr("codur.agents.cli_agent_base.subprocess.run", fake_run)
    assert agent._execute_cli(["dummy"]) == "ok"


def test_execute_cli_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = DummyCLIAgent()

    class Result:
        returncode = 2
        stdout = ""
        stderr = "bad"

    def fake_run(*args, **kwargs):
        return Result()

    monkeypatch.setattr("codur.agents.cli_agent_base.subprocess.run", fake_run)
    with pytest.raises(Exception, match="Dummy exited with code 2: bad"):
        agent._execute_cli(["dummy"], capture_stderr=True)


def test_execute_cli_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = DummyCLIAgent()

    def fake_run(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("codur.agents.cli_agent_base.subprocess.run", fake_run)
    with pytest.raises(Exception, match="Dummy not found"):
        agent._execute_cli(["dummy"])


def test_execute_cli_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = DummyCLIAgent()

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="dummy", timeout=1)

    monkeypatch.setattr("codur.agents.cli_agent_base.subprocess.run", fake_run)
    with pytest.raises(Exception, match="Dummy timed out"):
        agent._execute_cli(["dummy"], timeout=1)


def test_aexecute_cli_success(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = DummyCLIAgent()

    class DummyProc:
        returncode = 0

        async def communicate(self):
            return b" ok ", b""

    async def fake_create(*args, **kwargs):
        return DummyProc()

    monkeypatch.setattr("codur.agents.cli_agent_base.asyncio.create_subprocess_exec", fake_create)
    result = asyncio.run(agent._aexecute_cli(["dummy"]))
    assert result == "ok"


def test_aexecute_cli_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = DummyCLIAgent()

    class DummyProc:
        returncode = 2

        async def communicate(self):
            return b"", b"bad"

    async def fake_create(*args, **kwargs):
        return DummyProc()

    monkeypatch.setattr("codur.agents.cli_agent_base.asyncio.create_subprocess_exec", fake_create)
    with pytest.raises(Exception, match="Dummy exited with code 2: bad"):
        asyncio.run(agent._aexecute_cli(["dummy"], capture_stderr=True))
