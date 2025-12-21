"""
Base class for CLI-driven agents.
"""

from __future__ import annotations

import asyncio
import subprocess
from abc import ABC, abstractmethod
from typing import Optional

from codur.agents.base import BaseAgent
from codur.constants import DEFAULT_CLI_TIMEOUT

class CLIExecutionError(Exception):
    """Raised for CLI command failures with a formatted message."""


class BaseCLIAgent(BaseAgent, ABC):
    """Common execution helpers for agents that wrap a CLI tool."""

    default_timeout = DEFAULT_CLI_TIMEOUT

    @abstractmethod
    def _build_command(self, *args: str, **kwargs: str) -> list[str]:
        """Build the CLI command for the agent."""

    def _cli_name(self) -> str:
        return self.__class__.__name__.replace("Agent", "")

    def _not_found_message(self) -> str:
        return (
            f"{self._cli_name()} not found. Install it or configure the command path. "
            f"Current command: {self.command}"
        )

    def _timeout_message(self, timeout: int) -> str:
        return f"{self._cli_name()} timed out after {timeout} seconds"

    def _exit_code_message(self, returncode: int, stderr_text: str | None) -> str:
        if stderr_text:
            return f"{self._cli_name()} exited with code {returncode}: {stderr_text}"
        return f"{self._cli_name()} exited with code {returncode}"

    def _execution_failed_message(self, error: Exception) -> str:
        return f"{self._cli_name()} agent failed: {error}"

    def _get_timeout(self, timeout: Optional[int]) -> int:
        return timeout or self.default_timeout

    def _execute_cli(
        self,
        cmd: list[str],
        *,
        timeout: Optional[int] = None,
        capture_stderr: bool = False,
        suppress_stderr: bool = False,
    ) -> str:
        try:
            stderr_setting = None
            if capture_stderr:
                stderr_setting = subprocess.PIPE
            elif suppress_stderr:
                stderr_setting = subprocess.DEVNULL

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=stderr_setting,
                text=True,
                timeout=self._get_timeout(timeout),
            )

            if result.returncode != 0:
                raise CLIExecutionError(
                    self._exit_code_message(result.returncode, result.stderr)
                )

            return result.stdout.strip()

        except subprocess.TimeoutExpired as exc:
            raise Exception(self._timeout_message(self._get_timeout(timeout))) from exc
        except FileNotFoundError as exc:
            raise Exception(self._not_found_message()) from exc
        except CLIExecutionError:
            raise
        except Exception as exc:
            raise Exception(self._execution_failed_message(exc)) from exc

    async def _aexecute_cli(
        self,
        cmd: list[str],
        *,
        timeout: Optional[int] = None,
        capture_stderr: bool = False,
        suppress_stderr: bool = False,
    ) -> str:
        try:
            stderr_setting = None
            if capture_stderr:
                stderr_setting = asyncio.subprocess.PIPE
            elif suppress_stderr:
                stderr_setting = asyncio.subprocess.DEVNULL

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr_setting,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._get_timeout(timeout),
                )
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise Exception(self._timeout_message(self._get_timeout(timeout))) from exc

            if proc.returncode != 0:
                stderr_text = stderr.decode() if capture_stderr and stderr else None
                raise CLIExecutionError(
                    self._exit_code_message(proc.returncode, stderr_text)
                )

            return stdout.decode().strip()

        except FileNotFoundError as exc:
            raise Exception(self._not_found_message()) from exc
        except CLIExecutionError:
            raise
        except Exception as exc:
            raise Exception(self._execution_failed_message(exc)) from exc
