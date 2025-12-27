"""Verification logic for fix tasks."""

import os
import subprocess
from pathlib import Path
from rich.console import Console

from codur.graph.state import AgentState
from codur.config import CodurConfig
from codur.graph.state_operations import is_verbose
from codur.utils.text_helpers import truncate_lines

console = Console()


def _verify_fix(state: AgentState, config: CodurConfig) -> dict:
    """Verify if a fix actually works by running tests.

    Implements streaming verification with early exit on output mismatch.
    This prevents wasting time on obviously failed implementations.

    Args:
        state: Current agent state
        config: Codur configuration

    Returns:
        Dict with "success" (bool), "message" (str), and other diagnostic info
    """
    verbose = is_verbose(state)

    # Look for main.py or app.py in current directory
    cwd = Path.cwd()
    main_py = cwd / "main.py"
    app_py = cwd / "app.py"

    # Prefer main.py if it exists, otherwise use app.py
    entry_point = None
    if main_py.exists():
        entry_point = "main.py"
    elif app_py.exists():
        entry_point = "app.py"
    else:
        if verbose:
            console.print("[dim]No main.py or app.py found - skipping verification[/dim]")
        return {"success": True, "message": "No verification file found"}

    if verbose:
        console.print(f"[dim]Running verification: python {entry_point}[/dim]")

    # Check for fail-early mode to use shorter timeout
    fail_early = os.getenv("EARLY_FAILURE_HELPERS_FOR_TESTS") == "1"
    execution_timeout = 10 if fail_early else 60

    # Check for expected.txt to enable streaming verification
    expected_file = cwd / "expected.txt"
    use_streaming = expected_file.exists()

    if use_streaming:
        expected_output = expected_file.read_text().strip()
        expected_lines = expected_output.split('\n')

        try:
            # Run with streaming for early exit on mismatch
            process = subprocess.Popen(
                ["python", entry_point],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd)
            )

            output_lines = []
            mismatch_at_line = None

            try:
                # Stream output line by line and compare
                for line in process.stdout:
                    output_lines.append(line.rstrip('\n'))
                    line_idx = len(output_lines) - 1

                    # Check for mismatch (early exit)
                    if line_idx < len(expected_lines):
                        if line.rstrip('\n') != expected_lines[line_idx]:
                            mismatch_at_line = line_idx
                            process.terminate()
                            break
                    else:
                        # Output has more lines than expected
                        mismatch_at_line = line_idx
                        process.terminate()
                        break

                # Wait for process to finish
                try:
                    process.wait(timeout=execution_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

                stderr = process.stderr.read() if process.stderr else ""

            except Exception as e:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                return {"success": False, "message": f"Verification interrupted: {str(e)}"}

            # Check results
            actual_output = '\n'.join(output_lines).strip()

            if mismatch_at_line is None and actual_output == expected_output:
                return {
                    "success": True,
                    "message": f"Output matches expected: {actual_output}"
                }
            else:
                # Build detailed mismatch info
                return {
                    "success": False,
                    "message": f"Output mismatch.\nActual: {actual_output}\nExpected: {expected_output}",
                    "expected_output": expected_output,
                    "actual_output": actual_output,
                    "expected_truncated": truncate_lines(expected_output),
                    "actual_truncated": truncate_lines(actual_output),
                    "mismatch_at_line": mismatch_at_line,
                    "stderr": stderr.strip() if stderr else None
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "message": f"Execution timed out after {execution_timeout} seconds"}
        except Exception as e:
            return {"success": False, "message": f"Verification error: {str(e)}"}
    else:
        # No expected.txt - use standard verification
        try:
            result = subprocess.run(
                ["python", entry_point],
                capture_output=True,
                text=True,
                timeout=execution_timeout,
                cwd=str(cwd)
            )

            # Just check for success exit code
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Execution successful. Output: {result.stdout.strip()}"
                }
            else:
                std_out_stripped = result.stdout.strip()
                std_err_stripped = result.stderr.strip()
                if verbose:
                    # print std out and stderr for debugging
                    console.print(f"[dim]Stdout: {std_out_stripped}[/dim]")
                    console.print(f"[dim]Stderr: {std_err_stripped}[/dim]")
                return {
                    "success": False,
                    "message": f"Execution failed (exit code {result.returncode})\nStderr: {std_err_stripped}",
                    "stderr": std_err_stripped,
                    "stdout": std_out_stripped,
                    "return_code": result.returncode
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "message": f"Execution timed out after {execution_timeout} seconds"}
        except Exception as e:
            return {"success": False, "message": f"Verification error: {str(e)}"}
