"""Edge-case prompts for quick classifier (pattern-based Phase 0).

Passing cases guard recent fixes and capture prior misroutes.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage

from codur.graph.nodes.planning.classifier import quick_classify
from codur.graph.nodes.planning.types import TaskType


@pytest.mark.parametrize(
    ("prompt", "expected_type"),
    [
        pytest.param(
            "hey fix main.py",
            TaskType.CODE_FIX,
            id="greeting-overrides-fix",
        ),
        pytest.param(
            "remove debug prints from app.py",
            TaskType.CODE_FIX,
            id="remove-means-delete-file",
        ),
        pytest.param(
            "What does app.py do? Please fix the bug.",
            TaskType.CODE_FIX,
            id="explain-overrides-fix",
        ),
        pytest.param(
            "Implement the report generator in app.py",
            TaskType.CODE_GENERATION,
            id="implement-treated-as-fix",
        ),
        pytest.param(
            "Create a summary of today's weather",
            TaskType.WEB_SEARCH,
            id="generation-overrides-web-search",
        ),
        pytest.param(
            "Why is my test failing?",
            TaskType.CODE_FIX,
            id="broad-question-misroutes",
        ),
        pytest.param(
            "Delete the function foo in app.py",
            TaskType.CODE_FIX,
            id="delete-function-not-file",
        ),
        pytest.param(
            "Move the logic from app.py into utils.py",
            TaskType.COMPLEX_REFACTOR,
            id="move-logic-refactor",
        ),
        pytest.param(
            "Copy the validation rules from a.py to b.py",
            TaskType.COMPLEX_REFACTOR,
            id="copy-logic-refactor",
        ),
        pytest.param(
            "Build a CLI to fetch weather data",
            TaskType.CODE_GENERATION,
            id="weather-cli-generation",
        ),
        pytest.param(
            "Write a script to check stock price daily",
            TaskType.CODE_GENERATION,
            id="stock-script-generation",
        ),
        pytest.param(
            "Explain the weather script",
            TaskType.EXPLANATION,
            id="explain-weather-script",
        ),
        pytest.param(
            "Where is foo used?",
            TaskType.EXPLANATION,
            id="where-used-explanation",
        ),
        pytest.param(
            "When does the scheduler run in app.py?",
            TaskType.EXPLANATION,
            id="when-does-run-explanation",
        ),
        pytest.param(
            "Find usage of foo in the codebase",
            TaskType.EXPLANATION,
            id="find-usage-explanation",
        ),
        pytest.param(
            "Copy the behavior of foo into bar in app.py",
            TaskType.CODE_FIX,
            id="copy-behavior-edit",
        ),
        pytest.param(
            "Delete the logs in app.py",
            TaskType.CODE_FIX,
            id="delete-logs-edit",
        ),
        pytest.param(
            "Fix the report generation for today's weather",
            TaskType.CODE_FIX,
            id="fix-weather-report",
        ),
        pytest.param(
            "Create a dashboard for stock prices",
            TaskType.CODE_GENERATION,
            id="dashboard-stock-generation",
        ),
        pytest.param(
            "Build a UI for live stock prices",
            TaskType.CODE_GENERATION,
            id="ui-stock-generation",
        ),
        pytest.param(
            "Where is foo used in the project?",
            TaskType.EXPLANATION,
            id="where-used-project",
        ),
        pytest.param(
            "Summarize the codebase",
            TaskType.EXPLANATION,
            id="summarize-codebase",
        ),
        pytest.param(
            "Show me how to use foo",
            TaskType.EXPLANATION,
            id="how-to-use",
        ),
        pytest.param(
            "Show me how to use foo in app.py",
            TaskType.EXPLANATION,
            id="how-to-use-file",
        ),
        pytest.param(
            "Generate a report on current market trends",
            TaskType.WEB_SEARCH,
            id="report-market-trends",
        ),
        pytest.param(
            "Write code to fetch the current price of Bitcoin",
            TaskType.CODE_GENERATION,
            id="code-fetch-price",
        ),
        pytest.param(
            "List all files",
            TaskType.FILE_OPERATION,
            id="list-files",
        ),
        pytest.param(
            "Rename README.md to README_old.md",
            TaskType.FILE_OPERATION,
            id="rename-readme",
        ),
    ],
)
def test_quick_classify_misroutes(prompt: str, expected_type: TaskType) -> None:
    config = MagicMock()
    result = quick_classify([HumanMessage(content=prompt)], config)
    assert result.task_type == expected_type
