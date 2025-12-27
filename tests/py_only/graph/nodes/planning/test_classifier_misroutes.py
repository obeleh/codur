"""Edge-case prompts for quick classifier (pattern-based Phase 0).

Passing cases guard recent fixes and capture prior misroutes.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage

from codur.graph.nodes.planning.classifier import quick_classify
from codur.graph.nodes.planning.types import TaskType


@pytest.mark.parametrize(
    ("prompt", "expected_type", "min_confidence"),
    [
        pytest.param(
            "hey fix main.py",
            TaskType.CODE_FIX,
            None,
            id="greeting-overrides-fix",
        ),
        pytest.param(
            "remove debug prints from app.py",
            TaskType.CODE_FIX,
            None,
            id="remove-means-delete-file",
        ),
        pytest.param(
            "What does app.py do? Please fix the bug.",
            TaskType.CODE_FIX,
            None,
            id="explain-overrides-fix",
        ),
        pytest.param(
            "Implement the report generator in app.py",
            TaskType.CODE_GENERATION,
            None,
            id="implement-treated-as-fix",
        ),
        pytest.param(
            "fix the bug in @main.py. The subtotal and discounted total are wrong.",
            TaskType.CODE_FIX,
            None,
            id="fix-with-at-file",
        ),
        pytest.param(
            "Implement the title casing rules in @main.py based on the docstring.",
            TaskType.CODE_GENERATION,
            None,
            id="implement-with-at-file",
        ),
        pytest.param(
            "Create a summary of today's weather",
            TaskType.WEB_SEARCH,
            None,
            id="generation-overrides-web-search",
        ),
        pytest.param(
            "Why is my test failing?",
            TaskType.CODE_FIX,
            None,
            id="broad-question-misroutes",
        ),
        pytest.param(
            "Delete the function foo in app.py",
            TaskType.CODE_FIX,
            None,
            id="delete-function-not-file",
        ),
        pytest.param(
            "Move the logic from app.py into utils.py",
            TaskType.COMPLEX_REFACTOR,
            None,
            id="move-logic-refactor",
        ),
        pytest.param(
            "Copy the validation rules from a.py to b.py",
            TaskType.COMPLEX_REFACTOR,
            None,
            id="copy-logic-refactor",
        ),
        pytest.param(
            "Build a CLI to fetch weather data",
            TaskType.CODE_GENERATION,
            None,
            id="weather-cli-generation",
        ),
        pytest.param(
            "Write a script to check stock price daily",
            TaskType.CODE_GENERATION,
            None,
            id="stock-script-generation",
        ),
        pytest.param(
            "Explain the weather script",
            TaskType.EXPLANATION,
            None,
            id="explain-weather-script",
        ),
        pytest.param(
            "Where is foo used?",
            TaskType.EXPLANATION,
            None,
            id="where-used-explanation",
        ),
        pytest.param(
            "When does the scheduler run in app.py?",
            TaskType.EXPLANATION,
            None,
            id="when-does-run-explanation",
        ),
        pytest.param(
            "Find usage of foo in the codebase",
            TaskType.EXPLANATION,
            None,
            id="find-usage-explanation",
        ),
        pytest.param(
            "Copy the behavior of foo into bar in app.py",
            TaskType.CODE_FIX,
            None,
            id="copy-behavior-edit",
        ),
        pytest.param(
            "Delete the logs in app.py",
            TaskType.CODE_FIX,
            None,
            id="delete-logs-edit",
        ),
        pytest.param(
            "Fix the report generation for today's weather",
            TaskType.CODE_FIX,
            None,
            id="fix-weather-report",
        ),
        pytest.param(
            "Create a dashboard for stock prices",
            TaskType.CODE_GENERATION,
            None,
            id="dashboard-stock-generation",
        ),
        pytest.param(
            "Build a UI for live stock prices",
            TaskType.CODE_GENERATION,
            None,
            id="ui-stock-generation",
        ),
        pytest.param(
            "Where is foo used in the project?",
            TaskType.EXPLANATION,
            None,
            id="where-used-project",
        ),
        pytest.param(
            "Summarize the codebase",
            TaskType.EXPLANATION,
            None,
            id="summarize-codebase",
        ),
        pytest.param(
            "Show me how to use foo",
            TaskType.EXPLANATION,
            None,
            id="how-to-use",
        ),
        pytest.param(
            "Show me how to use foo in app.py",
            TaskType.EXPLANATION,
            0.7,
            id="how-to-use-file",
        ),
        pytest.param(
            "Generate a report on current market trends",
            TaskType.WEB_SEARCH,
            None,
            id="report-market-trends",
        ),
        pytest.param(
            "Write code to fetch the current price of Bitcoin",
            TaskType.CODE_GENERATION,
            0.7,
            id="code-fetch-price",
        ),
        pytest.param(
            "List all files",
            TaskType.FILE_OPERATION,
            0.8,
            id="list-files",
        ),
        pytest.param(
            "Rename README.md to README_old.md",
            TaskType.FILE_OPERATION,
            None,
            id="rename-readme",
        ),
        pytest.param(
            """Implement the `format_table` function in `main.py` to create a working Markdown table formatter.

Your implementation must:
1. Parse the raw input string (rows separated by newlines, cells by pipes).
2. Determine alignment from the separator row (2nd row):
   - `|:---|` : Left align (default)
   - `|---:|` : Right align
   - `|:---:|`: Center align
3. Calculate the required width for each column based on the longest cell content in that column (header or body).
4. Reformat the table with correct padding (1 space on each side of content) and alignment.
5. Ensure the separator row uses dashes to match the column width and preserves the colons for alignment.

Run `python main.py` to test your solution.""",
            TaskType.CODE_GENERATION,
            0.6,
            id="implement-markdown-table-formatter",
        ),
        pytest.param(
            "Explain main.py",
            TaskType.EXPLANATION,
            0.7,
            id="rename-explain-main.py",
        ),
    ],
)
def test_quick_classify_misroutes(prompt: str, expected_type: TaskType, min_confidence: float) -> None:
    config = MagicMock()
    result = quick_classify([HumanMessage(content=prompt)], config)
    assert result.task_type == expected_type
    if min_confidence:
        assert result.confidence >= min_confidence
