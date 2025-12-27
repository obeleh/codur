"""Tests for Python tool injector."""

import pytest

from codur.graph.planning.injectors.python import PythonToolInjector


class TestPythonToolInjector:
    """Test the Python tool injector."""

    @pytest.fixture
    def injector(self):
        """Create a Python tool injector."""
        return PythonToolInjector()

    def test_extensions(self, injector):
        """Test that Python injector handles .py and .pyi files."""
        assert ".py" in injector.extensions
        assert ".pyi" in injector.extensions
        assert len(injector.extensions) == 2

    def test_name(self, injector):
        """Test the injector name."""
        assert injector.name == "Python"

    def test_get_followup_tools_single_file(self, injector):
        """Test followup tools for a single Python file."""
        paths = ["main.py"]
        tools = injector.get_followup_tools(paths)

        assert len(tools) == 1
        assert tools[0]["tool"] == "python_ast_dependencies"
        assert tools[0]["args"] == {"path": "main.py"}

    def test_get_followup_tools_multiple_files(self, injector):
        """Test followup tools for multiple Python files."""
        paths = ["main.py", "utils.py", "config.py"]
        tools = injector.get_followup_tools(paths)

        assert len(tools) == 1
        assert tools[0]["tool"] == "python_ast_dependencies_multifile"
        assert tools[0]["args"] == {"paths": ["main.py", "utils.py", "config.py"]}

    def test_get_followup_tools_pyi_file(self, injector):
        """Test followup tools for a .pyi stub file."""
        paths = ["types.pyi"]
        tools = injector.get_followup_tools(paths)

        assert len(tools) == 1
        assert tools[0]["tool"] == "python_ast_dependencies"
        assert tools[0]["args"] == {"path": "types.pyi"}

    def test_get_planning_tools(self, injector):
        """Test the list of planning tools."""
        tools = injector.get_planning_tools()

        assert "python_ast_outline" in tools
        assert "python_ast_graph" in tools
        assert "python_dependency_graph" in tools
        assert "python_ast_dependencies" in tools
        assert "python_ast_dependencies_multifile" in tools

    def test_get_example_tool_calls(self, injector):
        """Test example tool calls for planning prompts."""
        example_path = "example.py"
        tools = injector.get_example_tool_calls(example_path)

        assert len(tools) == 2
        assert tools[0]["tool"] == "read_file"
        assert tools[0]["args"] == {"path": "example.py"}
        assert tools[1]["tool"] == "python_ast_dependencies"
        assert tools[1]["args"] == {"path": "example.py"}

    def test_get_example_tool_calls_with_path(self, injector):
        """Test example tool calls with a different path."""
        example_path = "src/module.py"
        tools = injector.get_example_tool_calls(example_path)

        assert tools[0]["args"]["path"] == "src/module.py"
        assert tools[1]["args"]["path"] == "src/module.py"
