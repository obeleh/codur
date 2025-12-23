"""Tests for tool analysis logic."""

import pytest
from langchain_core.messages import SystemMessage

from codur.graph.nodes.planning.tool_analysis import (
    tool_results_include_read_file,
    select_file_from_tool_results,
    extract_list_files,
    pick_preferred_python_file,
)

class TestToolAnalysis:
    def test_tool_results_include_read_file(self):
        messages = [
            SystemMessage(content="Tool results:\nread_file: content of file")
        ]
        assert tool_results_include_read_file(messages) is True

        messages = [
            SystemMessage(content="Tool results:\nlist_files: ['a.py']")
        ]
        assert tool_results_include_read_file(messages) is False

    def test_extract_list_files(self):
        messages = [
            SystemMessage(content="Tool results:\nlist_files: ['main.py', 'test.py']")
        ]
        files = extract_list_files(messages)
        assert files == ['main.py', 'test.py']

        messages = [
            SystemMessage(content="Tool results:\nsome other output")
        ]
        files = extract_list_files(messages)
        assert files == []

    def test_pick_preferred_python_file(self):
        # Case 1: Prefers main.py
        files = ['utils.py', 'main.py', 'test.py']
        assert pick_preferred_python_file(files) == 'main.py'

        # Case 2: Prefers app.py
        files = ['utils.py', 'app.py']
        assert pick_preferred_python_file(files) == 'app.py'

        # Case 3: Prefers shallowest path
        files = ['src/utils.py', 'root_script.py']
        assert pick_preferred_python_file(files) == 'root_script.py'

        # Case 4: No py files
        files = ['README.md', 'config.json']
        assert pick_preferred_python_file(files) is None

    def test_select_file_from_tool_results(self):
        messages = [
            SystemMessage(content="Tool results:\nlist_files: ['utils.py', 'main.py']")
        ]
        selected = select_file_from_tool_results(messages)
        assert selected == 'main.py'
