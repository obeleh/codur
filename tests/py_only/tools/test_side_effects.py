"""Tests for tool side effects annotation."""

from __future__ import annotations

import pytest

from codur.tools.tool_annotations import (
    ToolSideEffect,
    tool_side_effects,
    get_tool_side_effects,
)


class TestToolSideEffects:
    """Tests for ToolSideEffect enum and decorator."""

    def test_side_effect_enum_values(self):
        """Test that all side effect categories are defined."""
        assert ToolSideEffect.FILE_MUTATION.value == "file_mutation"
        assert ToolSideEffect.CODE_EXECUTION.value == "code_execution"
        assert ToolSideEffect.STATE_CHANGE.value == "state_change"
        assert ToolSideEffect.NETWORK.value == "network"

    def test_single_side_effect_annotation(self):
        """Test decorating with a single side effect."""
        @tool_side_effects(ToolSideEffect.FILE_MUTATION)
        def write_file():
            pass

        effects = get_tool_side_effects(write_file)
        assert effects == [ToolSideEffect.FILE_MUTATION]

    def test_multiple_side_effects_annotation(self):
        """Test decorating with multiple side effects."""
        @tool_side_effects(ToolSideEffect.FILE_MUTATION, ToolSideEffect.CODE_EXECUTION)
        def replace_and_execute():
            pass

        effects = get_tool_side_effects(replace_and_execute)
        assert ToolSideEffect.FILE_MUTATION in effects
        assert ToolSideEffect.CODE_EXECUTION in effects
        assert len(effects) == 2

    def test_no_side_effects(self):
        """Test that tools without decorator have no side effects."""
        def read_file():
            pass

        effects = get_tool_side_effects(read_file)
        assert effects == []

    def test_multiple_decorator_calls_accumulate(self):
        """Test that multiple decorators accumulate side effects."""
        @tool_side_effects(ToolSideEffect.FILE_MUTATION)
        @tool_side_effects(ToolSideEffect.CODE_EXECUTION)
        def complex_tool():
            pass

        effects = get_tool_side_effects(complex_tool)
        assert ToolSideEffect.FILE_MUTATION in effects
        assert ToolSideEffect.CODE_EXECUTION in effects

    def test_side_effects_with_list(self):
        """Test decorator with list of side effects."""
        @tool_side_effects(*[ToolSideEffect.FILE_MUTATION, ToolSideEffect.STATE_CHANGE])
        def git_write():
            pass

        effects = get_tool_side_effects(git_write)
        assert ToolSideEffect.FILE_MUTATION in effects
        assert ToolSideEffect.STATE_CHANGE in effects

    def test_filter_safe_tools(self):
        """Test filtering to find safe (no side effects) tools."""
        @tool_side_effects(ToolSideEffect.FILE_MUTATION)
        def write_tool():
            pass

        def read_tool():
            pass

        def validate_tool():
            pass

        tools = [write_tool, read_tool, validate_tool]
        safe_tools = [t for t in tools if not get_tool_side_effects(t)]

        assert write_tool not in safe_tools
        assert read_tool in safe_tools
        assert validate_tool in safe_tools
        assert len(safe_tools) == 2

    def test_filter_unsafe_tools_by_type(self):
        """Test filtering tools by specific side effect type."""
        @tool_side_effects(ToolSideEffect.FILE_MUTATION)
        def file_write():
            pass

        @tool_side_effects(ToolSideEffect.CODE_EXECUTION)
        def code_run():
            pass

        @tool_side_effects(ToolSideEffect.STATE_CHANGE)
        def git_op():
            pass

        def read_file():
            pass

        tools = [file_write, code_run, git_op, read_file]

        # Find tools that execute code (but allow file mutations)
        exec_tools = [
            t
            for t in tools
            if ToolSideEffect.CODE_EXECUTION in get_tool_side_effects(t)
        ]
        assert exec_tools == [code_run]

        # Find tools with NO code execution
        no_exec_tools = [
            t
            for t in tools
            if ToolSideEffect.CODE_EXECUTION not in get_tool_side_effects(t)
        ]
        assert code_run not in no_exec_tools
        assert len(no_exec_tools) == 3

    def test_verification_agent_safe_tools(self):
        """Test filtering tools safe for verification agent (read-only)."""
        @tool_side_effects(ToolSideEffect.CODE_EXECUTION)
        def run_pytest():
            pass

        @tool_side_effects(ToolSideEffect.FILE_MUTATION)
        def replace_function():
            pass

        def read_file():
            pass

        def validate_syntax():
            pass

        all_tools = [run_pytest, replace_function, read_file, validate_syntax]

        # Verification agent: only read-only tools (no FILE_MUTATION)
        safe_for_verification = [
            t
            for t in all_tools
            if ToolSideEffect.FILE_MUTATION not in get_tool_side_effects(t)
        ]

        assert replace_function not in safe_for_verification
        assert read_file in safe_for_verification
        assert validate_syntax in safe_for_verification
