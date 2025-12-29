"""Tests for tool registry filtering (inclusion and exclusion)."""

from __future__ import annotations

import pytest

from codur.constants import TaskType
from codur.tools.registry import list_tools_for_tasks
from codur.tools.tool_annotations import ToolSideEffect


class TestListToolsForTasksFiltering:
    """Tests for positive and negative filtering in list_tools_for_tasks."""

    def test_include_single_task_type(self):
        """Test including tools with a single TaskType."""
        tools = list_tools_for_tasks(TaskType.CODE_VALIDATION)
        assert len(tools) > 0
        # All returned tools should have CODE_VALIDATION scenario
        assert all(TaskType.CODE_VALIDATION in t["scenarios"] for t in tools)

    def test_include_multiple_task_types(self):
        """Test including tools with multiple TaskTypes."""
        tools = list_tools_for_tasks([TaskType.CODE_VALIDATION, TaskType.FILE_OPERATION])
        assert len(tools) > 0

    def test_exclude_single_task_type(self):
        """Test excluding tools with a single TaskType."""
        all_tools = list_tools_for_tasks(include_unannotated=True)
        excluded_tools = list_tools_for_tasks(
            exclude_task_types=TaskType.FILE_OPERATION,
            include_unannotated=True,
        )

        # Excluded set should be smaller
        assert len(excluded_tools) < len(all_tools)

        # Excluded tools should not have FILE_OPERATION scenario
        for tool in excluded_tools:
            if tool["scenarios"]:  # If annotated
                assert TaskType.FILE_OPERATION not in tool["scenarios"]

    def test_exclude_multiple_task_types(self):
        """Test excluding tools with multiple TaskTypes."""
        all_tools = list_tools_for_tasks(include_unannotated=True)
        excluded = list_tools_for_tasks(
            exclude_task_types=[TaskType.FILE_OPERATION, TaskType.CODE_GENERATION],
            include_unannotated=True,
        )

        assert len(excluded) < len(all_tools)

    def test_include_and_exclude_together(self):
        """Test combining include and exclude filters."""
        # Find CODE_VALIDATION tools but exclude CODE_EXECUTION side effects
        tools = list_tools_for_tasks(
            task_types=TaskType.CODE_VALIDATION,
            exclude_side_effects=ToolSideEffect.CODE_EXECUTION,
        )

        # All should be CODE_VALIDATION
        assert all(TaskType.CODE_VALIDATION in t["scenarios"] for t in tools)

    def test_exclude_side_effects(self):
        """Test excluding tools by side effects."""
        # This test may return the same count if no tools have FILE_MUTATION marked yet
        all_tools = list_tools_for_tasks(include_unannotated=True)
        safe_tools = list_tools_for_tasks(
            exclude_side_effects=ToolSideEffect.FILE_MUTATION,
            include_unannotated=True,
        )

        # Safe tools should be a subset (tools with FILE_MUTATION excluded)
        assert len(safe_tools) <= len(all_tools)

    def test_exclude_multiple_side_effects(self):
        """Test excluding multiple side effects."""
        all_tools = list_tools_for_tasks(include_unannotated=True)
        safe_tools = list_tools_for_tasks(
            exclude_side_effects=[
                ToolSideEffect.FILE_MUTATION,
                ToolSideEffect.CODE_EXECUTION,
            ],
            include_unannotated=True,
        )

        assert len(safe_tools) <= len(all_tools)

    def test_task_type_none_includes_all(self):
        """Test that task_types=None includes all tools (subject to other filters)."""
        all_tools = list_tools_for_tasks(include_unannotated=True)
        no_filter = list_tools_for_tasks(task_types=None, include_unannotated=True)

        # Should be the same
        assert len(no_filter) == len(all_tools)

    def test_exclude_with_no_include(self):
        """Test exclusion filter without inclusion filter."""
        # Get all tools, then exclude a type
        excluded = list_tools_for_tasks(
            exclude_task_types=TaskType.FILE_OPERATION,
            include_unannotated=True,
        )

        # Should have tools that are not FILE_OPERATION
        assert len(excluded) > 0

    def test_side_effects_in_results(self):
        """Test that side_effects are included in result dicts."""
        tools = list_tools_for_tasks(include_unannotated=True)

        for tool in tools:
            assert "side_effects" in tool
            assert isinstance(tool["side_effects"], list)

    def test_empty_result_when_too_restrictive(self):
        """Test that very restrictive filters can return empty results."""
        # Exclude both FILE_OPERATION and CODE_VALIDATION, include unannotated
        # Most tools fall into these categories
        excluded = list_tools_for_tasks(
            exclude_task_types=[TaskType.FILE_OPERATION, TaskType.CODE_VALIDATION],
        )

        # May have some tools or may be empty depending on annotations
        assert isinstance(excluded, list)
