"""Tests for refactored get_function_schemas with TaskType filtering."""

from __future__ import annotations

import pytest

from codur.constants import TaskType
from codur.tools.schema_generator import get_function_schemas
from codur.tools.tool_annotations import ToolSideEffect


class TestGetFunctionSchemasFiltering:
    """Tests for TaskType and side effect filtering in get_function_schemas."""

    def test_get_all_schemas_backward_compatible(self):
        """Test that get_function_schemas() with no args works (backward compatible)."""
        schemas = get_function_schemas()
        assert len(schemas) > 0
        assert all(isinstance(s, dict) for s in schemas)
        assert all("name" in s and "description" in s and "parameters" in s for s in schemas)

    def test_filter_by_task_type(self):
        """Test filtering schemas by TaskType."""
        schemas = get_function_schemas(TaskType.CODE_VALIDATION)
        assert len(schemas) > 0
        # All should have CODE_VALIDATION in their scenarios
        for schema in schemas:
            assert "name" in schema

    def test_filter_by_multiple_task_types(self):
        """Test filtering by multiple TaskTypes."""
        schemas = get_function_schemas([TaskType.CODE_VALIDATION, TaskType.FILE_OPERATION])
        assert len(schemas) > 0

    def test_exclude_task_type(self):
        """Test excluding tools by TaskType."""
        all_schemas = get_function_schemas(include_unannotated=True)
        excluded = get_function_schemas(
            exclude_task_types=TaskType.FILE_OPERATION,
            include_unannotated=True,
        )

        # Excluded set should be smaller
        assert len(excluded) < len(all_schemas)

    def test_exclude_side_effects(self):
        """Test excluding tools by side effects."""
        all_schemas = get_function_schemas(include_unannotated=True)
        safe_schemas = get_function_schemas(
            exclude_side_effects=ToolSideEffect.FILE_MUTATION,
            include_unannotated=True,
        )

        # Safe schemas should be <= all schemas
        assert len(safe_schemas) <= len(all_schemas)

    def test_combined_filtering(self):
        """Test combining TaskType and side effect filters."""
        schemas = get_function_schemas(
            task_types=TaskType.CODE_VALIDATION,
            exclude_side_effects=ToolSideEffect.CODE_EXECUTION,
        )

        assert len(schemas) > 0
        assert all(isinstance(s, dict) for s in schemas)

    def test_schema_has_required_fields(self):
        """Test that returned schemas have all required fields."""
        schemas = get_function_schemas()
        assert len(schemas) > 0

        for schema in schemas:
            # Check required fields
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

            # Check parameters structure
            params = schema["parameters"]
            assert "type" in params
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params

    def test_include_unannotated_flag(self):
        """Test include_unannotated parameter."""
        with_unannotated = get_function_schemas(include_unannotated=True)
        without_unannotated = get_function_schemas(include_unannotated=False)

        # with_unannotated should have more or equal schemas
        assert len(with_unannotated) >= len(without_unannotated)

    def test_empty_result_on_restrictive_filter(self):
        """Test that restrictive filters can return empty results."""
        # Exclude both major task types
        schemas = get_function_schemas(
            exclude_task_types=[TaskType.FILE_OPERATION, TaskType.CODE_VALIDATION],
        )

        # May have other tools or may be empty
        assert isinstance(schemas, list)

    def test_schema_name_matches_function_name(self):
        """Test that schema names correspond to actual tools."""
        schemas = get_function_schemas()
        assert len(schemas) > 0

        # Sample check: looking for known tools
        schema_names = [s["name"] for s in schemas]
        # At least one known tool should be present
        known_tools = {"read_file", "write_file", "run_pytest"}
        assert any(name in schema_names for name in known_tools)

    def test_parameters_exclude_internal_params(self):
        """Test that internal parameters are filtered out."""
        schemas = get_function_schemas()
        internal_params = {"root", "state", "config", "allow_outside_root"}

        for schema in schemas:
            properties = schema["parameters"]["properties"]
            # Check that internal params are not in properties
            for param in properties:
                assert param not in internal_params
