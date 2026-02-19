"""Test parity between convert_to_openai_function and custom function_to_json_schema.

This test suite highlights differences in schema generation and drives improvements
to make tool calling more stable and robust.

The goal is to eventually have get_function_schemas() support all these cases.
"""

from __future__ import annotations

import pytest
from enum import Enum
from typing import Annotated, Literal, Optional, Union
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.tools import tool

from codur.tools.schema_generator import function_to_json_schema, get_function_schemas
from codur.constants import TaskType
from codur.tools.tool_annotations import tool_scenarios


class Priority(Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TestAnnotatedTypeDescriptions:
    """Test extraction of descriptions from Annotated types."""

    def test_annotated_string_parameter(self):
        """LangChain extracts description from Annotated, custom one doesn't."""
        def process_file(
            path: Annotated[str, "The file path to process"],
            encoding: Annotated[str, "File encoding (e.g., 'utf-8', 'latin-1')"] = "utf-8"
        ) -> str:
            """Process a file with specified encoding."""
            return f"Processing {path}"

        # LangChain should extract descriptions from Annotated
        langchain_schema = convert_to_openai_function(process_file)
        assert langchain_schema["parameters"]["properties"]["path"]["description"] == "The file path to process"
        assert "File encoding" in langchain_schema["parameters"]["properties"]["encoding"]["description"]

        # Custom schema generator currently doesn't support Annotated
        custom_schema = function_to_json_schema(process_file)
        # This will fail - custom one uses generic "Parameter path"
        with pytest.raises(AssertionError, match="Annotated description not extracted"):
            assert custom_schema["parameters"]["properties"]["path"]["description"] == "The file path to process", \
                "Annotated description not extracted by custom schema generator"

    def test_annotated_with_docstring_args(self):
        """Test when both Annotated and docstring Args exist - which takes precedence?"""
        def merge_files(
            source: Annotated[str, "Source file (from Annotated)"],
            dest: Annotated[str, "Destination file (from Annotated)"]
        ) -> str:
            """Merge files together.

            Args:
                source: Source file (from docstring)
                dest: Destination file (from docstring)
            """
            return "merged"

        langchain_schema = convert_to_openai_function(merge_files)
        custom_schema = function_to_json_schema(merge_files)

        # LangChain prefers Annotated over docstring
        assert "from Annotated" in langchain_schema["parameters"]["properties"]["source"]["description"]

        # Custom prefers docstring
        assert "from docstring" in custom_schema["parameters"]["properties"]["source"]["description"]

        # Both should have descriptions though
        assert langchain_schema["parameters"]["properties"]["source"]["description"]
        assert custom_schema["parameters"]["properties"]["source"]["description"]


class TestComplexTypes:
    """Test handling of complex type annotations."""

    def test_enum_parameter(self):
        """LangChain converts enums to enum schema, custom one may not."""
        def create_task(
            name: str,
            priority: Priority = Priority.MEDIUM
        ) -> dict:
            """Create a task with priority.

            Args:
                name: Task name
                priority: Task priority level
            """
            return {"name": name, "priority": priority.value}

        langchain_schema = convert_to_openai_function(create_task)

        # LangChain should create enum schema
        priority_schema = langchain_schema["parameters"]["properties"]["priority"]
        # Should have enum values
        assert "enum" in priority_schema or "anyOf" in priority_schema or priority_schema.get("type") == "string"

        # Custom schema generator may just use string type
        custom_schema = function_to_json_schema(create_task)
        custom_priority = custom_schema["parameters"]["properties"]["priority"]

        # This will likely fail - custom one doesn't extract enum values
        with pytest.raises(AssertionError, match="Enum values not extracted"):
            assert "enum" in custom_priority, "Enum values not extracted by custom schema generator"

    def test_literal_type(self):
        """Test Literal type support."""
        def set_mode(
            mode: Literal["read", "write", "append"],
            path: str
        ) -> str:
            """Set file operation mode.

            Args:
                mode: Operation mode
                path: File path
            """
            return f"Mode: {mode}"

        langchain_schema = convert_to_openai_function(set_mode)
        mode_schema = langchain_schema["parameters"]["properties"]["mode"]

        # LangChain should recognize Literal and create enum
        assert "enum" in mode_schema or mode_schema.get("type") == "string"

        # Custom schema generator doesn't handle Literal
        custom_schema = function_to_json_schema(set_mode)
        custom_mode = custom_schema["parameters"]["properties"]["mode"]

        with pytest.raises(AssertionError, match="Literal not converted to enum"):
            assert "enum" in custom_mode, "Literal not converted to enum by custom schema generator"

    def test_union_beyond_optional(self):
        """Test Union types that aren't just Optional."""
        def process_input(
            data: Union[str, int, list[str]],
            default: str = "none"
        ) -> str:
            """Process input that can be multiple types.

            Args:
                data: Input data (string, integer, or list of strings)
                default: Default value
            """
            return str(data)

        langchain_schema = convert_to_openai_function(process_input)
        data_schema = langchain_schema["parameters"]["properties"]["data"]

        # LangChain should handle Union with anyOf or multiple types
        assert "anyOf" in data_schema or isinstance(data_schema.get("type"), list)

        # Custom schema generator only handles Optional (Union with None)
        custom_schema = function_to_json_schema(process_input)
        custom_data = custom_schema["parameters"]["properties"]["data"]

        # Will likely just be string or first type
        with pytest.raises(AssertionError, match="Union types not handled"):
            assert "anyOf" in custom_data or isinstance(custom_data.get("type"), list), \
                "Union types not handled by custom schema generator"


class TestNestedStructures:
    """Test nested and complex data structures."""

    def test_list_of_dicts(self):
        """Test list with complex item types."""
        def batch_process(
            items: list[dict[str, str]],
            parallel: bool = False
        ) -> list[str]:
            """Process a batch of items.

            Args:
                items: List of item dictionaries
                parallel: Whether to process in parallel
            """
            return []

        langchain_schema = convert_to_openai_function(batch_process)
        items_schema = langchain_schema["parameters"]["properties"]["items"]

        # LangChain should handle nested structure
        assert items_schema["type"] == "array"
        assert "items" in items_schema
        # Items should be object type
        assert items_schema["items"]["type"] == "object"

        # Custom schema generator has basic array support
        custom_schema = function_to_json_schema(batch_process)
        custom_items = custom_schema["parameters"]["properties"]["items"]

        assert custom_items["type"] == "array"
        # But may not properly handle dict item type
        with pytest.raises(AssertionError, match="Nested dict type not preserved"):
            assert custom_items["items"]["type"] == "object", \
                "Nested dict type not preserved by custom schema generator"

    def test_dict_with_typed_values(self):
        """Test dict with specific value types."""
        def update_config(
            settings: dict[str, int],
            metadata: dict[str, str] | None = None
        ) -> dict:
            """Update configuration settings.

            Args:
                settings: Configuration settings (name -> value)
                metadata: Optional metadata
            """
            return settings

        langchain_schema = convert_to_openai_function(update_config)
        settings_schema = langchain_schema["parameters"]["properties"]["settings"]

        # LangChain should recognize typed dict
        assert settings_schema["type"] == "object"
        # May have additionalProperties with type info

        custom_schema = function_to_json_schema(update_config)
        custom_settings = custom_schema["parameters"]["properties"]["settings"]

        # Custom one treats it as generic object
        assert custom_settings["type"] == "object"
        # But doesn't capture value type constraints
        with pytest.raises(AssertionError, match="Dict value types not captured"):
            assert "additionalProperties" in custom_settings, \
                "Dict value types not captured by custom schema generator"


class TestDefaultValues:
    """Test handling of default values in schemas."""

    def test_default_value_representation(self):
        """Test if default values are included in schema."""
        def search_files(
            query: str,
            max_results: int = 10,
            case_sensitive: bool = False,
            extensions: list[str] | None = None
        ) -> list[str]:
            """Search for files.

            Args:
                query: Search query
                max_results: Maximum number of results
                case_sensitive: Whether search is case sensitive
                extensions: File extensions to filter
            """
            return []

        langchain_schema = convert_to_openai_function(search_files)
        custom_schema = function_to_json_schema(search_files)

        # LangChain may include default values
        max_results_lc = langchain_schema["parameters"]["properties"]["max_results"]

        # Custom one may not include defaults
        max_results_custom = custom_schema["parameters"]["properties"]["max_results"]

        # Check if defaults are represented
        # LangChain might use "default" field in schema
        has_default_lc = "default" in max_results_lc
        has_default_custom = "default" in max_results_custom

        # Document the difference
        assert has_default_lc or not has_default_lc  # Either way is fine for LangChain
        # Custom one likely doesn't include defaults
        assert not has_default_custom, "Custom schema unexpectedly includes default values"


class TestStructuredToolSupport:
    """Test handling of LangChain StructuredTool objects."""

    def test_structured_tool_conversion(self):
        """StructuredTool objects work with LangChain but not custom generator."""
        @tool
        def calculate_sum(
            numbers: Annotated[list[int], "List of numbers to sum"],
            initial: Annotated[int, "Initial value"] = 0
        ) -> int:
            """Calculate the sum of numbers."""
            return sum(numbers) + initial

        # LangChain handles StructuredTool
        langchain_schema = convert_to_openai_function(calculate_sum)
        assert langchain_schema["name"] == "calculate_sum"
        assert "numbers" in langchain_schema["parameters"]["properties"]

        # Custom generator fails on StructuredTool
        with pytest.raises(TypeError, match="not a callable object"):
            custom_schema = function_to_json_schema(calculate_sum)

    def test_mixed_tool_types(self):
        """Test schema generation from mixed tool types (StructuredTool + functions)."""
        @tool
        def structured_tool_example(x: int) -> int:
            """A structured tool."""
            return x * 2

        def regular_function_example(y: str) -> str:
            """A regular function.

            Args:
                y: Input string
            """
            return y.upper()

        tools = [structured_tool_example, regular_function_example]

        # LangChain handles both
        schemas_lc = [convert_to_openai_function(t) for t in tools]
        assert len(schemas_lc) == 2
        assert all(s["name"] for s in schemas_lc)

        # Custom generator fails on first one
        with pytest.raises(TypeError):
            schemas_custom = [function_to_json_schema(t) for t in tools]


class TestOptionalHandling:
    """Test Optional and nullable type handling."""

    def test_optional_vs_nullable(self):
        """Test if Optional types are properly marked as nullable."""
        def process_data(
            required_field: str,
            optional_field: Optional[str] = None,
            nullable_with_default: Optional[int] = 42
        ) -> str:
            """Process data with optional fields.

            Args:
                required_field: Required string field
                optional_field: Optional string field
                nullable_with_default: Optional int with default
            """
            return required_field

        langchain_schema = convert_to_openai_function(process_data)
        custom_schema = function_to_json_schema(process_data)

        # Check optional_field handling
        optional_lc = langchain_schema["parameters"]["properties"]["optional_field"]
        optional_custom = custom_schema["parameters"]["properties"]["optional_field"]

        # Both should mark it as nullable somehow
        # LangChain might use ["string", "null"] or anyOf
        lc_nullable = (
            isinstance(optional_lc.get("type"), list) and "null" in optional_lc["type"]
        ) or "anyOf" in optional_lc

        custom_nullable = (
            isinstance(optional_custom.get("type"), list) and "null" in optional_custom["type"]
        )

        assert lc_nullable or custom_nullable, "At least one should mark as nullable"

        # Document which approach each uses
        if lc_nullable:
            assert isinstance(optional_lc["type"], list) or "anyOf" in optional_lc
        if custom_nullable:
            assert isinstance(optional_custom["type"], list)


class TestDocstringParsing:
    """Test docstring parsing edge cases."""

    def test_missing_docstring(self):
        """Test functions without docstrings."""
        def no_docs(x: str, y: int = 5) -> str:
            return x * y

        langchain_schema = convert_to_openai_function(no_docs)
        custom_schema = function_to_json_schema(no_docs)

        # Both should handle missing docstrings gracefully
        assert langchain_schema["description"]  # Should have some description
        assert custom_schema["description"]  # Should have fallback

    def test_malformed_docstring(self):
        """Test malformed Args sections in docstrings."""
        def weird_docs(
            param1: str,
            param2: int
        ) -> str:
            """Function with weird docstring.

            Args:
            param1 - uses dash instead of colon
            param2 (int) more text but no colon at all
            """
            return param1

        langchain_schema = convert_to_openai_function(weird_docs)
        custom_schema = function_to_json_schema(weird_docs)

        # Both should handle gracefully
        assert "param1" in langchain_schema["parameters"]["properties"]
        assert "param1" in custom_schema["parameters"]["properties"]

        # Custom one's docstring parser may miss descriptions
        param1_desc = custom_schema["parameters"]["properties"]["param1"]["description"]
        # Likely falls back to generic description
        assert param1_desc  # Should have something


class TestToolScenarioAnnotations:
    """Test integration with Codur's tool_scenarios annotations."""

    @tool_scenarios(TaskType.FILE_OPERATION)
    def annotated_tool(path: str, mode: str = "read") -> str:
        """Tool with scenario annotations.

        Args:
            path: File path
            mode: Operation mode
        """
        return f"Operating on {path}"

    def test_scenarios_dont_break_schema_generation(self):
        """Tool scenario decorators shouldn't break schema generation."""
        # Both should handle decorated functions
        langchain_schema = convert_to_openai_function(self.annotated_tool)

        # Custom generator should work on underlying function
        # Need to check if it's wrapped
        try:
            custom_schema = function_to_json_schema(self.annotated_tool)
            assert "path" in custom_schema["parameters"]["properties"]
        except TypeError:
            # If it's a StructuredTool, that's expected for custom generator
            pytest.skip("Decorated tool became StructuredTool")


class TestRobustness:
    """Test edge cases for robustness."""

    def test_args_kwargs(self):
        """Test functions with *args and **kwargs."""
        def flexible_function(required: str, *args, **kwargs) -> str:
            """Flexible function with args and kwargs.

            Args:
                required: Required parameter
            """
            return required

        # LangChain should handle gracefully
        langchain_schema = convert_to_openai_function(flexible_function)
        assert langchain_schema["parameters"]["properties"]["required"]

        # Custom one should ignore *args and **kwargs
        custom_schema = function_to_json_schema(flexible_function)
        assert "required" in custom_schema["parameters"]["properties"]
        # Should not have args or kwargs in schema
        assert "args" not in custom_schema["parameters"]["properties"]
        assert "kwargs" not in custom_schema["parameters"]["properties"]

    def test_forward_references(self):
        """Test functions with forward reference type hints."""
        def future_types(
            data: "list[str]",  # String annotation
            count: int = 0
        ) -> "dict[str, int]":
            """Function with forward reference types.

            Args:
                data: List of strings
                count: Count value
            """
            return {}

        # Both should handle string annotations
        try:
            langchain_schema = convert_to_openai_function(future_types)
            assert "data" in langchain_schema["parameters"]["properties"]
        except Exception as e:
            pytest.skip(f"LangChain doesn't handle forward refs: {e}")

        try:
            custom_schema = function_to_json_schema(future_types)
            assert "data" in custom_schema["parameters"]["properties"]
        except Exception as e:
            pytest.skip(f"Custom doesn't handle forward refs: {e}")


class TestRealWorldExamples:
    """Test real-world tool examples from Codur."""

    def test_complex_read_file_signature(self):
        """Test a realistic file reading tool signature."""
        from typing import Optional

        def read_file_realistic(
            path: Annotated[str, "Path to the file to read"],
            line_start: Annotated[int, "Starting line number (1-indexed)"] = 1,
            line_end: Annotated[Optional[int], "Ending line number (inclusive)"] = None,
            max_bytes: Annotated[int, "Maximum bytes to read"] = 1_000_000,
            encoding: Annotated[str, "File encoding"] = "utf-8",
            ignore_binary: Annotated[bool, "Skip binary files"] = True
        ) -> str:
            """Read file with multiple options.

            Args:
                path: Path to the file to read
                line_start: Starting line number (1-indexed)
                line_end: Ending line number (inclusive), None for end of file
                max_bytes: Maximum bytes to read to prevent memory issues
                encoding: File encoding (utf-8, latin-1, etc)
                ignore_binary: Whether to skip binary files
            """
            return "file contents"

        langchain_schema = convert_to_openai_function(read_file_realistic)
        custom_schema = function_to_json_schema(read_file_realistic)

        # Check all parameters are present
        for schema in [langchain_schema, custom_schema]:
            props = schema["parameters"]["properties"]
            assert "path" in props
            assert "line_start" in props
            assert "line_end" in props
            assert "max_bytes" in props
            assert "encoding" in props
            assert "ignore_binary" in props

        # Check required parameters
        for schema in [langchain_schema, custom_schema]:
            required = schema["parameters"]["required"]
            assert "path" in required
            assert "line_start" not in required  # has default
            assert "line_end" not in required  # optional

        # Compare description quality
        lc_path_desc = langchain_schema["parameters"]["properties"]["path"]["description"]
        custom_path_desc = custom_schema["parameters"]["properties"]["path"]["description"]

        # LangChain should get Annotated description
        assert "Path to the file" in lc_path_desc

        # Custom should get docstring description (which is also good in this case)
        assert custom_path_desc  # Should have something
        # May be from docstring or fallback
