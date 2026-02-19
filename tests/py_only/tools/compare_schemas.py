#!/usr/bin/env python3
"""Side-by-side comparison of schema generation approaches.

Run this to see actual schema differences visually.
"""

import json
from typing import Annotated, Optional
from enum import Enum
from langchain_core.utils.function_calling import convert_to_openai_function
from codur.tools.schema_generator import function_to_json_schema


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def compare_schemas(func, name: str):
    """Compare schemas from both generators."""
    print(f"\n{'='*80}")
    print(f"Function: {name}")
    print(f"{'='*80}")

    # LangChain schema
    try:
        lc_schema = convert_to_openai_function(func)
        print("\n🔵 LangChain (convert_to_openai_function):")
        print(json.dumps(lc_schema, indent=2))
    except Exception as e:
        print(f"\n🔵 LangChain: ERROR - {e}")

    # Custom schema
    try:
        custom_schema = function_to_json_schema(func)
        print("\n🟢 Custom (function_to_json_schema):")
        print(json.dumps(custom_schema, indent=2))
    except Exception as e:
        print(f"\n🟢 Custom: ERROR - {e}")

    print()


# Test 1: Annotated types
def test_annotated(
    path: Annotated[str, "The file path to process"],
    encoding: Annotated[str, "File encoding like 'utf-8'"] = "utf-8"
) -> str:
    """Process a file."""
    return path


# Test 2: Enum parameter
def test_enum(
    name: str,
    priority: Priority = Priority.MEDIUM
) -> dict:
    """Create task.

    Args:
        name: Task name
        priority: Priority level
    """
    return {}


# Test 3: Optional and complex types
def test_optional(
    required: str,
    optional: Optional[str] = None,
    numbers: list[int] = None
) -> str:
    """Process data.

    Args:
        required: Required field
        optional: Optional field
        numbers: List of numbers
    """
    return required


# Test 4: *args and **kwargs (bug demonstration)
def test_args_kwargs(
    required: str,
    *args,
    **kwargs
) -> str:
    """Flexible function.

    Args:
        required: Required parameter
    """
    return required


# Test 5: Docstring with both sources
def test_dual_description(
    source: Annotated[str, "From Annotated"],
) -> str:
    """Merge files.

    Args:
        source: From docstring
    """
    return source


if __name__ == "__main__":
    print("SCHEMA GENERATION COMPARISON")
    print("="*80)
    print("This shows differences between LangChain and Codur's schema generators")
    print()

    compare_schemas(test_annotated, "test_annotated")
    compare_schemas(test_enum, "test_enum")
    compare_schemas(test_optional, "test_optional")
    compare_schemas(test_args_kwargs, "test_args_kwargs")
    compare_schemas(test_dual_description, "test_dual_description")

    print("\n" + "="*80)
    print("SUMMARY OF KEY DIFFERENCES:")
    print("="*80)
    print("""
1. Annotated Descriptions:
   - LangChain: Can extract from Annotated[type, "desc"]
   - Custom: Uses docstring Args section or fallback

2. Enum Handling:
   - LangChain: Creates enum schema with values
   - Custom: Just uses "string" type

3. Optional Format:
   - LangChain: Uses anyOf schema
   - Custom: Uses ["type", "null"] array

4. *args/**kwargs:
   - LangChain: Correctly ignores them
   - Custom: BUG - includes them in schema

5. Description Precedence (when both exist):
   - Both: Prefer docstring over Annotated
    """)
