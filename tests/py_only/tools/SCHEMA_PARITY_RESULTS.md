# Schema Generation Parity Test Results

This document summarizes the differences between `convert_to_openai_function` (LangChain) and `function_to_json_schema` (Codur custom) based on the test suite in `test_schema_parity.py`.

## Test Results Summary

**Total Tests:** 17
- **Passed:** 11 (65%)
- **Failed:** 6 (35%)

## Key Findings

### ✅ What Custom Schema Generator Does Well

1. **Docstring Parsing** - Successfully extracts parameter descriptions from Args sections
2. **Internal Parameter Filtering** - Automatically filters out `state`, `config`, `root`, `allow_outside_root`
3. **Basic Type Support** - Handles str, int, bool, list, dict correctly
4. **Optional Handling** - Properly marks Optional types with `["type", "null"]`
5. **Required Fields** - Correctly identifies required vs optional parameters
6. **Nested Arrays** - Can handle `list[dict[str, str]]` properly (test proved this works!)
7. **Graceful Degradation** - Handles malformed docstrings without crashing

### ❌ Gaps That Need Improvement

#### 1. **Annotated Type Descriptions** ✓ (Test Passed - Expected Failure)
**Status:** Custom generator doesn't extract from `Annotated[type, "description"]`

```python
# LangChain extracts: "The file path to process"
# Custom falls back to: "Parameter path"
path: Annotated[str, "The file path to process"]
```

**Recommendation:** Add support for Annotated type introspection.

#### 2. **Enum Support** ✓ (Test Passed - Expected Failure)
**Status:** Custom generator doesn't create `enum` schemas from Enum types

```python
class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"

# LangChain: includes enum values
# Custom: just uses "string" type
```

**Recommendation:** Detect Enum types and extract values to `enum` field.

#### 3. **Literal Type Support** ✓ (Test Passed - Expected Failure)
**Status:** Literal types aren't converted to enums

```python
mode: Literal["read", "write", "append"]

# LangChain: creates enum with 3 values
# Custom: just "string" type
```

**Recommendation:** Extract Literal values to `enum` field.

#### 4. **Union Type Support** ✓ (Test Passed - Expected Failure)
**Status:** Only handles Optional, not general Union types

```python
data: Union[str, int, list[str]]

# LangChain: uses anyOf or multiple types
# Custom: picks first type (str)
```

**Recommendation:** Add `anyOf` support for Union types.

#### 5. **StructuredTool Support** ✓ (Test Passed - Expected Failure)
**Status:** Cannot handle LangChain StructuredTool objects

```python
@tool
def my_tool(x: int) -> int:
    return x * 2

# LangChain: works fine
# Custom: TypeError - not a callable
```

**Recommendation:** Detect StructuredTool and extract from `args_schema` or `.func`.

** Question from developer**: Do I need this?
---

### ⚠️ Interesting Differences (Tests Failed - Unexpected)

#### 1. **Annotated vs Docstring Precedence** (FAILED)
**Unexpected:** LangChain actually prefers **docstring** over Annotated when both exist

```python
def merge_files(
    source: Annotated[str, "Source file (from Annotated)"],
):
    """Args:
        source: Source file (from docstring)
    """
```

**Result:**
- LangChain: "Source file (from docstring)" ← Prefers docstring!
- Custom: "Source file (from docstring)" ← Also uses docstring

**Conclusion:** Both behave the same way. Test assumption was wrong.

#### 2. **Nested Dict Handling** (FAILED - Better Than Expected!)
**Unexpected:** Custom generator DOES preserve nested dict types!

```python
items: list[dict[str, str]]

# Expected: Custom would fail
# Actual: Custom correctly creates object type for items!
```

**Conclusion:** Custom schema generator is better than we thought at handling nested structures.

#### 3. **Optional Type Schema Format** (FAILED - Schema Structure)
**Issue:** LangChain uses `anyOf` schema, not `type` field directly

```python
optional_field: Optional[str] = None

# LangChain schema:
{
  "anyOf": [{"type": "string"}, {"type": "null"}]
}

# Custom schema:
{
  "type": ["string", "null"]
}
```

**Conclusion:** Both approaches are valid. Different schema formats for same concept.

#### 4. **Missing Docstring Handling** (FAILED - Bug!)
**Issue:** LangChain returns empty string for description, should have fallback

```python
def no_docs(x: str) -> str:
    return x

# LangChain: description = ""  ← Bug!
# Custom: description = "Execute no_docs"  ← Good fallback
```

**Conclusion:** Custom generator is actually better here!

#### 5. **Tool Scenario Decorator Bug** (FAILED - Bug in Custom!)
**Issue:** Custom generator is missing the `path` parameter after `@tool_scenarios` decorator

```python
@tool_scenarios(TaskType.FILE_OPERATION)
def annotated_tool(path: str, mode: str = "read") -> str:
    ...

# Custom only sees 'mode', not 'path'!
```

**Conclusion:** The decorator is interfering with signature inspection. Need to handle wrapped functions.

#### 6. ***args and **kwargs Handling** (FAILED - Bug in Custom!)
**Issue:** Custom generator includes `args` and `kwargs` in schema (should ignore them)

```python
def flexible(required: str, *args, **kwargs) -> str:
    ...

# Custom schema includes:
{
  "args": {"type": "string"},      ← Should not be here
  "kwargs": {"type": "string"}     ← Should not be here
}
```

**Conclusion:** Need to filter out VAR_POSITIONAL and VAR_KEYWORD parameter types.

---

## Priority Improvements

### High Priority (Stability)
1. **Filter *args and **kwargs** - Currently breaks schema (adds invalid params)
2. **Handle decorator wrapping** - `@tool_scenarios` breaks signature inspection
3. **Add StructuredTool support** - Needed for mixed tool lists (like planning)

### Medium Priority (Features)
4. **Add Enum support** - Would make schemas more precise
5. **Add Literal support** - Common pattern in type hints
6. **Extract from Annotated** - Would improve description quality

### Low Priority (Nice to Have)
7. **Union types with anyOf** - Complex but rarely needed
8. **Default values in schema** - Informational, not required
9. **Nested type preservation** - Already works well enough

---

## Implementation Recommendations

### 1. Filter VAR_POSITIONAL and VAR_KEYWORD (Critical Fix)

```python
# In function_to_json_schema(), around line 139:
for param_name, param in sig.parameters.items():
    # Skip internal parameters
    if param_name in INTERNAL_PARAMS:
        continue

    # !! ADD THIS !!
    # Skip *args and **kwargs
    if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
        continue

    # ... rest of code
```

### 2. Handle Wrapped Functions (Critical Fix)

```python
# At start of function_to_json_schema():
def function_to_json_schema(func: callable) -> dict:
    # Unwrap decorated functions
    while hasattr(func, '__wrapped__'):
        func = func.__wrapped__

    sig = inspect.signature(func)
    # ... rest of code
```

### 3. Add StructuredTool Support (High Priority)

```python
from langchain_core.tools import StructuredTool

def function_to_json_schema(func_or_tool):
    # Handle StructuredTool objects
    if isinstance(func_or_tool, StructuredTool):
        if hasattr(func_or_tool, 'func'):
            func = func_or_tool.func
        else:
            # Convert from args_schema (Pydantic model)
            return pydantic_to_json_schema(func_or_tool.args_schema)
    else:
        func = func_or_tool

    # ... rest of existing code
```

### 4. Add Enum Support (Medium Priority)

```python
# In _python_type_to_json_type():
from enum import Enum

def _python_type_to_json_type(py_type: Any) -> tuple[str, dict | None]:
    # Check for Enum
    if inspect.isclass(py_type) and issubclass(py_type, Enum):
        enum_values = [e.value for e in py_type]
        return "string", {"enum": enum_values}

    # ... rest of existing code
    return type_map.get(py_type, "string"), None

# Then update usage to handle the enum_schema:
param_type, enum_schema = _python_type_to_json_type(annotation)
if enum_schema:
    properties[param_name].update(enum_schema)
```

### 5. Extract from Annotated (Medium Priority)

```python
from typing import get_args, Annotated

def _extract_annotated_description(annotation: Any) -> str | None:
    """Extract description from Annotated type."""
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        # args[0] is the type, args[1:] are metadata
        for metadata in args[1:]:
            if isinstance(metadata, str):
                return metadata
    return None

# Then in function_to_json_schema():
# Prefer Annotated description over docstring
annotated_desc = _extract_annotated_description(annotation)
param_desc = annotated_desc or _extract_param_description(func, param_name)
```

---

## Conclusion

The custom `function_to_json_schema` is **surprisingly robust** and in some cases better than LangChain's implementation (fallback descriptions, nested types). However, it has **2 critical bugs** (*args/kwargs, decorator wrapping) that need immediate fixes for stability.

With the 5 recommended improvements above, the custom schema generator would be production-ready and could potentially replace `convert_to_openai_function` everywhere in the codebase, giving us:
- Consistent schema generation
- Better filtering capabilities (TaskType, side effects)
- No external dependencies (for this specific component)
- Full control over schema format

**Estimated effort:**
- Critical fixes (1-2): 2 hours
- StructuredTool support (3): 4 hours
- Enum/Literal/Annotated (4-5): 6 hours
- **Total: ~12 hours to achieve parity**
