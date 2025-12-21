# Codur Comprehensive Refactoring Plan

**User requested**: Comprehensive refactoring of Codur application addressing maintainability, extensibility, and code quality with breaking changes acceptable and comprehensive unit tests.

---

## Executive Summary

**Scope**: 5 phases covering ~4,100 lines across 15+ files
**Effort**: 15-20 days of focused development
**Risk**: Medium (breaking changes acceptable per requirements)
**Goal**: Eliminate duplication, simplify complex functions, improve abstractions, centralize configuration

---

## Key Issues to Address

### Critical (HIGH Priority)
1. **Code Duplication** (~300 lines total)
   - CLI execution duplicated in Codex and Claude Code agents (~170 lines)
   - Path resolution duplicated in filesystem.py and structured.py (~40 lines)
   - LLM retry logic duplicated 3+ times in planning.py (~50 lines)
   - JSON parsing duplicated

2. **Complex Functions**
   - `plan_node()`: 172 lines with 6+ retry paths
   - `_detect_tool_operations()`: 160 lines of regex matching
   - `execute_node()`: 114 lines with 3 execution paths

3. **Missing Abstractions**
   - No base class for CLI-based agents
   - No abstraction for retry/fallback logic
   - No shared path resolution utilities

### Important (MEDIUM Priority)
4. **Configuration Issues**: 10+ hardcoded values (DEBUG_TRUNCATE_SHORT=500, GREETING_MAX_WORDS=3, etc.)
5. **Inconsistent Patterns**: Default agent fallback implemented 4 different ways
6. **Error Handling**: 3 different error handling patterns across codebase

---

## Phase 1: Foundation - Create Core Abstractions (3-4 days)

**Goal**: Build foundational abstractions to eliminate duplication

### 1.1 Create CLI Agent Base Class
**New file**: `codur/agents/cli_agent_base.py`
- Extract common CLI execution patterns from Codex and Claude Code
- Implement `_build_command()` abstract method
- Implement `_execute_cli()` and `_aexecute_cli()` with proper error handling
- Eliminates ~170 lines of duplication

**Files to modify**:
- `codur/agents/codex_agent.py` - Inherit from BaseCLIAgent
- `codur/agents/claude_code_agent.py` - Inherit from BaseCLIAgent

**Tests**: `tests/agents/test_cli_agent_base.py` (20+ tests)

### 1.2 Create Path Resolution Utilities
**New file**: `codur/utils/path_utils.py`
- `resolve_root(root)` - Resolve root directory
- `resolve_path(path, root, allow_outside_root)` - Resolve with security checks
- Eliminates ~40 lines of duplication

**Files to modify**:
- `codur/tools/filesystem.py` - Import from path_utils
- `codur/tools/structured.py` - Import from path_utils

**Tests**: `tests/utils/test_path_utils.py` (15+ tests)

### 1.3 Create Retry Logic Abstraction
**New file**: `codur/utils/retry.py`
- `RetryStrategy` class with configurable backoff
- `retry_with_backoff(func, strategy)` generic retry function
- `LLMRetryStrategy` with `invoke_with_retries()` and `invoke_with_fallbacks()`
- Eliminates ~50 lines of retry duplication

**Files to modify**:
- `codur/graph/nodes/planning.py` - Replace retry functions with utilities

**Tests**: `tests/utils/test_retry.py` (10+ tests)

---

## Phase 2: Decompose Complex Functions (4-5 days)

**Goal**: Break down complex functions into testable units

### 2.1 Refactor Planning Node
**Current**: 172-line `plan_node()` function with nested logic

**New structure**: Extract into focused modules
```
codur/graph/nodes/planning/
├── __init__.py              # Public interface
├── core.py                  # PlanningOrchestrator class
├── json_parser.py           # JSONResponseParser class
├── prompt_builder.py        # PlanningPromptBuilder class
├── decision_handler.py      # PlanningDecisionHandler class
└── validators.py            # Validation functions
```

**Key classes**:
- `PlanningOrchestrator` - Main planning logic (~50 lines)
- `JSONResponseParser` - Parse and validate JSON responses
- `PlanningPromptBuilder` - Build system prompts and messages
- `PlanningDecisionHandler` - Handle planning decisions and errors
- Helper functions: `looks_like_change_request()`, `mentions_file_path()`, `has_mutation_tool()`

**Files to modify**:
- `codur/graph/nodes/planning.py` - Refactor to use new classes (650 lines → ~150 lines)

**Impact**: Reduces file by 500 lines, creates 5-6 testable modules

**Tests**:
- `tests/graph/nodes/planning/test_core.py`
- `tests/graph/nodes/planning/test_json_parser.py`
- `tests/graph/nodes/planning/test_prompt_builder.py`
- `tests/graph/nodes/planning/test_decision_handler.py`
- `tests/graph/nodes/planning/test_validators.py`
(50+ tests total)

### 2.2 Refactor Tool Detection
**Current**: 160-line `_detect_tool_operations()` function with regex hell

**New file**: `codur/graph/nodes/tool_detection.py`

**New approach**: Registry-based pattern matcher
- `ToolPattern` class - Represents a detection pattern
- `ToolDetector` class - Registry with `register()` and `detect()` methods
- Each tool type registered separately with priority

**Impact**: 160 lines → ~30 lines registry + separate extractors, makes patterns extensible

**Files to modify**:
- `codur/graph/nodes/non_llm_tools.py` - Use ToolDetector

**Tests**: `tests/graph/nodes/test_tool_detector.py` (25+ tests)

### 2.3 Simplify Execution Node
**Current**: 114-line function with 3 execution paths

**New approach**: Extract `AgentExecutor` class
- `execute()` - Main entry point
- `_execute_llm_profile()` - Handle LLM profiles
- `_execute_registered_agent()` - Handle agent registry
- Separate concerns clearly

**Files to modify**:
- `codur/graph/nodes/execution.py` - Use AgentExecutor class

**Tests**: `tests/graph/nodes/test_execution.py` (20+ tests)

---

## Phase 3: Configuration & Constants (2-3 days)

**Goal**: Eliminate hardcoded values, add validation

### 3.1 Centralize Constants
**New file**: `codur/constants.py`
- All magic numbers in one place
- DEBUG_TRUNCATE_SHORT, DEBUG_TRUNCATE_LONG, GREETING_MAX_WORDS, etc.

**Files to modify**: All files with magic numbers import from constants

### 3.2 Extend Configuration Model
**File to modify**: `codur/config.py`

**New config classes**:
```python
class PlanningSettings(BaseModel):
    debug_truncate_short: int = 500
    debug_truncate_long: int = 1000
    greeting_max_words: int = 3
    max_retry_attempts: int = 3
    retry_initial_delay: float = 0.5

class ToolSettings(BaseModel):
    default_max_bytes: int = 200_000
    default_max_results: int = 200
    default_timeout: int = 600
    exclude_dirs: set[str] = {".git", ".venv", ...}

class AgentExecutionSettings(BaseModel):
    default_cli_timeout: int = 600
    claude_code_max_tokens: int = 8000
```

**Add validation**:
- `@root_validator` - Validate default_agent is set
- `@validator` - Validate runtime settings
- Catch misconfigurations at startup

**Tests**: `tests/test_config.py` (30+ tests)

### 3.3 Create Migration Guide
**New file**: `docs/MIGRATION_GUIDE.md`
- Document breaking config changes
- Provide migration examples

---

## Phase 4: Error Handling & Testing (4-5 days)

**Goal**: Standardize error handling, achieve 80%+ test coverage

### 4.1 Create Custom Exception Hierarchy
**New file**: `codur/exceptions.py`

```python
class CodurError(Exception): pass
class ConfigurationError(CodurError): pass
class AgentError(CodurError): pass
class AgentNotFoundError(AgentError): pass
class AgentTimeoutError(AgentError): pass
class PlanningError(CodurError): pass
class ToolError(CodurError): pass
class PathSecurityError(ToolError): pass
class LLMError(CodurError): pass
class RetryExhaustedError(CodurError): pass
```

**Files to modify**: All files - use custom exceptions instead of generic Exception

### 4.2 Comprehensive Test Suite
**Test structure**:
```
tests/
├── agents/              # Agent tests
├── graph/nodes/         # Node tests (planning, execution, tools)
├── tools/               # Tool tests
├── utils/               # Utility tests (path_utils, retry)
├── providers/           # Provider tests
├── integration/         # End-to-end tests
├── test_config.py
├── test_llm.py
└── test_exceptions.py
```

**Coverage goals**:
- Overall: 80%+
- Critical paths (planning, execution): 90%+
- Utilities: 95%+

**Total**: 200+ new tests

### 4.3 Standardize Error Handling Patterns
- Agent errors: Use AgentError with context (agent name, original error)
- Tool errors: Use ToolError with context (tool name, original error)
- LLM errors: Use LLMError for invocation failures
- Consistent error messages and logging

---

## Phase 5: Documentation & Polish (2-3 days)

**Goal**: Clean up deprecated code, add documentation

### 5.1 Remove Deprecated Code
- Remove `PLANNING_SYSTEM_PROMPT` (line 132-178 in planning.py)
- Remove duplicate agent configurations
- Search and address TODO comments

### 5.2 Add Docstrings and Type Hints
- Complete docstrings for all public functions
- Class-level documentation
- Example usage in complex functions
- Type hints for all parameters and returns

### 5.3 Update Documentation
**Files to create/update**:
- `docs/ARCHITECTURE.md` - Document refactored architecture
- `docs/TESTING.md` - Testing strategy and how to run tests
- `docs/ERROR_HANDLING.md` - Exception hierarchy and patterns
- `docs/DEVELOPER_GUIDE.md` - How to add agents, tools, providers
- `README.md` - Update with refactoring notes

---

## Critical Files to Modify

**Top 5 priority files**:
1. `/Users/sjuul/workspace/codur/codur/graph/nodes/planning.py` (657 lines → ~150)
2. `/Users/sjuul/workspace/codur/codur/agents/base.py` (extend with BaseCLIAgent)
3. `/Users/sjuul/workspace/codur/codur/config.py` (add new config classes)
4. `/Users/sjuul/workspace/codur/codur/tools/filesystem.py` (extract path utils)
5. `/Users/sjuul/workspace/codur/codur/agents/codex_agent.py` (refactor to use BaseCLIAgent)

**Other important files**:
- `codur/graph/nodes/non_llm_tools.py` (tool detection refactor)
- `codur/graph/nodes/execution.py` (simplify execution)
- `codur/agents/claude_code_agent.py` (refactor to use BaseCLIAgent)
- `codur/tools/structured.py` (use path_utils)
- `codur/graph/nodes/routing.py` (remove MAX_ITERATIONS constant)

---

## Migration Strategy

1. **Create feature branch**: `git checkout -b refactor/comprehensive-cleanup`

2. **Incremental migration**:
   - Phase 1: Create new abstractions (coexist with old code)
   - Phase 2: Migrate one node at a time
   - Phase 3: Update configuration with defaults
   - Phase 4: Add tests incrementally
   - Phase 5: Final cleanup

3. **Testing safety net**:
   - Run `tests/test_challenges.py` after each phase
   - Create integration tests for full workflows
   - Use pytest fixtures with real configuration
   - Mock external dependencies (LLMs, subprocess)

4. **Backward compatibility**:
   - Support old config format during migration
   - Add deprecation warnings
   - Provide migration script

---

## Success Metrics

**Code Quality**:
- ✅ Reduce planning.py from 657 to ~150 lines (77% reduction)
- ✅ Eliminate 300+ lines of duplication
- ✅ Achieve 80%+ test coverage
- ✅ Zero hardcoded constants in logic

**Maintainability**:
- ✅ Max function length: 50 lines
- ✅ Max class length: 200 lines
- ✅ All public APIs documented
- ✅ Clear separation of concerns

**Reliability**:
- ✅ All retry logic tested
- ✅ All error paths tested
- ✅ Configuration validated at startup
- ✅ Type safety via Pydantic

---

## Risk Mitigation

**High-risk areas**:
1. **Planning node refactoring** - Create tests first, refactor incrementally
2. **Configuration changes** - Maintain backward compatibility initially
3. **Error handling changes** - Update all error handling in same commit

**Testing safety**:
- Comprehensive unit tests before refactoring
- Integration tests for critical workflows
- Feature flags to switch between old/new implementations
- Run challenges after each phase

---

## Next Steps

1. ✅ Review and approve this plan
2. Create GitHub issues for each phase
3. Set up feature branch with branch protection
4. Begin Phase 1: Foundation abstractions
5. Run CI/CD after each phase
6. Document learnings as you go

---

## Estimated Timeline

- **Phase 1**: 3-4 days (Foundation)
- **Phase 2**: 4-5 days (Decomposition)
- **Phase 3**: 2-3 days (Configuration)
- **Phase 4**: 4-5 days (Testing & Error Handling)
- **Phase 5**: 2-3 days (Documentation)

**Total**: 15-20 days of focused development

---

## Post-Refactoring Benefits

1. **Easier agent development**: BaseCLIAgent makes new CLI agents trivial
2. **Extensible tool detection**: Pattern registry for easy tool additions
3. **Better debugging**: Custom exceptions with context
4. **Configuration flexibility**: All tuning parameters in config
5. **Test confidence**: Comprehensive suite catches regressions
6. **Faster onboarding**: Clear architecture and documentation

---

*This plan can also be saved as `refactor_plan.md` in the project root for reference.*