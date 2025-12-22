import pytest
import sys

# This hook is a pluggy hook specification from pytest
# hookwrapper=True allows us to wrap the execution and access the result
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Extends the test report to surface better info on failure.
    """
    # Execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # We only care about the actual test execution (call), not setup or teardown
    if rep.when == "call" and rep.failed:
        # Retrieve captured stdout/stderr if available
        # Note: Pytest automatically captures this, but we can add custom sections
        
        # Example: Add a section with the test docstring if available
        doc = item.obj.__doc__
        if doc:
            # Clean up indentation
            from inspect import cleandoc
            clean_doc = cleandoc(doc)
            rep.sections.append(("Test Description", clean_doc))

@pytest.hookimpl
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    Add a custom summary section at the end of the test session using pluggy.
    """
    # Get all failed reports
    failures = terminalreporter.stats.get("failed", [])
    
    if failures:
        terminalreporter.section("Codur Debug Context", sep="=", red=True, bold=True)
        terminalreporter.write_line(f"Found {len(failures)} failures.")
        terminalreporter.write_line("Tip: Use 'codebase_investigator' to analyze the specific test files.")
