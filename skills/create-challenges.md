# Create Challenges Skill

Design coding challenges that test Codur's agentic capabilities: reading files, writing code, validating output, and iterating on failures.

---

## Challenge Structure

Each challenge must be in `challenges/<NN>-<name>/` with exactly three files:

### 1. `prompt.txt` - User instruction
```
Implement the password validator function in @main.py based on the docstring.
Write unit tests in @test_main.py. Run pytest to validate all requirements.
```

**Guidelines:**
- Clear, specific task (not ambiguous)
- Reference files with `@filename.py`
- Avoid hardcoding the solution

### 2. `main.py` - Starter code or buggy code
```python
"""Challenge harness."""

def validate_password(password: str) -> bool:
    """
    Validate a password:
    1. At least 8 characters
    2. One uppercase letter
    3. One lowercase letter
    4. One digit
    5. One special character (!@#$%^&*)
    """
    # TODO: Implement
    pass

if __name__ == "__main__":
    import subprocess, sys
    result = subprocess.run([sys.executable, "-m", "pytest", "test_main.py", "-v"],
                          capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("TESTS FAILED!")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED!")
```

**Guidelines:**
- Executable with `python main.py`
- Prints deterministic output
- For bugs: include buggy code that differs from expected output
- For implementation: include docstring spec + test runner

### 3. `expected.txt` - Exact expected output
```
test_main.py::test_valid_password PASSED
test_main.py::test_password_too_short PASSED
test_main.py::test_missing_uppercase PASSED
test_main.py::test_missing_lowercase PASSED
test_main.py::test_missing_digit PASSED
test_main.py::test_missing_special_char PASSED

====== 6 passed in 0.01s ======
ALL TESTS PASSED!
```

**Guidelines:**
- Must be EXACT output of `python main.py` when solved
- Include all output: tests, messages, everything
- Trailing newlines matter

---

## Challenge Levels

**Level 1: Bug Fixes** (Challenge 01)
- Tools: read_file, replace_in_file
- Example: Fix off-by-one error
- Agent: Read file → analyze → fix → verify

**Level 2: Implementation + Testing** (Challenge 02)
- Tools: read_file, write_file, bash (pytest)
- Example: Implement password validator with tests
- Agent: Read spec → write tests → implement → run pytest → iterate if needed

**Level 3: Module Creation** (Challenge 03+)
- Tools: write_file (multiple), bash (pytest, mypy)
- Example: Create logging module with type hints
- Agent: Create multiple files → type check → test → iterate

---

## Design Guidelines

### ✅ DO:
- Write clear docstrings (agent learns from specs)
- Require multiple tools (read, write, bash)
- Allow iteration (tests fail → fix → re-run)
- Test through execution (`python main.py` + pytest/mypy)
- Use real-world patterns
- Make it solvable without hardcoding

### ❌ DON'T:
- Hardcode solutions in main.py
- Make ambiguous prompts
- Require external APIs or network calls
- Mix multiple programming languages
- Skip tests for verification
- Use vague error messages

---

## Creating a Challenge

### 1. Design
Ask yourself:
- What agentic capability am I testing? (reading, writing, validating, iterating?)
- What tools should the agent use? (2+ different ones?)
- Is it solvable by an LLM? (Does spec follow from docstring/requirements?)
- Is it harder than previous challenges?

### 2. Write files
- **prompt.txt**: Clear instruction referencing @files
- **main.py**: Starter code (buggy or incomplete) with docstring spec
- **expected.txt**: Run correct implementation, capture output

### 3. Test manually
```bash
cd challenges/02-implement-password-validator
python -m codur.cli --command "$(cat prompt.txt)" --raw --config ../../codur.yaml
python main.py
# Compare output with expected.txt
```

### 4. Test with pytest
```bash
pytest tests/test_challenges.py -v
```

---

## Quick Checklist

- [ ] Directory: `challenges/NN-name/` exists
- [ ] `prompt.txt`: Clear, references @files
- [ ] `main.py`: Executable, has bugs/incomplete code, includes docstring spec
- [ ] `expected.txt`: Exact output of correct implementation
- [ ] Manual test: Codur can solve it, output matches expected.txt
- [ ] Harder than previous? Uses 2+ tools? Requires iteration?

---

## Challenge Ideas

**Implementation + Testing:**
- Implement sorting algorithm + write tests
- Implement JSON parser + validation tests
- Implement cache data structure + performance tests

**Module Creation:**
- Create configuration loader with validation
- Create logging framework with handlers
- Create API client with retry logic

**Refactoring:**
- Refactor to async/await
- Add type hints to untyped code
- Optimize slow queries

---

**Last Updated:** 2025-12-21 | **Status:** Ready to use