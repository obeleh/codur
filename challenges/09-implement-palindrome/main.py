'''Challenge harness.'''

def is_palindrome(s: str) -> bool:
    """
    Determine if the given string is a palindrome.
    Requirements:
    1. Consider only alphanumeric characters and ignore case.
    2. Empty string or single character is considered a palindrome.
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