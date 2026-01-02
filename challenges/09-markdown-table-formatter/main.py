import sys


def format_table(markdown_table: str) -> str:
    """
    Parses and reformats a Markdown table.

    Rules:
    1. Columns are delimited by '|'.
    2. The second row contains the separator and alignment indicators.
       - '---' or ':---' means Left Align (default).
       - '---:' means Right Align.
       - ':---:' means Center Align.
    3. Output columns should be wide enough to fit the widest content + 1 space padding on each side.
    4. Leading and trailing pipes must be preserved and aligned.
    5. Strip whitespace from cell content before determining width.
    6. In case of center alignment with odd extra space, favor left padding.
    """
    # TODO: Implement parsing and formatting logic.
    # Currently just returns input, which will fail the tests.
    return markdown_table.strip()


def run_tests():
    test_cases = [
        {
            "name": "Basic Left Alignment",
            "input": """
| Name | Age | City |
|---|---|---|
| Alice | 30 | New York |
| Bob | 25 | Los Angeles |
""",
            "expected": """
| Name  | Age | City        |
|-------|-----|-------------|
| Alice | 30  | New York    |
| Bob   | 25  | Los Angeles |
""".strip()
        },
        {
            "name": "Mixed Alignment",
            "input": """
| Item | Price | Stock |
|:---|---:|:---:|
| Apple | $1.00 | 50 |
| Banana | $0.50 | 100 |
""",
            "expected": """
| Item   | Price | Stock |
|:-------|------:|:-----:|
| Apple  | $1.00 |  50   |
| Banana | $0.50 |  100  |
""".strip()
        },
        {
            "name": "Uneven Whitespace",
            "input": """
|   Col A   |Col B|    Col C    |
|:---:|:---|---:|
|   1   |   Left   |   Right   |
| 100 | Loooooong | R |
""",
            "expected": """
| Col A | Col B     | Col C |
|:-----:|:----------|------:|
|   1   | Left      | Right |
|  100  | Loooooong |     R |
""".strip()
        }
    ]

    failures = []
    for case in test_cases:
        actual = format_table(case["input"].strip())
        expected = case["expected"]
        if actual != expected:
            failures.append((case["name"], expected, actual))
        else:
            print(f"Test '{case['name']}': PASSED")

    if failures:
        print("\nFAILURES:")
        for name, exp, act in failures:
            print(f"--- {name} ---")
            print("EXPECTED:")
            print(exp)
            print("ACTUAL:")
            print(act)
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
