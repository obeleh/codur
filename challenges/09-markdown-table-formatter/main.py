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
    """
    # Split input into non‑empty lines, keep trailing spaces stripped only
    lines = [ln.rstrip() for ln in markdown_table.splitlines() if ln.strip()]
    if len(lines) < 2:
        return markdown_table.strip()

    # Helper to split a line into cells, ignoring leading/trailing pipes
    def split_line(line: str) -> list[str]:
        parts = line.split('|')[1:-1]
        return [p.strip() for p in parts]

    header_cells = split_line(lines[0])
    separator_raw = split_line(lines[1])
    body_rows = [split_line(l) for l in lines[2:]]

    # Determine alignment and colon presence for each column
    alignments: list[str] = []          # "left", "right", "center"
    left_colon: list[bool] = []
    right_colon: list[bool] = []
    for cell in separator_raw:
        stripped = cell.strip()
        left = stripped.startswith(":")
        right = stripped.endswith(":")
        left_colon.append(left)
        right_colon.append(right)
        if left and right:
            alignments.append("center")
        elif right:
            alignments.append("right")
        else:
            alignments.append("left")

    # Compute column widths based on header and body content (without surrounding spaces)
    col_count = len(header_cells)
    col_widths = [0] * col_count
    for row in [header_cells] + body_rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    # Helper to format a regular row according to alignment and column widths
    def format_row(cells: list[str]) -> str:
        formatted = []
        for idx, cell in enumerate(cells):
            width = col_widths[idx]
            align = alignments[idx]
            if align == "right":
                content = cell.rjust(width)
            elif align == "center":
                content = cell.center(width)
            else:  # left
                content = cell.ljust(width)
            # add a single space on each side
            formatted.append(f" {content} ")
        return "|" + "|".join(formatted) + "|"

    # Build the separator line preserving colon placement and matching column width
    separator_parts = []
    for idx, width in enumerate(col_widths):
        # For left/right alignment we need one extra dash to account for the padding space on that side
        if alignments[idx] == "center":
            dash_len = width  # exactly the content width
        else:
            dash_len = width + 1  # include the extra dash for the side padding
        if left_colon[idx] and right_colon[idx]:
            # colon both sides
            sep = ":" + "-" * dash_len + ":"
        elif left_colon[idx]:
            # colon on the left only
            sep = ":" + "-" * dash_len
        elif right_colon[idx]:
            # colon on the right only
            sep = "-" * dash_len + ":"
        else:
            sep = "-" * dash_len
        separator_parts.append(sep)
    separator_line = "|" + "|".join(separator_parts) + "|"

    # Assemble final table
    formatted_lines = [format_row(header_cells), separator_line]
    for row in body_rows:
        formatted_lines.append(format_row(row))
    return "\n".join(formatted_lines)


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
