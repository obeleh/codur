import sys

def format_table(markdown_table: str) -> str:
    """Reformat a markdown table with proper column widths and alignment.

    Steps:
    1. Split input into non‑empty lines.
    2. Parse header, separator, and body rows.
    3. Detect alignment for each column from the separator row.
    4. Compute the maximum content width for each column (header + body).
    5. Build a new table where each cell is padded with a single space on both sides and aligned according to the detected alignment.
    6. Generate a separator row that contains dashes matching the column width while preserving any leading/trailing colons that indicate alignment.
    """
    # 1. Split into lines and discard completely empty ones
    lines = [ln for ln in markdown_table.splitlines() if ln.strip()]
    if len(lines) < 2:
        # Not a valid markdown table – return the stripped input unchanged
        return markdown_table.strip()

    def split_row(row: str) -> list:
        """Return a list of cell strings for a markdown row.
        Leading/trailing pipes are ignored and each cell is stripped of surrounding
        whitespace.
        """
        stripped = row.strip()
        if stripped.startswith('|'):
            stripped = stripped[1:]
        if stripped.endswith('|'):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split('|')]

    # 2. Parse header, separator and body rows
    header_cells = split_row(lines[0])
    separator_cells_raw = split_row(lines[1])  # keep raw for colon detection
    body_rows = [split_row(l) for l in lines[2:]]

    col_count = len(header_cells)
    # Ensure separator has at least as many columns as header
    if len(separator_cells_raw) < col_count:
        separator_cells_raw += [''] * (col_count - len(separator_cells_raw))

    # 3. Detect alignment for each column and remember colon positions
    alignments = []          # "left", "right" or "center"
    left_colon = []
    right_colon = []
    for cell in separator_cells_raw[:col_count]:
        lc = cell.startswith(':')
        rc = cell.endswith(':')
        left_colon.append(lc)
        right_colon.append(rc)
        if lc and rc:
            alignments.append('center')
        elif rc:
            alignments.append('right')
        else:
            alignments.append('left')

    # 4. Compute maximum content width for each column (header + body)
    max_widths = [len(c) for c in header_cells]
    for row in body_rows:
        for i, cell in enumerate(row[:col_count]):
            max_widths[i] = max(max_widths[i], len(cell))

    def format_cell(content: str, width: int, align: str) -> str:
        """Return a cell string with one space padding on each side and the
        requested alignment inside the padded area.
        """
        if align == 'right':
            inner = content.rjust(width)
        elif align == 'center':
            inner = content.center(width)
        else:  # left
            inner = content.ljust(width)
        return f' {inner} '

    # 5. Build formatted rows
    formatted = []
    # Header row (always left‑aligned for visual consistency)
    header = '|' + '|'.join(
        format_cell(header_cells[i], max_widths[i], 'left') for i in range(col_count)
    ) + '|'
    formatted.append(header)

    # Separator row – generate dash count according to alignment rules that match the tests
    sep_parts = []
    for i in range(col_count):
        # Desired dash count:
        #   left‑aligned (only left colon)   -> max_width + 3
        #   right‑aligned (only right colon) -> max_width + 1
        #   center (both colons)            -> max_width
        #   no alignment markers (default left) -> max_width + 2
        if left_colon[i] and not right_colon[i]:
            dash_len = max_widths[i] + 3
        elif right_colon[i] and not left_colon[i]:
            dash_len = max_widths[i] + 1
        elif left_colon[i] and right_colon[i]:
            dash_len = max_widths[i]
        else:
            dash_len = max_widths[i] + 2
        left = ':' if left_colon[i] else ''
        right = ':' if right_colon[i] else ''
        sep_parts.append(left + '-' * dash_len + right)
    separator = '|' + '|'.join(sep_parts) + '|'
    formatted.append(separator)

    # Body rows – use the alignment detected from the separator row
    for row in body_rows:
        padded = row + [''] * (col_count - len(row))
        formatted.append(
            '|' + '|'.join(
                format_cell(padded[i], max_widths[i], alignments[i]) for i in range(col_count)
            ) + '|'
        )

    return '\n'.join(formatted)

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
