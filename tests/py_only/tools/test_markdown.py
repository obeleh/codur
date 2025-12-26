"""Tests for markdown tools."""

import pytest
from pathlib import Path

from codur.tools.markdown import (
    markdown_outline,
    markdown_extract_sections,
    markdown_extract_tables,
)


@pytest.fixture
def temp_fs(tmp_path):
    """Create a temporary workspace directory."""
    root = tmp_path / "workspace"
    root.mkdir()
    return root


class TestMarkdownOutline:
    """Tests for markdown_outline function."""

    def test_simple_headers(self, temp_fs):
        """Test extracting outline from simple headers."""
        content = """# Main Title
## Section 1
### Subsection 1.1
## Section 2
### Subsection 2.1
### Subsection 2.2
"""
        md_file = temp_fs / "test.md"
        md_file.write_text(content, encoding='utf-8')

        outline = markdown_outline(str(md_file))

        expected = """# Main Title
  ## Section 1
    ### Subsection 1.1
  ## Section 2
    ### Subsection 2.1
    ### Subsection 2.2"""

        assert outline == expected

    def test_deep_nesting(self, temp_fs):
        """Test outline with deeply nested headers."""
        content = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6
"""
        md_file = temp_fs / "deep.md"
        md_file.write_text(content, encoding='utf-8')

        outline = markdown_outline(str(md_file))

        assert "# Level 1" in outline
        assert "  ## Level 2" in outline
        assert "    ### Level 3" in outline
        assert "      #### Level 4" in outline
        assert "        ##### Level 5" in outline
        assert "          ###### Level 6" in outline

    def test_headers_with_mixed_content(self, temp_fs):
        """Test that only headers are extracted, not body text."""
        content = """# Title
Some body text here.

## Section
More text.
- Bullet point
- Another point

### Subsection
Even more text.
"""
        md_file = temp_fs / "mixed.md"
        md_file.write_text(content, encoding='utf-8')

        outline = markdown_outline(str(md_file))

        assert "# Title" in outline
        assert "  ## Section" in outline
        assert "    ### Subsection" in outline
        assert "body text" not in outline
        assert "Bullet point" not in outline

    def test_no_headers(self, temp_fs):
        """Test file with no headers returns appropriate message."""
        content = """Just some text without any headers.
More text.
Even more text.
"""
        md_file = temp_fs / "no_headers.md"
        md_file.write_text(content, encoding='utf-8')

        outline = markdown_outline(str(md_file))

        assert outline == "No headers found in markdown file"

    def test_empty_file(self, temp_fs):
        """Test empty file returns appropriate message."""
        md_file = temp_fs / "empty.md"
        md_file.write_text("", encoding='utf-8')

        outline = markdown_outline(str(md_file))

        assert outline == "No headers found in markdown file"


class TestMarkdownExtractSections:
    """Tests for markdown_extract_sections function."""

    def test_extract_single_section(self, temp_fs):
        """Test extracting a single section."""
        content = """# Main Title
Introduction text.

## Installation
Install via pip:
```bash
pip install package
```

## Usage
Use it like this.
"""
        md_file = temp_fs / "readme.md"
        md_file.write_text(content, encoding='utf-8')

        sections = markdown_extract_sections(str(md_file), ["Installation"])

        assert "Installation" in sections
        assert "## Installation" in sections["Installation"]
        assert "pip install package" in sections["Installation"]
        assert "Usage" not in sections

    def test_extract_multiple_sections(self, temp_fs):
        """Test extracting multiple sections."""
        content = """# Title
## Intro
Some intro.

## Features
- Feature 1
- Feature 2

## Installation
Install steps.

## Contributing
Contribution guide.
"""
        md_file = temp_fs / "doc.md"
        md_file.write_text(content, encoding='utf-8')

        sections = markdown_extract_sections(str(md_file), ["Features", "Contributing"])

        assert "Features" in sections
        assert "Contributing" in sections
        assert "Feature 1" in sections["Features"]
        assert "Contribution guide" in sections["Contributing"]
        assert "Installation" not in sections

    def test_extract_nested_subsections(self, temp_fs):
        """Test that subsections are included in parent section."""
        content = """# Main
## Section A
Content A.

### Subsection A.1
More content.

### Subsection A.2
Even more.

## Section B
Content B.
"""
        md_file = temp_fs / "nested.md"
        md_file.write_text(content, encoding='utf-8')

        sections = markdown_extract_sections(str(md_file), ["Section A"])

        assert "Section A" in sections
        assert "Subsection A.1" in sections["Section A"]
        assert "Subsection A.2" in sections["Section A"]
        assert "Section B" not in sections["Section A"]

    def test_section_not_found(self, temp_fs):
        """Test requesting non-existent section."""
        content = """# Title
## Existing Section
Content here.
"""
        md_file = temp_fs / "test.md"
        md_file.write_text(content, encoding='utf-8')

        sections = markdown_extract_sections(str(md_file), ["NonExistent"])

        assert "NonExistent" not in sections
        assert len(sections) == 0

    def test_empty_section_names(self, temp_fs):
        """Test with empty section names list."""
        content = """# Title
## Section
Content.
"""
        md_file = temp_fs / "test.md"
        md_file.write_text(content, encoding='utf-8')

        sections = markdown_extract_sections(str(md_file), [])

        assert len(sections) == 0


class TestMarkdownExtractTables:
    """Tests for markdown_extract_tables function."""

    def test_simple_table(self, temp_fs):
        """Test extracting a simple table."""
        content = """# Document
Some text.

| Name  | Age |
| ----- | --- |
| Alice | 30  |
| Bob   | 25  |

More text.
"""
        md_file = temp_fs / "table.md"
        md_file.write_text(content, encoding='utf-8')

        tables = markdown_extract_tables(str(md_file))

        assert len(tables) == 1
        assert tables[0]["headers"] == ["Name", "Age"]
        assert tables[0]["rows"] == [["Alice", "30"], ["Bob", "25"]]

    def test_multiple_tables(self, temp_fs):
        """Test extracting multiple tables."""
        content = """# Report
## Users
| Name  | Role  |
| ----- | ----- |
| Alice | Admin |
| Bob   | User  |

## Products
| Product | Price |
| ------- | ----- |
| Widget  | $10   |
| Gadget  | $20   |
"""
        md_file = temp_fs / "multi.md"
        md_file.write_text(content, encoding='utf-8')

        tables = markdown_extract_tables(str(md_file))

        assert len(tables) == 2
        assert tables[0]["headers"] == ["Name", "Role"]
        assert tables[0]["rows"] == [["Alice", "Admin"], ["Bob", "User"]]
        assert tables[1]["headers"] == ["Product", "Price"]
        assert tables[1]["rows"] == [["Widget", "$10"], ["Gadget", "$20"]]

    def test_table_with_varying_columns(self, temp_fs):
        """Test table with different column counts."""
        content = """| Col1 | Col2 | Col3 |
| ---- | ---- | ---- |
| A    | B    | C    |
| X    | Y    | Z    |
"""
        md_file = temp_fs / "cols.md"
        md_file.write_text(content, encoding='utf-8')

        tables = markdown_extract_tables(str(md_file))

        assert len(tables) == 1
        assert len(tables[0]["headers"]) == 3
        assert len(tables[0]["rows"]) == 2

    def test_no_tables(self, temp_fs):
        """Test file with no tables."""
        content = """# Title
Just some text without tables.
No pipes here.
"""
        md_file = temp_fs / "no_tables.md"
        md_file.write_text(content, encoding='utf-8')

        tables = markdown_extract_tables(str(md_file))

        assert len(tables) == 0

    def test_table_with_empty_cells(self, temp_fs):
        """Test table with some empty cells."""
        content = """| Name  | Value |
| ----- | ----- |
| Key1  |       |
|       | Val2  |
| Key3  | Val3  |
"""
        md_file = temp_fs / "empty.md"
        md_file.write_text(content, encoding='utf-8')

        tables = markdown_extract_tables(str(md_file))

        assert len(tables) == 1
        assert tables[0]["rows"][0] == ["Key1", ""]
        assert tables[0]["rows"][1] == ["", "Val2"]
        assert tables[0]["rows"][2] == ["Key3", "Val3"]

    def test_table_at_end_of_file(self, temp_fs):
        """Test that table at end of file is captured."""
        content = """# Title
Some text.

| Col1 | Col2 |
| ---- | ---- |
| A    | B    |
| C    | D    |"""
        md_file = temp_fs / "end.md"
        md_file.write_text(content, encoding='utf-8')

        tables = markdown_extract_tables(str(md_file))

        assert len(tables) == 1
        assert tables[0]["headers"] == ["Col1", "Col2"]
        assert len(tables[0]["rows"]) == 2
