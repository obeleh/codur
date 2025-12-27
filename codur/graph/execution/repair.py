"""Local repair logic for failed fixes."""

import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from codur.graph.state import AgentState
from codur.tools.project_discovery import get_primary_entry_point


def _attempt_local_repair(state: AgentState) -> dict:
    """Attempt a small, local repair for common mismatch patterns.

    This is a last-resort fallback when external agents are unavailable.
    Uses parallel execution for faster mutation testing.
    Supports both main.py and app.py entry points.
    """
    cwd = Path.cwd()
    expected_file = cwd / "expected.txt"

    # Discover the entry point (main.py or app.py)
    entry_point_name = get_primary_entry_point(root=cwd)
    if entry_point_name.startswith("Error"):
        return {"success": False, "message": entry_point_name}

    entry_point_file = cwd / entry_point_name

    if not entry_point_file.exists() or not expected_file.exists():
        return {"success": False, "message": "No repair target found"}

    original = entry_point_file.read_text()
    expected_output = expected_file.read_text().strip()

    def mutate_range_inclusive(text: str) -> str:
        def _replace(match: re.Match) -> str:
            start = match.group(1).strip()
            end = match.group(2).strip()
            if end.endswith("+ 1") or end.endswith("+1"):
                return match.group(0)
            return f"range({start}, {end} + 1)"
        return re.sub(r"\brange\(([^,]+),\s*([^)]+)\)", _replace, text)

    def mutate_remove_continue_guard(text: str) -> str:
        pattern = re.compile(r"^(?P<indent>\s*)if\s+(?P<cond>.+):\n(?P=indent)\s+continue\b", re.MULTILINE)
        return pattern.sub(lambda m: f"{m.group('indent')}if {m.group('cond')}:\n{m.group('indent')}    pass", text)

    def mutate_remove_div_100(text: str) -> str:
        text = re.sub(r"\(([^()]+?)\s*/\s*100(?:\.0+)?\)", r"\1", text)
        return re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*/\s*100(?:\.0+)?\b", r"\1", text)

    def mutate_fix_comparison(text: str) -> str:
        """Fix common comparison mistakes like >= vs >, <= vs <"""
        # Try flipping >= to >
        alt1 = text.replace(">=", "__GTE__").replace(">", ">=").replace("__GTE__", ">")
        if alt1 != text:
            return alt1
        # Try flipping <= to <
        alt2 = text.replace("<=", "__LTE__").replace("<", "<=").replace("__LTE__", "<")
        if alt2 != text:
            return alt2
        return text

    def mutate_fix_loop_condition(text: str) -> str:
        """Fix loop conditions that are too strict or too loose"""
        # Change 'while n' to 'while n > 0' (common mistake)
        text = re.sub(r"\bwhile\s+([a-z_]\w*)\s*:", r"while \1 > 0:", text)
        return text

    def mutate_add_f_prefix(text: str) -> str:
        """Add missing f-string prefix"""
        # Finds "{var}" inside quotes without f-prefix
        return re.sub(r'(?<!f)"(.*?\{[a-zA-Z_]\w*\}.*?)"', r'f"\1"', text)

    def mutate_fix_list_access(text: str) -> str:
        """Fix off-by-one list access [i] -> [i-1]"""
        return re.sub(r'\[([a-zA-Z_]\w*)\]', r'[\1 - 1]', text)

    mutations = [
        mutate_range_inclusive,
        mutate_remove_continue_guard,
        mutate_remove_div_100,
        mutate_fix_comparison,
        mutate_fix_loop_condition,
        mutate_add_f_prefix,
        mutate_fix_list_access,
    ]

    # Build all candidate mutations
    candidates = []
    for mutate in mutations:
        updated = mutate(original)
        if updated != original:
            candidates.append(updated)

    for i in range(len(mutations)):
        for j in range(i + 1, len(mutations)):
            first = mutations[i](original)
            if first == original:
                continue
            second = mutations[j](first)
            if second != first and second != original:
                candidates.append(second)

    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    if not unique_candidates:
        return {"success": False, "message": "No applicable mutations found"}

    def try_mutation(candidate_code: str) -> dict:
        """Test a mutation in a temporary file to allow parallel execution."""
        try:
            # Create temp file with mutation
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                dir=str(cwd),
                delete=False
            ) as tmp:
                tmp.write(candidate_code)
                tmp_path = Path(tmp.name)

            try:
                result = subprocess.run(
                    ["python", str(tmp_path)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(cwd),
                )
                if result.returncode == 0 and result.stdout.strip() == expected_output:
                    return {"success": True, "code": candidate_code}
                return {"success": False}
            finally:
                tmp_path.unlink(missing_ok=True)
        except Exception:
            return {"success": False}

    # Run mutations in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(try_mutation, c): c for c in unique_candidates}

        for future in as_completed(futures, timeout=10):
            try:
                result = future.result()
                if result.get("success"):
                    # Found a working mutation - apply it to the entry point
                    entry_point_file.write_text(result["code"])
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    return {"success": True, "message": "Applied local repair based on verification output"}
            except Exception:
                continue

    return {"success": False, "message": "Local repair did not find a matching fix"}
