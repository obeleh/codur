"""Tests for python_ast tools."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from codur.tools.python_ast import (
    python_ast_dependencies,
    python_ast_dependencies_multifile,
    python_ast_graph,
    python_ast_outline,
)

class TestPythonAstDependencies:
    """Test the dependency extraction logic."""

    def test_basic_function_call(self):
        """Test simple function calling another function."""
        code = """
def func_b():
    pass

def func_a():
    func_b()
"""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text(code)
            
            deps = python_ast_dependencies(str(p), allow_outside_root=True)
            assert "func_a -> func_b" in deps

    def test_class_method_calls(self):
        """Test method calls within a class."""
        code = """
class MyClass:
    def method_a(self):
        self.method_b()
        
    def method_b(self):
        print("hello")
"""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text(code)
            
            deps = python_ast_dependencies(str(p), allow_outside_root=True)
            assert "MyClass.method_a -> self.method_b" in deps
            assert "MyClass.method_b -> print" in deps

    def test_inheritance(self):
        """Test class inheritance dependency."""
        code = """
class Parent:
    pass

class Child(Parent):
    pass
"""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text(code)
            
            deps = python_ast_dependencies(str(p), allow_outside_root=True)
            assert "Child -> Parent" in deps

    def test_nested_attribute_calls(self):
        """Test calls to nested attributes like module.func()."""
        code = """
import os

def do_stuff():
    os.path.join("a", "b")
"""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text(code)
            
            deps = python_ast_dependencies(str(p), allow_outside_root=True)
            assert "do_stuff -> os.path.join" in deps

    def test_main_block_calls(self):
        """Test calls from top-level or main block (no parent scope)."""
        code = """
def foo():
    pass

foo()
"""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text(code)
            
            # Note: Calls from module level don't have a parent scope in current implementation
            # The current implementation checks: if self.scope_stack: ...
            # So top level calls are currently ignored by design or by current logic.
            # Let's verify what happens.
            deps = python_ast_dependencies(str(p), allow_outside_root=True)
            # Expect empty because top-level calls are ignored by `if self.scope_stack:` check
            assert deps == []

    def test_async_functions(self):
        """Test async function dependencies."""
        code = """
async def fetch():
    pass

async def main():
    await fetch()
"""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text(code)
            
            deps = python_ast_dependencies(str(p), allow_outside_root=True)
            assert "main -> fetch" in deps

    def test_imported_function_call(self):
        """Test calling a function imported from another module."""
        code = """
from math import sqrt

def calculate():
    sqrt(16)
"""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text(code)
            
            deps = python_ast_dependencies(str(p), allow_outside_root=True)
            assert "calculate -> sqrt" in deps
            assert "sqrt -> math" in deps


class TestPythonAstDependenciesMultifile:
    """Test multifile dependency extraction."""

    def test_multifile_collects_results(self):
        with TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "a.py"
            p1.write_text("def a():\n    return b()\n\ndef b():\n    return 1\n")
            p2 = Path(tmpdir) / "b.py"
            p2.write_text("def c():\n    return 2\n")

            deps = python_ast_dependencies_multifile([str(p1), str(p2)], allow_outside_root=True)
            assert str(p1) in deps
            assert str(p2) in deps
            assert "a -> b" in deps[str(p1)]


class TestPythonAstGraphOutline:
    def test_ast_graph_truncates(self):
        code = """
def a():
    return 1

def b():
    return a()
"""
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text(code)
            result = python_ast_graph(str(p), allow_outside_root=True, max_nodes=5)
            assert result["truncated"] is True
            assert result["node_count"] == 5
            assert result["edge_count"] <= result["node_count"] - 1

    def test_ast_outline_docstrings_and_decorators(self):
        code = '''
def deco(fn):
    return fn

@deco
def foo(x):
    """Doc foo."""
    def inner():
        """Inner doc."""
        return x
    return x

class Bar:
    """Bar doc."""
    @deco
    async def baz(self):
        """Baz doc."""
        return 1
'''
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text(code)
            outline = python_ast_outline(
                str(p),
                allow_outside_root=True,
                include_docstrings=True,
                include_decorators=True,
                max_results=20,
            )
            defs = {entry["qualname"]: entry for entry in outline["definitions"]}
            assert "foo" in defs
            assert defs["foo"]["docstring"] == "Doc foo."
            assert "deco" in defs["foo"]["decorators"]
            assert "foo.inner" in defs
            assert defs["foo.inner"]["type"] == "function"
            assert "Bar" in defs
            assert defs["Bar"]["type"] == "class"
            assert "Bar.baz" in defs
            assert defs["Bar.baz"]["type"] == "async_function"
            assert "deco" in defs["Bar.baz"]["decorators"]

            truncated = python_ast_outline(
                str(p),
                allow_outside_root=True,
                max_results=1,
            )
            assert truncated["truncated"] is True
            assert truncated["count"] == 1
