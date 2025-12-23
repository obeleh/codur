"""Tests for python_ast tools."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from codur.tools.python_ast import python_ast_dependencies, python_ast_dependencies_multifile

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
