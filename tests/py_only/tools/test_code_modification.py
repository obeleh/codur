"""Tests for high-level code modification tools."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from codur.tools.code_modification import (
    replace_function,
    replace_class,
    replace_method,
    replace_file_content,
    inject_function,
)

class TestCodeModificationTools:
    
    def test_replace_function(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.py"
            content = """
def keep_me():
    pass

def target_func(a, b):
    print("old")
    return a + b

def also_keep():
    pass
"""
            file_path.write_text(content.strip())
            
            new_code = """def target_func(a, b):
    print("new implementation")
    return a * b"""
            
            # Change cwd so relative paths work if needed, or pass absolute
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = replace_function(str(file_path), "target_func", new_code)
                assert "Successfully replaced function" in result
                
                new_content = file_path.read_text()
                assert 'print("new implementation")' in new_content
                assert 'print("old")' not in new_content
                assert 'def keep_me():' in new_content
                assert 'def also_keep():' in new_content
            finally:
                os.chdir(old_cwd)

    def test_replace_class(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_cls.py"
            content = """
class OldClass:
    pass

class TargetClass:
    def __init__(self):
        self.x = 1

    def method(self):
        pass
"""
            file_path.write_text(content.strip())
            
            new_code = """class TargetClass:
    def __init__(self):
        self.x = 2"""
            
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = replace_class(str(file_path), "TargetClass", new_code)
                assert "Successfully replaced class" in result
                
                new_content = file_path.read_text()
                assert 'self.x = 2' in new_content
                assert 'self.x = 1' not in new_content
                assert 'class OldClass:' in new_content
            finally:
                os.chdir(old_cwd)

    def test_replace_method(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_meth.py"
            content = """
class MyClass:
    def keep_method(self):
        pass

    def target_method(self, arg):
        return arg

    def another_keep(self):
        pass
"""
            file_path.write_text(content.strip())
            
            new_code = """    def target_method(self, arg):
        return arg * 2"""
            
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = replace_method(str(file_path), "MyClass", "target_method", new_code)
                assert "Successfully replaced method" in result
                
                new_content = file_path.read_text()
                assert 'return arg * 2' in new_content
                assert 'return arg' in new_content  # substring match, need to be careful
                # Use strict check
                lines = new_content.splitlines()
                assert '        return arg * 2' in lines
            finally:
                os.chdir(old_cwd)

    def test_replace_file_content(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_full.py"
            file_path.write_text("old content")
            
            new_code = "x = 1"
            
            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = replace_file_content(str(file_path), new_code)
                assert "Successfully replaced file content" in result
                assert file_path.read_text() == "x = 1"
            finally:
                os.chdir(old_cwd)

    def test_replace_class_not_found(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_cls.py"
            file_path.write_text("class Existing:\n    pass\n")

            new_code = "class Missing:\n    pass"

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = replace_class(str(file_path), "Missing", new_code)
                assert "Could not find class 'Missing'" in result
            finally:
                os.chdir(old_cwd)

    def test_replace_method_not_found(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_meth.py"
            file_path.write_text("class Example:\n    def keep(self):\n        pass\n")

            new_code = "    def missing(self):\n        return 1"

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = replace_method(str(file_path), "Example", "missing", new_code)
                assert "Could not find method 'Example.missing'" in result
            finally:
                os.chdir(old_cwd)

    def test_replace_file_content_invalid_python_does_not_write(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "bad.py"
            file_path.write_text("print('ok')\n")
            invalid_code = "def bad(:\n    pass"

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = replace_file_content(str(file_path), invalid_code)
                assert "Invalid Python syntax" in result
                assert file_path.read_text() == "print('ok')\n"
            finally:
                os.chdir(old_cwd)

    def test_inject_function_inserts_before_main_guard(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "main.py"
            file_path.write_text(
                "def existing():\n    return 1\n\n"
                "if __name__ == \"__main__\":\n    print(existing())\n"
            )
            new_code = "def injected():\n    return 2"

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = inject_function(str(file_path), new_code)
                assert "Successfully injected function 'injected'" in result
                content = file_path.read_text()
                assert content.index("def injected") < content.index("if __name__ == \"__main__\":")
            finally:
                os.chdir(old_cwd)

    def test_inject_function_rejects_existing_name(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "main.py"
            file_path.write_text("def existing():\n    return 1\n")
            new_code = "def existing():\n    return 2"

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = inject_function(str(file_path), new_code)
                assert "already exists" in result
            finally:
                os.chdir(old_cwd)

    def test_inject_function_invalid_syntax(self):
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "main.py"
            file_path.write_text("print('ok')\n")
            invalid_code = "def bad(:\n    pass"

            import os
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = inject_function(str(file_path), invalid_code)
                assert "Invalid Python syntax" in result
            finally:
                os.chdir(old_cwd)
