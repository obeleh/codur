import json
import pytest
from codur.tools.structured import (
    read_json, write_json, set_json_value,
    read_yaml, write_yaml, set_yaml_value,
    read_ini, write_ini, set_ini_value
)

@pytest.fixture
def temp_fs(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    return root

def test_json_ops(temp_fs):
    data = {"key": "value", "nested": {"foo": "bar"}}
    write_json("test.json", data, root=temp_fs)
    
    loaded = read_json("test.json", root=temp_fs)
    assert loaded == data
    
    set_json_value("test.json", "nested.foo", "baz", root=temp_fs)
    loaded = read_json("test.json", root=temp_fs)
    assert loaded["nested"]["foo"] == "baz"

def test_yaml_ops(temp_fs):
    data = {"key": "value", "list": [1, 2, 3]}
    write_yaml("test.yaml", data, root=temp_fs)
    
    loaded = read_yaml("test.yaml", root=temp_fs)
    assert loaded == data
    
    set_yaml_value("test.yaml", "key", "updated", root=temp_fs)
    loaded = read_yaml("test.yaml", root=temp_fs)
    assert loaded["key"] == "updated"

def test_ini_ops(temp_fs):
    data = {"Section": {"key": "value"}}
    write_ini("test.ini", data, root=temp_fs)
    
    loaded = read_ini("test.ini", root=temp_fs)
    assert loaded["Section"]["key"] == "value"
    
    set_ini_value("test.ini", "Section", "key", "updated", root=temp_fs)
    loaded = read_ini("test.ini", root=temp_fs)
    assert loaded["Section"]["key"] == "updated"
