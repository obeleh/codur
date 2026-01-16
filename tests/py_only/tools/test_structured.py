import json
import pytest
from codur.tools.structured import (
    read_json, json_decode, write_json, set_json_value,
    read_yaml, yaml_decode, write_yaml, set_yaml_value,
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
    assert json_decode(json.dumps(data)) == data

def test_yaml_ops(temp_fs):
    data = {"key": "value", "list": [1, 2, 3]}
    write_yaml("test.yaml", data, root=temp_fs)
    
    loaded = read_yaml("test.yaml", root=temp_fs)
    assert loaded == data
    
    set_yaml_value("test.yaml", "key", "updated", root=temp_fs)
    loaded = read_yaml("test.yaml", root=temp_fs)
    assert loaded["key"] == "updated"
    assert yaml_decode("key: value\nlist:\n  - 1\n  - 2") == {"key": "value", "list": [1, 2]}

def test_ini_ops(temp_fs):
    data = {"Section": {"key": "value"}}
    write_ini("test.ini", data, root=temp_fs)
    
    loaded = read_ini("test.ini", root=temp_fs)
    assert loaded["Section"]["key"] == "value"
    
    set_ini_value("test.ini", "Section", "key", "updated", root=temp_fs)
    loaded = read_ini("test.ini", root=temp_fs)
    assert loaded["Section"]["key"] == "updated"


def test_set_json_value_rejects_non_dict_container(temp_fs):
    data = {"key": "value"}
    write_json("test.json", data, root=temp_fs)

    with pytest.raises(ValueError, match="Cannot set nested value on non-dict container"):
        set_json_value("test.json", "key.sub", "nope", root=temp_fs)


def test_read_yaml_empty_file_returns_none(temp_fs):
    (temp_fs / "empty.yaml").write_text("", encoding="utf-8")
    assert read_yaml("empty.yaml", root=temp_fs) is None


def test_set_ini_value_creates_section(temp_fs):
    (temp_fs / "new.ini").write_text("", encoding="utf-8")
    set_ini_value("new.ini", "NewSection", "flag", "true", root=temp_fs)

    loaded = read_ini("new.ini", root=temp_fs)
    assert loaded["NewSection"]["flag"] == "true"


# Path resolution tests for structured tools
def test_read_json_rejects_outside_root(temp_fs):
    """Test that read_json rejects paths outside the workspace root."""
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        read_json("../outside.json", root=temp_fs)


def test_write_json_rejects_outside_root(temp_fs):
    """Test that write_json rejects paths outside the workspace root."""
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        write_json("../outside.json", {"key": "value"}, root=temp_fs)


def test_set_json_value_rejects_outside_root(temp_fs):
    """Test that set_json_value rejects paths outside the workspace root."""
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        set_json_value("../outside.json", "key", "value", root=temp_fs)


def test_read_yaml_rejects_outside_root(temp_fs):
    """Test that read_yaml rejects paths outside the workspace root."""
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        read_yaml("../outside.yaml", root=temp_fs)


def test_write_yaml_rejects_outside_root(temp_fs):
    """Test that write_yaml rejects paths outside the workspace root."""
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        write_yaml("../outside.yaml", {"key": "value"}, root=temp_fs)


def test_set_yaml_value_rejects_outside_root(temp_fs):
    """Test that set_yaml_value rejects paths outside the workspace root."""
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        set_yaml_value("../outside.yaml", "key", "value", root=temp_fs)


def test_read_ini_rejects_outside_root(temp_fs):
    """Test that read_ini rejects paths outside the workspace root."""
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        read_ini("../outside.ini", root=temp_fs)


def test_write_ini_rejects_outside_root(temp_fs):
    """Test that write_ini rejects paths outside the workspace root."""
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        write_ini("../outside.ini", {}, root=temp_fs)


def test_set_ini_value_rejects_outside_root(temp_fs):
    """Test that set_ini_value rejects paths outside the workspace root."""
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        set_ini_value("../outside.ini", "Section", "key", "value", root=temp_fs)
