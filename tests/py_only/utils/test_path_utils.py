import pytest

from codur.utils.path_utils import resolve_path, resolve_root


def test_resolve_root_defaults_to_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_root(None) == tmp_path.resolve()


def test_resolve_path_relative(tmp_path) -> None:
    result = resolve_path("dir/file.txt", tmp_path)
    assert result == (tmp_path / "dir" / "file.txt").resolve()


def test_resolve_path_absolute_inside_root(tmp_path) -> None:
    target = tmp_path / "file.txt"
    result = resolve_path(str(target), tmp_path)
    assert result == target.resolve()


def test_resolve_path_rejects_outside_root(tmp_path) -> None:
    outside = tmp_path.parent / "outside.txt"
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        resolve_path(str(outside), tmp_path)


def test_resolve_path_allows_outside_root(tmp_path) -> None:
    outside = tmp_path.parent / "outside.txt"
    result = resolve_path(str(outside), tmp_path, allow_outside_root=True)
    assert result == outside.resolve()


def test_resolve_path_rejects_parent_traversal(tmp_path) -> None:
    with pytest.raises(ValueError, match="Path escapes workspace root"):
        resolve_path("../outside.txt", tmp_path)
