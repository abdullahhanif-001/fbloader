from __future__ import annotations

import os

import pytest

from fbloader._spawn import mark_main, require_main


def test_require_main_raises_without_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("FBLOADER_SKIP_SPAWN_GUARD", raising=False)
    monkeypatch.setattr("fbloader._spawn.is_spawn_platform", lambda: True)
    monkeypatch.setattr("fbloader._spawn.mp.current_process", lambda: type("P", (), {"name": "MainProcess"})())
    import sys

    main = sys.modules["__main__"]
    monkeypatch.setattr(main, "__file__", str(Path_main_file()), raising=False)
    if hasattr(main, "__fbloader_main_ok__"):
        monkeypatch.delattr(main, "__fbloader_main_ok__", raising=False)
    with pytest.raises(RuntimeError, match="mark_main"):
        require_main(num_workers=2)


def Path_main_file() -> str:
    return os.path.join(os.getcwd(), "train_app.py")


def test_mark_main_allows_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr("fbloader._spawn.is_spawn_platform", lambda: True)
    monkeypatch.setattr("fbloader._spawn.mp.current_process", lambda: type("P", (), {"name": "MainProcess"})())
    import sys

    main = sys.modules["__main__"]
    monkeypatch.setattr(main, "__file__", Path_main_file(), raising=False)
    mark_main()
    require_main(num_workers=2)
