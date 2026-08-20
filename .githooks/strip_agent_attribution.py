"""Strip blocked agent co-author lines from commit messages."""
from __future__ import annotations

import sys
from pathlib import Path

BLOCKED = (
    "Co-authored-by: Cursor",
    "cursoragent@cursor.com",
    "Made-with: Cursor",
)


def strip_file(path: Path) -> None:
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    cleaned = [ln for ln in lines if not any(b in ln for b in BLOCKED)]
    path.write_text("".join(cleaned), encoding="utf-8")


def contains_blocked(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    for marker in BLOCKED:
        if marker in text:
            return marker
    return None


def main() -> int:
    if len(sys.argv) < 3:
        return 0
    hook = sys.argv[1]
    msg_file = Path(sys.argv[2])

    if hook == "prepare-commit-msg":
        strip_file(msg_file)
        return 0

    if hook == "commit-msg":
        hit = contains_blocked(msg_file)
        if hit:
            print(
                "Commit blocked: remove agent co-author attribution. "
                "Only Abdullah Hanif should appear on commits.",
                file=sys.stderr,
            )
            return 1
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
