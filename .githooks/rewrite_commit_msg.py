import sys

msg = sys.stdin.read()

replacements = (
    (
        "Add Cursor IDE hook to deny commits with agent co-author lines.",
        "Add IDE hook to deny agent co-author lines on commits.",
    ),
    (
        "Add permanent git hooks to block Cursor co-author attribution.",
        "Add permanent git hooks to block agent co-author attribution.",
    ),
)
for old, new in replacements:
    msg = msg.replace(old, new)

blocked = (
    "Co-authored-by: Cursor",
    "cursoragent@cursor.com",
    "Made-with: Cursor",
)
lines = [
    line
    for line in msg.splitlines(keepends=True)
    if not any(marker in line for marker in blocked)
]
sys.stdout.write("".join(lines))
