#!/usr/bin/env python3
"""
remove_css_comments.py

A small utility to strip all CSS comments (/* ... */) from the target
app.css file (or any CSS file supplied on the command-line).

Usage
-----
    # use default app.css path
    python scripts/remove_css_comments.py

    # or specify a file
    python scripts/remove_css_comments.py path/to/file.css
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT_CSS_PATH = Path(
    "/Users/steven/projects/ZenBot/ZenBot.Screener.Web/Client/wwwroot/css/app.css"
)

# non-greedy to handle multiple comments; DOTALL so newlines are included
CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_comments(css: str) -> str:
    """Return *css* with all /* … */ comment blocks removed."""
    return CSS_COMMENT_RE.sub("", css)


def main() -> None:
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSS_PATH

    if not target_path.is_file():
        sys.exit(f"CSS file not found: {target_path}")

    original = target_path.read_text(encoding="utf-8")
    cleaned = strip_comments(original)

    # write atomically: write to temp then replace
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(cleaned, encoding="utf-8")
    tmp_path.replace(target_path)

    print(f"Removed comments from {target_path}")


if __name__ == "__main__":
    main()
