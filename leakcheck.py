#!/usr/bin/env python3
"""leakcheck -- grep your own agent session logs for secrets that leaked into them.

Read-only. Nothing leaves your machine. Scans, by default:

  1. this repo's traces/ directory (the agent's own run records)
  2. ~/.claude/projects/**/*.jsonl (Claude Code session transcripts), if present

Add any other directory with --dir. Prints file, line number, which pattern
matched, and a redacted preview. Exit 1 if anything was found, 0 if clean.

Deliberately NOT scanned: .env and .env.* files -- your key legitimately lives
there. The question this tool answers is whether a key ended up somewhere it
shouldn't: a transcript, a trace, an answer an agent wrote down.

Patterns are a starting set, not a guarantee. Add your own below.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PATTERNS = [
    ("openai/anthropic-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    ("this repo's planted demo secret", re.compile(r"\bnw_live_sk_[A-Za-z0-9]+")),
]

SKIP_NAMES = {".env", ".env.example", "leakcheck.py"}
SCAN_SUFFIXES = {".jsonl", ".json", ".log", ".txt", ".md"}


def redact(match: str) -> str:
    return match[:8] + "…" + "*" * 6 if len(match) > 8 else "********"


def scan_file(path: Path):
    hits = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return hits
    for lineno, line in enumerate(text.splitlines(), 1):
        for label, rx in PATTERNS:
            for m in rx.finditer(line):
                hits.append((path, lineno, label, redact(m.group(0))))
    return hits


def scan_dir(root: Path):
    hits = []
    if not root.is_dir():
        return hits
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        if path.name.startswith(".env"):
            continue
        if path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        hits.extend(scan_file(path))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", action="append", default=[], metavar="PATH",
                    help="extra directory to scan (repeatable)")
    ap.add_argument("--no-defaults", action="store_true",
                    help="scan only --dir paths, skip traces/ and ~/.claude")
    args = ap.parse_args()

    roots = [Path(d).expanduser() for d in args.dir]
    if not args.no_defaults:
        roots.append(Path(__file__).resolve().parent / "traces")
        roots.append(Path.home() / ".claude" / "projects")

    all_hits = []
    for root in roots:
        all_hits.extend(scan_dir(root))

    if not all_hits:
        print("clean: no key-shaped strings found in", ", ".join(str(r) for r in roots))
        return 0

    print(f"FOUND {len(all_hits)} key-shaped string(s) in your logs/traces:\n")
    for path, lineno, label, preview in all_hits:
        print(f"  {path}:{lineno}  [{label}]  {preview}")
    print("\nThese are places a secret ended up OUTSIDE your config. Rotate anything real.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
