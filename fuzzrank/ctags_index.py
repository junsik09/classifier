from __future__ import annotations

import json
import subprocess
from pathlib import Path


CTAGS_LANGUAGE_MAP = {
    "c": "C",
    "cpp": "C++",
}


def run_ctags(repo: Path, languages: set[str] | None = None) -> tuple[list[dict], list[str]]:
    selected = languages or {"c", "cpp"}
    ctags_languages = ",".join(
        CTAGS_LANGUAGE_MAP[lang] for lang in sorted(selected) if lang in CTAGS_LANGUAGE_MAP
    )
    warnings: list[str] = []
    cmd = [
        "ctags",
        "-R",
        f"--languages={ctags_languages}",
        "--fields=+n+S+K+Z+t+a",
        "--extras=+q",
        "--output-format=json",
        "-f",
        "-",
        str(repo),
    ]

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return [], ["ctags executable not found; symbols.jsonl will be empty"]

    if proc.returncode != 0:
        warnings.append(proc.stderr.strip() or f"ctags exited with status {proc.returncode}")

    records: list[dict] = []
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if obj.get("_type") != "tag":
            continue

        if obj.get("kind") in {"function", "prototype", "method"}:
            records.append(obj)

    return records, warnings
