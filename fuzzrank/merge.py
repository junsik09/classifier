from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .model import FunctionRecord, Param, stable_function_id


def merge_ctags_functions(
    ctags_records: Iterable[dict],
    functions: list[FunctionRecord],
    repo: Path,
) -> list[FunctionRecord]:
    merged = list(functions)
    matched_ctags: set[int] = set()

    for index, ctag in enumerate(ctags_records):
        match = _find_matching_function(ctag, merged)
        if not match:
            continue
        matched_ctags.add(index)
        if "ctags" not in match.source:
            match.source.append("ctags")
        if not match.signature and ctag.get("signature"):
            match.signature = f"{ctag.get('name')}{ctag.get('signature')}"
        if not match.scope:
            match.scope = ctag.get("scope") or ctag.get("class") or ctag.get("namespace")
        if match.is_static is None and ctag.get("access"):
            match.is_static = ctag.get("access") == "file"

    for index, ctag in enumerate(ctags_records):
        if index in matched_ctags:
            continue
        record = _function_from_ctag(ctag, repo)
        if record:
            merged.append(record)

    return sorted(merged, key=lambda fn: (fn.file, fn.line_start, fn.name))


def _find_matching_function(ctag: dict, functions: list[FunctionRecord]) -> FunctionRecord | None:
    ctag_path = Path(ctag.get("path") or "")
    ctag_name = ctag.get("name")
    try:
        ctag_line = int(ctag.get("line", -1))
    except (TypeError, ValueError):
        ctag_line = -1

    for fn in functions:
        if Path(fn.file).name != ctag_path.name:
            continue
        if fn.name != ctag_name:
            continue
        if ctag_line > 0 and abs(ctag_line - fn.line_start) <= 5:
            return fn

    return None


def _function_from_ctag(ctag: dict, repo: Path) -> FunctionRecord | None:
    name = ctag.get("name")
    path = ctag.get("path")
    if not name or not path:
        return None

    try:
        line = int(ctag.get("line") or 1)
    except (TypeError, ValueError):
        line = 1

    file = _relative_ctag_path(path, repo)
    signature = None
    if ctag.get("signature"):
        signature = f"{name}{ctag.get('signature')}"

    return FunctionRecord(
        id=stable_function_id(file, name, line),
        name=name,
        file=file,
        line_start=line,
        line_end=None,
        language=_language_from_path(file),
        signature=signature,
        return_type=ctag.get("typeref") or None,
        params=[],
        scope=ctag.get("scope") or ctag.get("class") or ctag.get("namespace"),
        is_static=ctag.get("access") == "file" if ctag.get("access") else None,
        is_method=ctag.get("kind") == "method",
        source=["ctags"],
    )


def _relative_ctag_path(path: str, repo: Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(repo.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def _language_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"}:
        return "cpp"
    if suffix in {".c", ".h"}:
        return "c"
    return "unknown"
