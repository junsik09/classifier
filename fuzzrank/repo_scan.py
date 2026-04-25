from __future__ import annotations

import re
from pathlib import Path


SOURCE_EXTENSIONS = {
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
}

DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    "__pycache__",
    "build",
    "cmake-build-debug",
    "cmake-build-release",
    "dist",
    "node_modules",
}

DEFAULT_SKIP_FILE_KEYWORDS = {
    "mock",
    "mocks",
    "test",
    "tests",
}


def parse_languages(raw: str | None) -> set[str]:
    if not raw:
        return {"c", "cpp"}
    languages = {item.strip().lower() for item in raw.split(",") if item.strip()}
    if "c++" in languages:
        languages.add("cpp")
        languages.remove("c++")
    return languages or {"c", "cpp"}


def language_for_path(path: Path) -> str | None:
    return SOURCE_EXTENSIONS.get(path.suffix.lower())


def should_skip_source_file(
    path: Path,
    skip_file_keywords: set[str] | None = None,
) -> bool:
    keywords = skip_file_keywords or DEFAULT_SKIP_FILE_KEYWORDS
    stem = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", path.stem).lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", stem) if token]

    return any(token in keywords for token in tokens)


def scan_source_files(
    repo: Path,
    languages: set[str] | None = None,
    skip_dirs: set[str] | None = None,
    skip_file_keywords: set[str] | None = None,
) -> list[Path]:
    languages = languages or {"c", "cpp"}
    skip_dirs = skip_dirs or DEFAULT_SKIP_DIRS
    skip_file_keywords = skip_file_keywords or DEFAULT_SKIP_FILE_KEYWORDS
    repo = repo.resolve()
    files: list[Path] = []

    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.relative_to(repo).parts[:-1]):
            continue
        if should_skip_source_file(path, skip_file_keywords):
            continue
        language = language_for_path(path)
        if language not in languages:
            continue
        files.append(path)

    return sorted(files)


def relative_path(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
