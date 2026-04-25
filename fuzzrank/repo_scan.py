from __future__ import annotations

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


def scan_source_files(
    repo: Path,
    languages: set[str] | None = None,
    skip_dirs: set[str] | None = None,
) -> list[Path]:
    languages = languages or {"c", "cpp"}
    skip_dirs = skip_dirs or DEFAULT_SKIP_DIRS
    repo = repo.resolve()
    files: list[Path] = []

    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.relative_to(repo).parts[:-1]):
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
