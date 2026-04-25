from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .model import FunctionRecord, Param, stable_function_id
from .repo_scan import language_for_path, relative_path


CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "sizeof",
    "alignof",
    "decltype",
    "new",
    "delete",
}

CALL_EXCLUDE = CONTROL_KEYWORDS | {
    "static_cast",
    "reinterpret_cast",
    "const_cast",
    "dynamic_cast",
}


def safe_text(source: bytes, node) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def extract_repo_functions(repo: Path, files: Iterable[Path] | None = None) -> list[FunctionRecord]:
    if files is None:
        from .repo_scan import scan_source_files

        files = scan_source_files(repo)

    records: list[FunctionRecord] = []
    for path in files:
        language = language_for_path(path) or "unknown"
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = relative_path(path, repo)
        records.extend(extract_file_functions(text, rel, language))
    return records


def extract_file_functions(text: str, file: str, language: str) -> list[FunctionRecord]:
    tree_sitter_records = _extract_with_tree_sitter(text, file, language)
    if tree_sitter_records:
        return tree_sitter_records
    return _extract_with_fallback(text, file, language)


def _extract_with_tree_sitter(text: str, file: str, language: str) -> list[FunctionRecord]:
    try:
        from tree_sitter_languages import get_parser  # type: ignore
    except Exception:
        return []

    parser_language = "cpp" if language == "cpp" else "c"
    try:
        parser = get_parser(parser_language)
        source = text.encode("utf-8")
        tree = parser.parse(source)
    except Exception:
        return []

    records: list[FunctionRecord] = []

    def walk(node) -> None:
        if node.type == "function_definition":
            record = _record_from_ts_node(source, text, file, language, node)
            if record:
                records.append(record)
            return
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return records


def _record_from_ts_node(
    source: bytes,
    text: str,
    file: str,
    language: str,
    node,
) -> FunctionRecord | None:
    body = None
    for child in node.children:
        if child.type == "compound_statement":
            body = child
            break

    signature_end = body.start_byte if body else node.end_byte
    signature = source[node.start_byte : signature_end].decode("utf-8", errors="replace").strip()
    parsed = parse_signature(signature)
    if not parsed:
        return None

    name, return_type, params, scope, is_method = parsed
    line_start = node.start_point[0] + 1
    line_end = node.end_point[0] + 1
    body_text = safe_text(source, body) if body else ""
    calls = extract_calls(body_text)

    return FunctionRecord(
        id=stable_function_id(file, name, line_start),
        name=name,
        file=file,
        line_start=line_start,
        line_end=line_end,
        language=language,
        signature=one_line(signature),
        return_type=return_type,
        params=params,
        scope=scope,
        is_static=_looks_static(signature),
        is_method=is_method,
        body_excerpt=body_text[:3000] if body_text else None,
        calls=calls,
        source=["tree-sitter"],
    )


def _extract_with_fallback(text: str, file: str, language: str) -> list[FunctionRecord]:
    masked = mask_comments_and_strings(text)
    records: list[FunctionRecord] = []
    seen: set[tuple[str, int]] = set()

    for open_brace in _iter_candidate_open_braces(masked):
        start = _signature_start(masked, open_brace)
        signature_start, signature = _trim_signature_region(text, start, open_brace)
        _, masked_signature = _trim_signature_region(masked, start, open_brace)

        if not _looks_like_function_signature(masked_signature):
            continue

        parsed = parse_signature(signature)
        if not parsed:
            continue

        name, return_type, params, scope, is_method = parsed
        line_start = text.count("\n", 0, signature_start) + 1
        key = (name, line_start)
        if key in seen:
            continue
        seen.add(key)

        close_brace = find_matching_brace(masked, open_brace)
        line_end = text.count("\n", 0, close_brace) + 1 if close_brace is not None else None
        body_text = text[open_brace : close_brace + 1] if close_brace is not None else ""
        calls = extract_calls(body_text)
        parse_errors = [] if close_brace is not None else ["could not find matching function body brace"]

        records.append(
            FunctionRecord(
                id=stable_function_id(file, name, line_start),
                name=name,
                file=file,
                line_start=line_start,
                line_end=line_end,
                language=language,
                signature=one_line(signature),
                return_type=return_type,
                params=params,
                scope=scope,
                is_static=_looks_static(signature),
                is_method=is_method,
                body_excerpt=body_text[:3000] if body_text else None,
                calls=calls,
                source=["fallback"],
                parse_errors=parse_errors,
            )
        )

    return records


def mask_comments_and_strings(text: str) -> str:
    chars = list(text)
    i = 0
    n = len(chars)
    while i < n:
        c = chars[i]
        nxt = chars[i + 1] if i + 1 < n else ""

        if c == "/" and nxt == "/":
            chars[i] = chars[i + 1] = " "
            i += 2
            while i < n and chars[i] != "\n":
                chars[i] = " "
                i += 1
            continue

        if c == "/" and nxt == "*":
            chars[i] = chars[i + 1] = " "
            i += 2
            while i + 1 < n and not (chars[i] == "*" and chars[i + 1] == "/"):
                if chars[i] != "\n":
                    chars[i] = " "
                i += 1
            if i + 1 < n:
                chars[i] = chars[i + 1] = " "
                i += 2
            continue

        if c in {"'", '"'}:
            quote = c
            chars[i] = " "
            i += 1
            escaped = False
            while i < n:
                current = chars[i]
                if current != "\n":
                    chars[i] = " "
                if current == quote and not escaped:
                    i += 1
                    break
                escaped = current == "\\" and not escaped
                if current != "\\":
                    escaped = False
                i += 1
            continue

        i += 1

    return "".join(chars)


def _iter_candidate_open_braces(masked: str) -> Iterable[int]:
    for index, char in enumerate(masked):
        if char == "{":
            yield index


def _signature_start(masked: str, open_brace: int) -> int:
    boundary = max(
        masked.rfind(";", 0, open_brace),
        masked.rfind("}", 0, open_brace),
        masked.rfind("{", 0, open_brace),
    )
    return boundary + 1


def _trim_signature_region(text: str, start: int, end: int) -> tuple[int, str]:
    region = text[start:end]
    offset = 0
    lines = region.splitlines(keepends=True)
    while lines:
        stripped = lines[0].strip()
        if stripped == "" or stripped.startswith("#"):
            offset += len(lines.pop(0))
            continue
        break
    return start + offset, "".join(lines).strip()


def _looks_like_function_signature(signature: str) -> bool:
    signature = _clean_signature_prefix(signature)
    if not signature or "(" not in signature or ")" not in signature:
        return False

    first_word = re.match(r"([A-Za-z_]\w*)", signature)
    if first_word and first_word.group(1) in {
        "class",
        "struct",
        "union",
        "enum",
        "namespace",
        *CONTROL_KEYWORDS,
    }:
        return False

    name_match = _function_name_match(signature)
    if not name_match:
        return False

    short_name = name_match.group("name").split("::")[-1]
    return short_name not in CONTROL_KEYWORDS


def parse_signature(signature: str) -> tuple[str, str | None, list[Param], str | None, bool] | None:
    cleaned = _clean_signature_prefix(signature)
    match = _function_name_match(cleaned)
    if not match:
        return None

    name = match.group("name").replace(" ", "")
    open_paren = match.end() - 1
    close_paren = find_matching_paren(cleaned, open_paren)
    if close_paren is None:
        return None

    params_text = cleaned[open_paren + 1 : close_paren]
    before_name = cleaned[: match.start("name")].strip()
    return_type = before_name or None
    scope = None
    is_method = "::" in name
    if "::" in name:
        scope, name = name.rsplit("::", 1)

    params = parse_params(params_text)
    return name, return_type, params, scope, is_method


def _clean_signature_prefix(signature: str) -> str:
    lines = [
        line.strip()
        for line in signature.strip().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    while lines and lines[0] in {"public:", "private:", "protected:"}:
        lines.pop(0)
    cleaned = " ".join(lines)
    cleaned = re.sub(r"\b(public|private|protected)\s*:\s*", "", cleaned)
    cleaned = re.sub(r"\[\[[^\]]+\]\]\s*", "", cleaned)
    cleaned = re.sub(r"\bextern\s+\"C\"\s*", "", cleaned)
    return cleaned.strip()


def _function_name_match(signature: str) -> re.Match[str] | None:
    candidates = list(
        re.finditer(
            r"(?P<name>(?:~?[A-Za-z_]\w*::)*~?[A-Za-z_]\w*|operator\s*[^\s(]+)\s*\(",
            signature,
        )
    )
    if not candidates:
        return None
    return candidates[-1]


def find_matching_paren(text: str, open_paren: int) -> int | None:
    depth = 0
    for i in range(open_paren, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def find_matching_brace(masked: str, open_brace: int) -> int | None:
    depth = 0
    for i in range(open_brace, len(masked)):
        if masked[i] == "{":
            depth += 1
        elif masked[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def parse_params(params_text: str) -> list[Param]:
    params: list[Param] = []
    for raw in split_top_level(params_text):
        raw = raw.strip()
        if not raw or raw == "void":
            continue
        params.append(parse_param(raw))
    return params


def split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    paren = bracket = angle = brace = 0
    for i, char in enumerate(text):
        if char == "(":
            paren += 1
        elif char == ")":
            paren = max(0, paren - 1)
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket = max(0, bracket - 1)
        elif char == "<":
            angle += 1
        elif char == ">":
            angle = max(0, angle - 1)
        elif char == "{":
            brace += 1
        elif char == "}":
            brace = max(0, brace - 1)
        elif char == sep and paren == bracket == angle == brace == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def parse_param(raw: str) -> Param:
    without_default = split_top_level(raw, "=")[0].strip()

    function_pointer = re.search(r"\(\s*\*\s*(?P<name>[A-Za-z_]\w*)\s*\)", without_default)
    if function_pointer:
        return Param(name=function_pointer.group("name"), type=without_default, raw=raw)

    array_match = re.search(r"(?P<name>[A-Za-z_]\w*)\s*(?:\[[^\]]*\])+\s*$", without_default)
    if array_match:
        name = array_match.group("name")
        return Param(name=name, type=without_default[: array_match.start("name")].strip(), raw=raw)

    identifiers = list(re.finditer(r"\b[A-Za-z_]\w*\b", without_default))
    if len(identifiers) >= 2:
        name = identifiers[-1].group(0)
        type_text = without_default[: identifiers[-1].start()].rstrip()
        if type_text:
            return Param(name=name, type=type_text, raw=raw)

    return Param(name=None, type=without_default, raw=raw)


def extract_calls(body_text: str) -> list[str]:
    masked = mask_comments_and_strings(body_text)
    calls: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\(", masked):
        name = match.group(1)
        short = name.rsplit("::", 1)[-1]
        if short in CALL_EXCLUDE:
            continue
        if name not in seen:
            calls.append(name)
            seen.add(name)
    return calls


def one_line(text: str) -> str:
    return " ".join(text.strip().split())


def _looks_static(signature: str) -> bool:
    return bool(re.search(r"(^|\s)static(\s|$)", signature))
