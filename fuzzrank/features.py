from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .model import FeatureRecord, FunctionRecord, Param


DEFAULT_RULES: dict[str, Any] = {
    "name_keywords": {
        "parse": 5,
        "decode": 5,
        "deserialize": 5,
        "unmarshal": 5,
        "read": 4,
        "load": 4,
        "validate": 4,
        "check": 3,
        "scan": 3,
        "lex": 3,
        "tokenize": 3,
        "decompress": 4,
        "inflate": 4,
        "unpack": 4,
    },
    "path_keywords": {
        "third_party": -2,
        "vendor": -2,
        "external": -2,
        "test": -2,
        "tests": -2,
        "example": -2,
        "examples": -2,
        "generated": -3,
    },
    "good_types": {
        "byte_pointer": {
            "patterns": ["uint8_t *", "unsigned char *", "std::byte *", "byte *"],
            "score": 5,
        },
        "string_like": {
            "patterns": [
                "char *",
                "std::string",
                "std::string_view",
                "llvm::StringRef",
                "absl::string_view",
                "folly::StringPiece",
            ],
            "score": 4,
        },
    },
    "bad_types": {
        "context_like": {
            "patterns": ["Context", "Ctx", "Session", "Manager", "Engine"],
            "penalty": -4,
        },
        "io_like": {
            "patterns": ["FILE *", "FILE*", "fstream", "ifstream", "Socket"],
            "penalty": -3,
        },
    },
}

LENGTH_NAMES = {
    "len",
    "length",
    "size",
    "sz",
    "n",
    "count",
    "buflen",
    "buf_len",
    "data_len",
    "data_size",
}

MEMORY_SENSITIVE_CALLS = {
    "memcpy",
    "memmove",
    "memcmp",
    "memset",
    "strlen",
    "strnlen",
    "strcpy",
    "strncpy",
    "strcat",
    "strncat",
    "sprintf",
    "snprintf",
    "sscanf",
}

PARSE_DECODE_CALL_KEYWORDS = {
    "parse",
    "decode",
    "deserialize",
    "unmarshal",
    "inflate",
    "decompress",
    "unpack",
    "scan",
    "lex",
    "token",
}

PRIMITIVE_TYPE_WORDS = {
    "void",
    "char",
    "short",
    "int",
    "long",
    "float",
    "double",
    "signed",
    "unsigned",
    "bool",
    "_Bool",
    "size_t",
    "ssize_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "std::string",
    "std::string_view",
}


def load_rules(path: Path | None = None) -> dict[str, Any]:
    if not path:
        return DEFAULT_RULES

    try:
        import yaml  # type: ignore
    except Exception:
        return DEFAULT_RULES

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    return deep_merge(DEFAULT_RULES, loaded)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            result[key] = deep_merge(value, override.get(key, {}))
        else:
            result[key] = value
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def extract_features(fn: FunctionRecord, rules: dict[str, Any] | None = None) -> FeatureRecord:
    rules = rules or DEFAULT_RULES
    normalized_params = [(param, normalize_type(param.type)) for param in fn.params]
    lower_name = fn.name.lower()
    lower_path = fn.file.lower()

    name_keywords = [
        keyword for keyword in rules.get("name_keywords", {}) if keyword.lower() in lower_name
    ]
    path_keywords = [
        keyword for keyword in rules.get("path_keywords", {}) if keyword.lower() in lower_path
    ]

    byte_patterns = _patterns(rules, "good_types", "byte_pointer")
    string_patterns = _patterns(rules, "good_types", "string_like")
    context_patterns = _patterns(rules, "bad_types", "context_like")
    io_patterns = _patterns(rules, "bad_types", "io_like")

    has_byte_pointer = any(_contains_pattern(t, byte_patterns) for _, t in normalized_params)
    has_char_pointer = any("char *" in t or "char*" in t for _, t in normalized_params)
    has_void_pointer = any("void *" in t or "void*" in t for _, t in normalized_params)
    has_length_param = any(is_length_param(param.name, t) for param, t in normalized_params)
    has_pointer_length_pair, _ = detect_pointer_length_pair(fn.params)
    has_string_like_param = any(_contains_pattern(t, string_patterns) for _, t in normalized_params)
    has_context_like_param = any(
        _contains_pattern(param.raw, context_patterns)
        or _contains_pattern(t, context_patterns)
        or (param.name or "").lower() in {"ctx", "context", "session", "manager", "engine"}
        for param, t in normalized_params
    )
    has_file_or_fd_param = any(
        _contains_pattern(param.raw, io_patterns)
        or _contains_pattern(t, io_patterns)
        or (param.name or "").lower() in {"file", "fp", "fd", "path", "filename"}
        for param, t in normalized_params
    )
    has_callback_param = any(is_callback_param(param) for param in fn.params)
    returns_status_like = is_status_like_return(fn.return_type)
    calls_memory_sensitive = any(call.rsplit("::", 1)[-1] in MEMORY_SENSITIVE_CALLS for call in fn.calls)
    calls_parse_decode = any(
        any(keyword in call.lower() for keyword in PARSE_DECODE_CALL_KEYWORDS) for call in fn.calls
    )
    has_custom_type_param = any(is_custom_type_param(param) for param in fn.params)

    return FeatureRecord(
        function_id=fn.id,
        name_keywords=name_keywords,
        path_keywords=path_keywords,
        has_byte_pointer=has_byte_pointer,
        has_char_pointer=has_char_pointer,
        has_void_pointer=has_void_pointer,
        has_length_param=has_length_param,
        has_pointer_length_pair=has_pointer_length_pair,
        has_string_like_param=has_string_like_param,
        has_custom_type_param=has_custom_type_param,
        has_context_like_param=has_context_like_param,
        has_file_or_fd_param=has_file_or_fd_param,
        has_callback_param=has_callback_param,
        returns_status_like=returns_status_like,
        calls_memory_sensitive=calls_memory_sensitive,
        calls_parse_decode=calls_parse_decode,
    )


def normalize_type(t: str) -> str:
    t = t.strip()
    t = re.sub(r"\b(const|volatile|restrict|__restrict|__restrict__)\b", "", t)
    t = t.replace(" *", " *").replace("*", " *")
    t = t.replace(" &", " &").replace("&", " &")
    return " ".join(t.split())


def is_pointer_type(t: str) -> bool:
    lower = t.lower()
    return "*" in t or "span" in lower or "string_view" in lower or "array_view" in lower


def is_length_param(name: str | None, t: str) -> bool:
    if name:
        lname = name.lower()
        if lname in LENGTH_NAMES:
            return True
        if any(x in lname for x in ["len", "size", "count", "nbytes"]):
            return True

    nt = normalize_type(t)
    return nt in {
        "size_t",
        "ssize_t",
        "int",
        "unsigned",
        "unsigned int",
        "uint32_t",
        "uint64_t",
    }


def detect_pointer_length_pair(params: list[Param]) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    for i, p in enumerate(params):
        if not is_pointer_type(normalize_type(p.type)):
            continue

        pname = (p.name or "").lower()

        for q in params[i + 1 : i + 3]:
            qname = (q.name or "").lower()
            if not is_length_param(q.name, q.type):
                continue

            if (
                "len" in qname
                or "size" in qname
                or pname
                and pname in qname
                or pname in {"buf", "buffer", "data", "input", "bytes"}
            ):
                reasons.append(
                    f"pointer parameter {p.raw!r} appears paired with length {q.raw!r}"
                )
                return True, reasons

    return False, reasons


def is_status_like_return(return_type: str | None) -> bool:
    if not return_type:
        return False
    rt = normalize_type(return_type)
    if rt in {"int", "bool", "unsigned int", "size_t", "ssize_t"}:
        return True
    lower = rt.lower()
    return any(token in lower for token in ["status", "result", "error", "errc"])


def is_callback_param(param: Param) -> bool:
    raw = param.raw
    lower = raw.lower()
    return "(*" in raw or "std::function" in raw or "callback" in lower or "handler" in lower


def is_custom_type_param(param: Param) -> bool:
    t = normalize_type(param.type)
    if not t:
        return False
    words = [word for word in re.findall(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)?", t)]
    if not words:
        return False
    if all(word in PRIMITIVE_TYPE_WORDS for word in words):
        return False
    if any(word in {"const", "volatile", "struct", "enum", "class"} for word in words):
        words = [word for word in words if word not in {"const", "volatile", "struct", "enum", "class"}]
    return any(word not in PRIMITIVE_TYPE_WORDS for word in words)


def _patterns(rules: dict[str, Any], section: str, name: str) -> list[str]:
    return list(rules.get(section, {}).get(name, {}).get("patterns", []))


def _contains_pattern(value: str, patterns: list[str]) -> bool:
    lower = normalize_type(value).lower()
    return any(normalize_type(pattern).lower() in lower for pattern in patterns)
