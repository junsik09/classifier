from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Param:
    name: str | None
    type: str
    raw: str


@dataclass
class FunctionRecord:
    id: str
    name: str
    file: str
    line_start: int
    line_end: int | None
    language: str
    signature: str | None
    return_type: str | None
    params: list[Param]
    scope: str | None = None
    is_static: bool | None = None
    is_method: bool = False
    body_excerpt: str | None = None
    calls: list[str] = field(default_factory=list)
    source: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


@dataclass
class FeatureRecord:
    function_id: str
    name_keywords: list[str]
    path_keywords: list[str]
    has_byte_pointer: bool
    has_char_pointer: bool
    has_void_pointer: bool
    has_length_param: bool
    has_pointer_length_pair: bool
    has_string_like_param: bool
    has_custom_type_param: bool
    has_context_like_param: bool
    has_file_or_fd_param: bool
    has_callback_param: bool
    returns_status_like: bool
    calls_memory_sensitive: bool
    calls_parse_decode: bool


@dataclass
class RankResult:
    function_id: str
    base_score: int
    final_score: int
    confidence: str
    decision: str
    reasons: list[str]
    concerns: list[str]
    llm_used: bool = False
    llm_adjustment: int = 0
    harness_cost: str = "unknown"
    input_mapping: dict[str, Any] = field(default_factory=dict)


def stable_function_id(file: str, name: str, line_start: int) -> str:
    raw = f"{Path(file).as_posix()}:{line_start}:{name}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"{Path(file).name}:{name}:{line_start}:{digest}"


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_dict(record), sort_keys=True, ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def param_from_dict(obj: dict[str, Any]) -> Param:
    return Param(
        name=obj.get("name"),
        type=obj.get("type") or "",
        raw=obj.get("raw") or "",
    )


def function_from_dict(obj: dict[str, Any]) -> FunctionRecord:
    return FunctionRecord(
        id=obj["id"],
        name=obj["name"],
        file=obj["file"],
        line_start=int(obj["line_start"]),
        line_end=obj.get("line_end"),
        language=obj.get("language") or "unknown",
        signature=obj.get("signature"),
        return_type=obj.get("return_type"),
        params=[param_from_dict(p) for p in obj.get("params", [])],
        scope=obj.get("scope"),
        is_static=obj.get("is_static"),
        is_method=bool(obj.get("is_method", False)),
        body_excerpt=obj.get("body_excerpt"),
        calls=list(obj.get("calls", [])),
        source=list(obj.get("source", [])),
        parse_errors=list(obj.get("parse_errors", [])),
    )


def feature_from_dict(obj: dict[str, Any]) -> FeatureRecord:
    return FeatureRecord(
        function_id=obj["function_id"],
        name_keywords=list(obj.get("name_keywords", [])),
        path_keywords=list(obj.get("path_keywords", [])),
        has_byte_pointer=bool(obj.get("has_byte_pointer", False)),
        has_char_pointer=bool(obj.get("has_char_pointer", False)),
        has_void_pointer=bool(obj.get("has_void_pointer", False)),
        has_length_param=bool(obj.get("has_length_param", False)),
        has_pointer_length_pair=bool(obj.get("has_pointer_length_pair", False)),
        has_string_like_param=bool(obj.get("has_string_like_param", False)),
        has_custom_type_param=bool(obj.get("has_custom_type_param", False)),
        has_context_like_param=bool(obj.get("has_context_like_param", False)),
        has_file_or_fd_param=bool(obj.get("has_file_or_fd_param", False)),
        has_callback_param=bool(obj.get("has_callback_param", False)),
        returns_status_like=bool(obj.get("returns_status_like", False)),
        calls_memory_sensitive=bool(obj.get("calls_memory_sensitive", False)),
        calls_parse_decode=bool(obj.get("calls_parse_decode", False)),
    )


def rank_from_dict(obj: dict[str, Any]) -> RankResult:
    return RankResult(
        function_id=obj["function_id"],
        base_score=int(obj.get("base_score", 0)),
        final_score=int(obj.get("final_score", obj.get("base_score", 0))),
        confidence=obj.get("confidence") or "low",
        decision=obj.get("decision") or "exclude",
        reasons=list(obj.get("reasons", [])),
        concerns=list(obj.get("concerns", [])),
        llm_used=bool(obj.get("llm_used", False)),
        llm_adjustment=int(obj.get("llm_adjustment", 0)),
        harness_cost=obj.get("harness_cost") or "unknown",
        input_mapping=dict(obj.get("input_mapping", {})),
    )
