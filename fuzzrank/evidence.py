from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import FeatureRecord, FunctionRecord, RankResult, to_dict


def extract_lines(source_text: str, start: int, end: int) -> str:
    lines = source_text.splitlines()
    start = max(1, start)
    end = min(len(lines), max(start, end))
    selected = []
    for lineno in range(start, end + 1):
        selected.append(f"{lineno}: {lines[lineno - 1]}")
    return "\n".join(selected)


def build_evidence_pack(
    fn: FunctionRecord,
    features: FeatureRecord,
    rank: RankResult,
    repo: Path,
) -> dict[str, Any]:
    source_path = (repo / fn.file).resolve()
    try:
        source_text = source_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        source_text = fn.body_excerpt or ""

    excerpt = extract_lines(
        source_text,
        start=max(1, fn.line_start - 40),
        end=(fn.line_end or fn.line_start) + 60,
    )

    return {
        "target": {
            "name": fn.name,
            "file": fn.file,
            "absolute_file": source_path.as_posix(),
            "line_start": fn.line_start,
            "signature": fn.signature,
            "return_type": fn.return_type,
            "params": [to_dict(param) for param in fn.params],
        },
        "body_excerpt": excerpt,
        "calls": fn.calls[:50],
        "features": to_dict(features),
        "base_rank": {
            "score": rank.base_score,
            "confidence": rank.confidence,
            "reasons": rank.reasons,
            "concerns": rank.concerns,
        },
        "uncertainties": fn.parse_errors,
    }
