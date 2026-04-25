from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .model import FunctionRecord, RankResult


def build_candidates(
    functions: list[FunctionRecord],
    ranks: list[RankResult],
) -> list[dict[str, Any]]:
    fn_by_id = {fn.id: fn for fn in functions}
    ordered = sorted(ranks, key=lambda rank: (-rank.final_score, rank.function_id))
    candidates: list[dict[str, Any]] = []

    for index, rank in enumerate(ordered, start=1):
        fn = fn_by_id.get(rank.function_id)
        if not fn:
            continue
        candidates.append(
            {
                "rank": index,
                "function": fn.name,
                "file": fn.file,
                "line": fn.line_start,
                "signature": fn.signature,
                "base_score": rank.base_score,
                "final_score": rank.final_score,
                "confidence": rank.confidence,
                "decision": rank.decision,
                "harness_cost": rank.harness_cost,
                "input_mapping": rank.input_mapping,
                "reasons": rank.reasons,
                "concerns": rank.concerns,
                "llm_used": rank.llm_used,
                "llm_adjustment": rank.llm_adjustment,
            }
        )

    return candidates


def write_candidates_jsonl(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for candidate in candidates:
            f.write(json.dumps(candidate, sort_keys=True, ensure_ascii=False))
            f.write("\n")


def write_candidates_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "file",
        "line",
        "function",
        "signature",
        "final_score",
        "confidence",
        "decision",
        "harness_cost",
        "reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "rank": candidate["rank"],
                    "file": candidate["file"],
                    "line": candidate["line"],
                    "function": candidate["function"],
                    "signature": candidate["signature"] or "",
                    "final_score": candidate["final_score"],
                    "confidence": candidate["confidence"],
                    "decision": candidate["decision"],
                    "harness_cost": candidate["harness_cost"],
                    "reasons": "; ".join(candidate["reasons"]),
                }
            )
