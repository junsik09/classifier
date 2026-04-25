from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .evidence import build_evidence_pack
from .model import FeatureRecord, FunctionRecord, RankResult
from .ranker import decision_from_score


PROMPT_VERSION = "fuzzrank-review-v1"
HEURISTIC_VERSION = "heuristic-v1"
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_HARNESS_COST = {"low", "medium", "high", "unknown"}


def should_llm_review(rank: RankResult, features: FeatureRecord, fn: FunctionRecord) -> bool:
    if rank.final_score < 6:
        return False

    if rank.confidence == "low":
        return True

    if features.has_custom_type_param and rank.final_score >= 8:
        return True

    if fn.parse_errors and rank.final_score >= 8:
        return True

    if features.has_context_like_param and features.has_pointer_length_pair:
        return True

    return False


def review_cache_key(
    evidence: dict[str, Any],
    prompt_version: str = PROMPT_VERSION,
    heuristic_version: str = HEURISTIC_VERSION,
) -> str:
    payload = {
        "prompt_version": prompt_version,
        "heuristic_version": heuristic_version,
        "evidence": evidence,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def review_ranks(
    repo: Path,
    functions: list[FunctionRecord],
    features: list[FeatureRecord],
    ranks: list[RankResult],
    budget: int,
    cache_dir: Path,
    llm_command: str | None = None,
) -> list[RankResult]:
    fn_by_id = {fn.id: fn for fn in functions}
    features_by_id = {feature.function_id: feature for feature in features}
    reviewed: list[RankResult] = []
    used = 0
    cache_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(ranks, key=lambda rank: rank.final_score, reverse=True)
    review_ids: set[str] = set()
    for rank in ordered:
        fn = fn_by_id.get(rank.function_id)
        feature = features_by_id.get(rank.function_id)
        if not fn or not feature:
            continue
        if should_llm_review(rank, feature, fn):
            review_ids.add(rank.function_id)
        if len(review_ids) >= budget:
            break

    for rank in ranks:
        fn = fn_by_id.get(rank.function_id)
        feature = features_by_id.get(rank.function_id)
        if not fn or not feature or rank.function_id not in review_ids:
            reviewed.append(rank)
            continue

        evidence = build_evidence_pack(fn, feature, rank, repo)
        cache_key = review_cache_key(evidence)
        cache_path = cache_dir / f"{cache_key}.json"
        review = load_cached_review(cache_path)

        if review is None and llm_command and used < budget:
            review = invoke_llm_command(llm_command, evidence)
            save_cached_review(cache_path, review)
            used += 1

        if review is None:
            reviewed.append(rank)
        else:
            reviewed.append(apply_llm_review(rank, review))

    return reviewed


def load_cached_review(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_cached_review(path: Path, review: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(review, f, sort_keys=True, indent=2)
        f.write("\n")


def invoke_llm_command(command: str, evidence: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        shlex.split(command),
        input=json.dumps(evidence, sort_keys=True),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "score_adjustment": 0,
            "confidence": "low",
            "harness_cost": "unknown",
            "reasons": [],
            "blockers": [proc.stderr.strip() or f"LLM command exited with {proc.returncode}"],
        }
    try:
        return validate_review(json.loads(proc.stdout))
    except json.JSONDecodeError as exc:
        return {
            "score_adjustment": 0,
            "confidence": "low",
            "harness_cost": "unknown",
            "reasons": [],
            "blockers": [f"LLM command returned invalid JSON: {exc}"],
        }


def validate_review(review: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(review, dict):
        return {}
    normalized = dict(review)
    normalized["score_adjustment"] = _clamp_int(normalized.get("score_adjustment", 0), -8, 8)
    if normalized.get("confidence") not in VALID_CONFIDENCE:
        normalized["confidence"] = "low"
    if normalized.get("harness_cost") not in VALID_HARNESS_COST:
        normalized["harness_cost"] = "unknown"
    normalized.setdefault("reasons", [])
    normalized.setdefault("blockers", [])
    normalized.setdefault("input_mapping", {})
    normalized.setdefault("rules_to_add", [])
    return normalized


def apply_llm_review(rank: RankResult, review: dict[str, Any]) -> RankResult:
    review = validate_review(review)
    adj = int(review.get("score_adjustment", 0))

    rank.llm_used = True
    rank.llm_adjustment = adj
    rank.final_score = max(0, min(30, rank.base_score + adj))
    rank.confidence = review.get("confidence", rank.confidence)
    rank.decision = decision_from_score(rank.final_score)
    rank.harness_cost = review.get("harness_cost", rank.harness_cost)
    rank.input_mapping = dict(review.get("input_mapping", {}))

    rank.reasons.extend([f"LLM: {r}" for r in review.get("reasons", [])])
    rank.concerns.extend([f"LLM blocker: {b}" for b in review.get("blockers", [])])

    return rank


def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    return max(minimum, min(maximum, parsed))
