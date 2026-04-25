from __future__ import annotations

from typing import Any

from .features import DEFAULT_RULES, detect_pointer_length_pair
from .model import FeatureRecord, FunctionRecord, RankResult


def decision_from_score(score: int) -> str:
    if score >= 15:
        return "high_priority"
    if score >= 9:
        return "medium_priority"
    if score >= 5:
        return "low_priority"
    return "exclude"


def compute_confidence(fn: FunctionRecord, features: FeatureRecord, score: int) -> str:
    uncertainty = 0

    if fn.signature is None:
        uncertainty += 2

    if fn.parse_errors:
        uncertainty += 2

    if features.has_custom_type_param:
        uncertainty += 1

    if features.has_context_like_param:
        uncertainty += 1

    if features.has_pointer_length_pair or features.has_string_like_param:
        uncertainty -= 1

    if uncertainty <= 0:
        return "high"
    if uncertainty <= 2:
        return "medium"
    return "low"


def rank_function(
    fn: FunctionRecord,
    features: FeatureRecord,
    rules: dict[str, Any] | None = None,
) -> RankResult:
    rules = rules or DEFAULT_RULES
    score = 0
    reasons: list[str] = []
    concerns: list[str] = []

    for keyword, points in rules.get("name_keywords", {}).items():
        if keyword in fn.name.lower():
            score += int(points)
            reasons.append(f"name contains {keyword!r} (+{points})")
            break

    if features.has_pointer_length_pair:
        score += 6
        pair_reasons = detect_pointer_length_pair(fn.params)[1]
        reasons.append(pair_reasons[0] + " (+6)" if pair_reasons else "has pointer + length pair (+6)")

    if features.has_byte_pointer:
        points = int(rules.get("good_types", {}).get("byte_pointer", {}).get("score", 5))
        score += points
        reasons.append(f"has byte pointer input (+{points})")

    if features.has_string_like_param:
        points = int(rules.get("good_types", {}).get("string_like", {}).get("score", 4))
        score += points
        reasons.append(f"has string-like input (+{points})")

    if features.returns_status_like:
        score += 3
        reasons.append("returns status-like value (+3)")

    if features.calls_parse_decode:
        score += 2
        reasons.append("calls parser/decoder-like functions (+2)")

    if features.calls_memory_sensitive:
        score += 2
        reasons.append("calls memory-sensitive functions (+2)")

    if features.has_context_like_param:
        penalty = int(rules.get("bad_types", {}).get("context_like", {}).get("penalty", -4))
        score += penalty
        concerns.append(f"has context/session/manager-like parameter ({penalty})")

    if features.has_file_or_fd_param:
        penalty = int(rules.get("bad_types", {}).get("io_like", {}).get("penalty", -3))
        score += penalty
        concerns.append(f"has file/fd/path-like parameter ({penalty})")

    if features.has_callback_param:
        score -= 3
        concerns.append("has callback/function-pointer parameter (-3)")

    for keyword in features.path_keywords:
        penalty = int(rules.get("path_keywords", {}).get(keyword, -2))
        score += penalty
        concerns.append(f"path contains {keyword!r} ({penalty})")

    score = max(0, min(30, score))
    confidence = compute_confidence(fn, features, score)

    return RankResult(
        function_id=fn.id,
        base_score=score,
        final_score=score,
        confidence=confidence,
        decision=decision_from_score(score),
        reasons=reasons,
        concerns=concerns,
    )
