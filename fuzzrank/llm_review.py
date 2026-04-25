from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from importlib.resources import files
from pathlib import Path
from typing import Any

try:
    from jinja2 import Environment, StrictUndefined
except ModuleNotFoundError:  # pragma: no cover - exercised only without package deps
    Environment = None  # type: ignore[assignment]
    StrictUndefined = None  # type: ignore[assignment]

from .evidence import build_evidence_pack
from .model import FeatureRecord, FunctionRecord, RankResult
from .ranker import decision_from_score


PROMPT_VERSION = "fuzzrank-review-v3-jinja-cline"
HEURISTIC_VERSION = "heuristic-v1"
DEFAULT_LLM_COMMAND = "cline -y {prompt}"
DEFAULT_PROMPT_TEMPLATE = "fuzzrank/prompts/review.j2"
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
    prompt_template_hash: str | None = None,
) -> str:
    payload = {
        "prompt_version": prompt_version,
        "heuristic_version": heuristic_version,
        "prompt_template_hash": prompt_template_hash,
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
    prompt_template: Path | None = None,
) -> list[RankResult]:
    if budget <= 0:
        return list(ranks)

    template_text = load_prompt_template(prompt_template)
    template_hash = hashlib.sha256(template_text.encode("utf-8")).hexdigest()
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
        cache_key = review_cache_key(evidence, prompt_template_hash=template_hash)
        cache_path = cache_dir / f"{cache_key}.json"
        review = load_cached_review(cache_path)

        if review is None and llm_command and used < budget:
            review = invoke_llm_command(llm_command, evidence, template_text=template_text)
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


def load_prompt_template(path: Path | None = None) -> str:
    if path:
        return path.read_text(encoding="utf-8")
    return files("fuzzrank").joinpath("prompts/review.j2").read_text(encoding="utf-8")


def build_review_prompt(
    evidence: dict[str, Any],
    template_text: str | None = None,
) -> str:
    if Environment is None or StrictUndefined is None:
        raise RuntimeError(
            "Jinja2 is required for LLM prompt rendering; install with `pip install -e .`"
        )

    template_text = template_text or load_prompt_template()
    evidence_json = json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=False)
    env = Environment(
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["json_dumps"] = _json_dumps
    template = env.from_string(template_text)
    return template.render(
        evidence=evidence,
        evidence_json=evidence_json,
        target=evidence.get("target", {}),
        features=evidence.get("features", {}),
        base_rank=evidence.get("base_rank", {}),
        calls=evidence.get("calls", []),
        body_excerpt=evidence.get("body_excerpt", ""),
        uncertainties=evidence.get("uncertainties", []),
    )


def invoke_llm_command(
    command: str,
    evidence: dict[str, Any],
    template_text: str | None = None,
) -> dict[str, Any]:
    try:
        prompt = build_review_prompt(evidence, template_text=template_text)
    except Exception as exc:
        return {
            "score_adjustment": 0,
            "confidence": "low",
            "harness_cost": "unknown",
            "reasons": [],
            "blockers": [str(exc)],
        }

    try:
        argv = build_command_argv(command, prompt)
    except ValueError as exc:
        return {
            "score_adjustment": 0,
            "confidence": "low",
            "harness_cost": "unknown",
            "reasons": [],
            "blockers": [str(exc)],
        }
    stdin_payload = None if "{prompt}" in command else json.dumps(evidence, sort_keys=True)

    try:
        proc = subprocess.run(
            argv,
            input=stdin_payload,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "score_adjustment": 0,
            "confidence": "low",
            "harness_cost": "unknown",
            "reasons": [],
            "blockers": [f"LLM command not found: {exc.filename}"],
        }

    if proc.returncode != 0:
        return {
            "score_adjustment": 0,
            "confidence": "low",
            "harness_cost": "unknown",
            "reasons": [],
            "blockers": [proc.stderr.strip() or f"LLM command exited with {proc.returncode}"],
        }
    review = parse_review_output(proc.stdout)
    if review is not None:
        return validate_review(review)

    return {
        "score_adjustment": 0,
        "confidence": "low",
        "harness_cost": "unknown",
        "reasons": [],
        "blockers": ["LLM command returned no valid JSON object"],
    }


def build_command_argv(command: str, prompt: str) -> list[str]:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("empty LLM command")
    return [part.replace("{prompt}", prompt) for part in argv]


def parse_review_output(output: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(output)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char != "{":
            continue
        parsed = _try_decode_object(decoder, output[index:])
        if parsed is not None:
            return parsed
    return None


def _try_decode_object(decoder: json.JSONDecoder, text: str) -> dict[str, Any] | None:
    try:
        parsed, _ = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)


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
