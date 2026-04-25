from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ctags_index import run_ctags
from .features import extract_features, load_rules
from .llm_review import (
    DEFAULT_LLM_COMMAND,
    DEFAULT_PROMPT_TEMPLATE,
    review_ranks,
    select_llm_review_ids,
)
from .merge import merge_ctags_functions
from .model import (
    feature_from_dict,
    function_from_dict,
    rank_from_dict,
    read_jsonl,
    write_jsonl,
)
from .output import build_candidates, write_candidates_csv, write_candidates_jsonl
from .ranker import rank_function
from .repo_scan import parse_languages, scan_source_files
from .ts_extract import extract_repo_functions


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fuzzrank")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="run the full pipeline")
    scan.add_argument("repo", type=Path)
    scan.add_argument("--out", type=Path, required=True)
    scan.add_argument("--languages", default="c,cpp")
    scan.add_argument("--rules", type=Path)
    scan.add_argument("--use-ctags", action=argparse.BooleanOptionalAction, default=True)
    scan.add_argument("--use-tree-sitter", action=argparse.BooleanOptionalAction, default=True)
    scan.add_argument("--llm-review", action="store_true")
    scan.add_argument("--llm-budget", type=int, default=100)
    scan.add_argument(
        "--llm-command",
        default=DEFAULT_LLM_COMMAND,
        help="LLM command template; use {prompt} to pass the function-specific prompt",
    )
    scan.add_argument(
        "--llm-prompt-template",
        type=Path,
        help=f"Jinja prompt template path; default is {DEFAULT_PROMPT_TEMPLATE}",
    )
    scan.set_defaults(func=cmd_scan)

    ctags = sub.add_parser("ctags", help="write ctags symbols JSONL")
    ctags.add_argument("repo", type=Path)
    ctags.add_argument("--out", type=Path, required=True)
    ctags.add_argument("--languages", default="c,cpp")
    ctags.set_defaults(func=cmd_ctags)

    extract = sub.add_parser("extract", help="extract functions JSONL")
    extract.add_argument("repo", type=Path)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--languages", default="c,cpp")
    extract.set_defaults(func=cmd_extract)

    features = sub.add_parser("features", help="extract feature JSONL")
    features.add_argument("functions", type=Path)
    features.add_argument("--out", type=Path, required=True)
    features.add_argument("--rules", type=Path)
    features.set_defaults(func=cmd_features)

    rank = sub.add_parser("rank", help="rank functions")
    rank.add_argument("functions", type=Path)
    rank.add_argument("features", type=Path)
    rank.add_argument("--out", type=Path, required=True)
    rank.add_argument("--rules", type=Path)
    rank.set_defaults(func=cmd_rank)

    review = sub.add_parser("review", help="apply cached or command-backed LLM review")
    review.add_argument("functions", type=Path)
    review.add_argument("features", type=Path)
    review.add_argument("ranked", type=Path)
    review.add_argument("--repo", type=Path, default=Path("."))
    review.add_argument("--out", type=Path, required=True)
    review.add_argument("--budget", type=int, default=100)
    review.add_argument("--cache-dir", type=Path, default=Path(".cache/llm_reviews"))
    review.add_argument(
        "--llm-command",
        default=DEFAULT_LLM_COMMAND,
        help="LLM command template; use {prompt} to pass the function-specific prompt",
    )
    review.add_argument(
        "--llm-prompt-template",
        type=Path,
        help=f"Jinja prompt template path; default is {DEFAULT_PROMPT_TEMPLATE}",
    )
    review.set_defaults(func=cmd_review)

    export = sub.add_parser("export", help="export candidates JSONL/CSV")
    export.add_argument("functions", type=Path)
    export.add_argument("ranked", type=Path)
    export.add_argument("--jsonl", type=Path, required=True)
    export.add_argument("--csv", type=Path, required=True)
    export.set_defaults(func=cmd_export)

    return parser


def cmd_scan(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    languages = parse_languages(args.languages)
    rules = load_rules(args.rules)

    files = scan_source_files(repo, languages)
    print(f"found {len(files)} source files", file=sys.stderr)

    ctags_records = []
    if args.use_ctags:
        ctags_records, warnings = run_ctags(repo, languages)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
    write_jsonl(out / "symbols.jsonl", ctags_records)

    if args.use_tree_sitter:
        functions = extract_repo_functions(repo, files)
    else:
        functions = []
    functions = merge_ctags_functions(ctags_records, functions, repo)
    write_jsonl(out / "functions.jsonl", functions)

    features = [extract_features(fn, rules) for fn in functions]
    write_jsonl(out / "features.jsonl", features)

    ranks = [rank_function(fn, feature, rules) for fn, feature in zip(functions, features)]
    write_jsonl(out / "ranked.jsonl", ranks)

    if args.llm_review:
        review_ids = select_llm_review_ids(
            functions=functions,
            features=features,
            ranks=ranks,
            budget=args.llm_budget,
        )
        print(
            f"llm review selected {len(review_ids)} candidates "
            f"(budget {args.llm_budget}, command: {args.llm_command!r})",
            file=sys.stderr,
        )
        reviewed = review_ranks(
            repo=repo,
            functions=functions,
            features=features,
            ranks=ranks,
            budget=args.llm_budget,
            cache_dir=out / ".cache" / "llm_reviews",
            llm_command=args.llm_command,
            prompt_template=args.llm_prompt_template,
            progress=lambda message: print(message, file=sys.stderr),
        )
        llm_used = sum(1 for rank in reviewed if rank.llm_used)
        llm_blockers = sum(
            1
            for rank in reviewed
            if any(concern.startswith("LLM blocker:") for concern in rank.concerns)
        )
        print(
            f"llm review applied to {llm_used} candidates "
            f"({llm_blockers} with blockers)",
            file=sys.stderr,
        )
    else:
        reviewed = ranks
    write_jsonl(out / "reviewed.jsonl", reviewed)

    candidates = build_candidates(functions, reviewed)
    write_candidates_jsonl(out / "candidates.jsonl", candidates)
    write_candidates_csv(out / "candidates.csv", candidates)
    print(f"wrote {len(candidates)} candidates to {out}", file=sys.stderr)
    return 0


def cmd_ctags(args: argparse.Namespace) -> int:
    records, warnings = run_ctags(args.repo.resolve(), parse_languages(args.languages))
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    write_jsonl(args.out, records)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    files = scan_source_files(repo, parse_languages(args.languages))
    functions = extract_repo_functions(repo, files)
    write_jsonl(args.out, functions)
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    rules = load_rules(args.rules)
    functions = [function_from_dict(obj) for obj in read_jsonl(args.functions)]
    features = [extract_features(fn, rules) for fn in functions]
    write_jsonl(args.out, features)
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    rules = load_rules(args.rules)
    functions = [function_from_dict(obj) for obj in read_jsonl(args.functions)]
    features = [feature_from_dict(obj) for obj in read_jsonl(args.features)]
    features_by_id = {feature.function_id: feature for feature in features}
    ranks = [
        rank_function(fn, features_by_id[fn.id], rules)
        for fn in functions
        if fn.id in features_by_id
    ]
    write_jsonl(args.out, ranks)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    functions = [function_from_dict(obj) for obj in read_jsonl(args.functions)]
    features = [feature_from_dict(obj) for obj in read_jsonl(args.features)]
    ranks = [rank_from_dict(obj) for obj in read_jsonl(args.ranked)]
    review_ids = select_llm_review_ids(
        functions=functions,
        features=features,
        ranks=ranks,
        budget=args.budget,
    )
    print(
        f"llm review selected {len(review_ids)} candidates "
        f"(budget {args.budget}, command: {args.llm_command!r})",
        file=sys.stderr,
    )
    reviewed = review_ranks(
        repo=args.repo.resolve(),
        functions=functions,
        features=features,
        ranks=ranks,
        budget=args.budget,
        cache_dir=args.cache_dir,
        llm_command=args.llm_command,
        prompt_template=args.llm_prompt_template,
        progress=lambda message: print(message, file=sys.stderr),
    )
    llm_used = sum(1 for rank in reviewed if rank.llm_used)
    llm_blockers = sum(
        1
        for rank in reviewed
        if any(concern.startswith("LLM blocker:") for concern in rank.concerns)
    )
    print(
        f"llm review applied to {llm_used} candidates ({llm_blockers} with blockers)",
        file=sys.stderr,
    )
    write_jsonl(args.out, reviewed)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    functions = [function_from_dict(obj) for obj in read_jsonl(args.functions)]
    ranks = [rank_from_dict(obj) for obj in read_jsonl(args.ranked)]
    candidates = build_candidates(functions, ranks)
    write_candidates_jsonl(args.jsonl, candidates)
    write_candidates_csv(args.csv, candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
