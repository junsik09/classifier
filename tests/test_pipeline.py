from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fuzzrank.cli import main
from fuzzrank.features import extract_features
from fuzzrank.llm_review import (
    build_command_argv,
    build_review_prompt,
    invoke_llm_command,
    parse_review_output,
    review_ranks,
)
from fuzzrank.model import RankResult
from fuzzrank.ranker import rank_function
from fuzzrank.ts_extract import extract_repo_functions


FIXTURE_REPO = Path(__file__).parent / "fixtures"


class PipelineTest(unittest.TestCase):
    def test_fallback_extractor_finds_buffer_parser(self) -> None:
        functions = extract_repo_functions(FIXTURE_REPO)
        by_name = {fn.name: fn for fn in functions}

        self.assertIn("parse_frame", by_name)
        self.assertIn("decode_message", by_name)
        self.assertEqual(by_name["parse_frame"].line_start, 6)
        self.assertEqual([param.name for param in by_name["parse_frame"].params], ["data", "len"])
        self.assertIn("memcmp", by_name["parse_frame"].calls)

    def test_ranker_prioritizes_pointer_length_parser(self) -> None:
        functions = extract_repo_functions(FIXTURE_REPO)
        parse_frame = next(fn for fn in functions if fn.name == "parse_frame")
        feature = extract_features(parse_frame)
        rank = rank_function(parse_frame, feature)

        self.assertEqual(rank.decision, "high_priority")
        self.assertGreaterEqual(rank.final_score, 15)
        self.assertTrue(any("pointer" in reason for reason in rank.reasons))

    def test_scan_writes_candidates_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            status = main(
                [
                    "scan",
                    str(FIXTURE_REPO),
                    "--out",
                    str(out),
                    "--languages",
                    "c",
                    "--no-use-ctags",
                ]
            )
            self.assertEqual(status, 0)

            with (out / "candidates.csv").open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(rows[0]["function"], "parse_frame")
            self.assertEqual(rows[0]["decision"], "high_priority")

    def test_cline_command_template_receives_custom_prompt(self) -> None:
        evidence = {
            "target": {
                "name": "parse_frame",
                "signature": "int parse_frame(const uint8_t *data, size_t len)",
            },
            "features": {"has_pointer_length_pair": True},
        }
        prompt = build_review_prompt(evidence)
        argv = build_command_argv("cline -y {prompt}", prompt)

        self.assertEqual(argv[:2], ["cline", "-y"])
        self.assertIn("parse_frame", argv[2])
        self.assertIn("Allowed JSON schema", argv[2])

    def test_review_prompt_uses_jinja_template_context(self) -> None:
        prompt = build_review_prompt(
            {
                "target": {"name": "parse_frame"},
                "features": {"has_pointer_length_pair": True},
            },
            template_text="target={{ target.name }} pair={{ features.has_pointer_length_pair }}",
        )

        self.assertEqual(prompt, "target=parse_frame pair=True")

    def test_invoke_llm_command_parses_cline_json_output(self) -> None:
        evidence = {
            "target": {
                "name": "parse_frame",
                "signature": "int parse_frame(const uint8_t *data, size_t len)",
            }
        }

        class Result:
            returncode = 0
            stdout = 'ok\n{"score_adjustment": 2, "confidence": "high", "harness_cost": "low"}\n'
            stderr = ""

        with patch("fuzzrank.llm_review.subprocess.run", return_value=Result()) as run:
            review = invoke_llm_command("cline -y {prompt}", evidence)

        argv = run.call_args.args[0]
        kwargs = run.call_args.kwargs
        self.assertEqual(argv[0:2], ["cline", "-y"])
        self.assertIn("parse_frame", argv[2])
        self.assertIsNone(kwargs["input"])
        self.assertEqual(review["score_adjustment"], 2)
        self.assertEqual(review["harness_cost"], "low")

    def test_review_ranks_calls_cline_only_for_selected_candidates(self) -> None:
        functions = extract_repo_functions(FIXTURE_REPO)
        features = [extract_features(fn) for fn in functions]
        parse_frame = next(fn for fn in functions if fn.name == "parse_frame")
        helper = next(fn for fn in functions if fn.name == "helper")
        ranks = [
            RankResult(
                function_id=parse_frame.id,
                base_score=10,
                final_score=10,
                confidence="low",
                decision="medium_priority",
                reasons=[],
                concerns=[],
            ),
            RankResult(
                function_id=helper.id,
                base_score=4,
                final_score=4,
                confidence="low",
                decision="exclude",
                reasons=[],
                concerns=[],
            ),
        ]

        class Result:
            returncode = 0
            stdout = '{"score_adjustment": 1, "confidence": "high", "harness_cost": "low"}'
            stderr = ""

        with tempfile.TemporaryDirectory() as tmp:
            with patch("fuzzrank.llm_review.subprocess.run", return_value=Result()) as run:
                reviewed = review_ranks(
                    repo=FIXTURE_REPO,
                    functions=functions,
                    features=features,
                    ranks=ranks,
                    budget=10,
                    cache_dir=Path(tmp),
                    llm_command="cline -y {prompt}",
                )

        self.assertEqual(run.call_count, 1)
        self.assertTrue(next(rank for rank in reviewed if rank.function_id == parse_frame.id).llm_used)
        self.assertFalse(next(rank for rank in reviewed if rank.function_id == helper.id).llm_used)

    def test_parse_review_output_finds_embedded_json(self) -> None:
        parsed = parse_review_output('cline says:\n{"score_adjustment": -1}\n')
        self.assertEqual(parsed, {"score_adjustment": -1})


if __name__ == "__main__":
    unittest.main()
