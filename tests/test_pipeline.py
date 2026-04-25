from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from fuzzrank.cli import main
from fuzzrank.features import extract_features
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


if __name__ == "__main__":
    unittest.main()
