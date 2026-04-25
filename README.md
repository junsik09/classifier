# fuzzrank

`fuzzrank` ranks C/C++ functions that look suitable for fuzzing. It stores each
pipeline stage as JSONL so extraction, ranking, review, and export can be debugged
or rerun independently.

The MVP is intentionally conservative:

- Universal Ctags is used when available, but the pipeline continues without it.
- Tree-sitter is used when `tree-sitter-languages` is installed, with a lightweight
  C/C++ fallback extractor otherwise.
- Ranking is rule based and explanation first.
- LLM review is optional, budget limited, cached, and only adjusts heuristic scores.

## Quick Start

```bash
python3 -m fuzzrank scan /path/to/repo --out outputs/repo1 --languages c,cpp
```

Outputs:

- `symbols.jsonl`
- `functions.jsonl`
- `features.jsonl`
- `ranked.jsonl`
- `reviewed.jsonl`
- `candidates.jsonl`
- `candidates.csv`

## Optional LLM Review

`fuzzrank` does not hard-code one LLM provider. Pass a command that reads one
evidence pack from stdin and writes one JSON review to stdout:

```bash
python3 -m fuzzrank scan /path/to/repo \
  --out outputs/repo1 \
  --llm-review \
  --llm-budget 100 \
  --llm-command "./review_one_function"
```

Review output must be JSON:

```json
{
  "decision": "medium_priority",
  "score_adjustment": 2,
  "confidence": "high",
  "harness_cost": "low",
  "input_mapping": {
    "fuzz_bytes_parameter": "data",
    "fixed_parameters": {
      "flags": "0"
    }
  },
  "reasons": [],
  "blockers": [],
  "rules_to_add": []
}
```

Reviews are cached under `.cache/llm_reviews` inside the output directory.

## Stage Commands

```bash
python3 -m fuzzrank ctags /path/to/repo --out outputs/symbols.jsonl
python3 -m fuzzrank extract /path/to/repo --out outputs/functions.jsonl
python3 -m fuzzrank features outputs/functions.jsonl --out outputs/features.jsonl
python3 -m fuzzrank rank outputs/functions.jsonl outputs/features.jsonl --out outputs/ranked.jsonl
python3 -m fuzzrank review outputs/functions.jsonl outputs/features.jsonl outputs/ranked.jsonl --out outputs/reviewed.jsonl
python3 -m fuzzrank export outputs/functions.jsonl outputs/reviewed.jsonl --jsonl outputs/candidates.jsonl --csv outputs/candidates.csv
```

## Notes

Install Universal Ctags for broader symbol coverage:

```bash
ctags --version
```

Install optional Python parsers when you want tree-sitter extraction:

```bash
python3 -m pip install '.[tree-sitter,yaml]'
```
