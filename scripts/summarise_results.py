"""Recompute correctness figures from the raw evaluation runs in results/runs/.

Each run file holds one record per question, retaining the question, the ground
truth, the generated answer, the retrieved documents and the judge's verdict.
Correctness is the proportion of records the judge marked "Y", reported
separately for text-based and table-based questions.

  python scripts/summarise_results.py
  python scripts/summarise_results.py --failures per_page_colpali
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def is_correct(record) -> bool:
    return str(record.get("llm_correct", "")).upper().startswith("Y")


def fields(record):
    """Normalise the two run schemas.

    The earlier runs (the paper's, n=103) store `expected_answer` and
    `actual_answer` and carry no page provenance. The later runs (n=154) nest
    the ground truth under `ground_truth` and retain the retrieved documents.
    """
    if "ground_truth" in record:
        truth = record["ground_truth"]
        return {
            "expected": truth.get("answer"),
            "generated": record.get("llm_answer"),
            "document": truth.get("document"),
            "page": truth.get("page"),
        }
    return {
        "expected": record.get("expected_answer"),
        "generated": record.get("actual_answer"),
        "document": None,
        "page": None,
    }


def summarise() -> int:
    rows = []
    for path in sorted(RUNS.glob("*.json")):
        records = load(path)
        by_type = defaultdict(lambda: [0, 0])
        for record in records:
            bucket = by_type[record.get("question_type", "unknown")]
            bucket[1] += 1
            if is_correct(record):
                bucket[0] += 1

        correct = sum(v[0] for v in by_type.values())
        rows.append(
            {
                "run": path.stem,
                "n": len(records),
                "overall": 100 * correct / len(records) if records else 0.0,
                "by_type": {
                    k: (100 * v[0] / v[1], v[1]) for k, v in sorted(by_type.items())
                },
            }
        )

    if not rows:
        print(f"No run files found in {RUNS.relative_to(ROOT)}")
        return 1

    width = max(len(r["run"]) for r in rows)
    print(f"{'run':<{width}}  {'n':>4}  {'overall':>8}  {'text':>8}  {'table':>8}")
    print("-" * (width + 34))
    for r in sorted(rows, key=lambda r: -r["overall"]):
        text = r["by_type"].get("text", (None, 0))[0]
        table = r["by_type"].get("table", (None, 0))[0]
        fmt = lambda v: f"{v:>7.1f}%" if v is not None else f"{'-':>8}"
        print(
            f"{r['run']:<{width}}  {r['n']:>4}  {r['overall']:>7.1f}%  "
            f"{fmt(text)}  {fmt(table)}"
        )

    print(
        "\nNote: these are the 154- and 103-question working sets. The paper's "
        "headline table\nwas computed over the final 104 text / 101 table set; "
        "see README."
    )
    return 0


def show_failures(pattern: str, limit: int) -> int:
    matches = sorted(RUNS.glob(f"*{pattern}*.json"))
    if not matches:
        print(f"No run matching '{pattern}' in {RUNS.relative_to(ROOT)}")
        available = sorted(p.stem for p in RUNS.glob("*.json"))
        print("Available runs:\n  " + "\n  ".join(available))
        return 1

    for path in matches:
        wrong = [r for r in load(path) if not is_correct(r)]
        print(f"\n=== {path.stem}: {len(wrong)} incorrect")
        for record in wrong[:limit]:
            f = fields(record)
            print(f"\n  [{record.get('question_type')}] {record.get('question')}")
            print(f"    expected:  {f['expected']}")
            print(f"    generated: {f['generated']}")
            if f["document"]:
                print(f"    source:    p{f['page']} of {f['document']}")
        if len(wrong) > limit:
            print(f"\n  ... {len(wrong) - limit} more (raise --limit to see them)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failures",
        metavar="RUN",
        help="show incorrectly answered questions for runs matching this substring",
    )
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    if args.failures:
        return show_failures(args.failures, args.limit)
    return summarise()


if __name__ == "__main__":
    raise SystemExit(main())
