from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import DEFAULT_MODEL, TraceFlowAgent, load_jsonl


def write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fair TraceFlow with Qwen2.5-7B over train-split processes only.")
    parser.add_argument("--data_file", required=True, help="Path to the MatProcBench evaluation JSONL.")
    parser.add_argument("--train_file", required=True, help="Path to the MatProcBench training JSONL.")
    parser.add_argument("--raw_file", required=True, help="Path to MatPROV.jsonl.")
    parser.add_argument("--save_file", required=True, help="Output JSONL path compatible with MatProcAgent/eval.py.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM used by TraceFlow. Default: Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--top_k", type=int, default=8, help="Number of retrieved train processes.")
    parser.add_argument("--plan_max_new_tokens", type=int, default=96, help="Max tokens for the planning call.")
    parser.add_argument("--answer_max_new_tokens", type=int, default=48, help="Max tokens for the answer call.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of records to run for a smoke test.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load Qwen in 4-bit mode.")
    parser.add_argument("--disable_retrieval", action="store_true", help="Disable retrieved train-process analogs.")
    parser.add_argument("--disable_symbolic", action="store_true", help="Disable symbolic scoring and symbolic priors.")
    parser.add_argument("--disable_planning", action="store_true", help="Disable the planning pass before answer selection.")
    parser.add_argument(
        "--disable_symbolic_fallback",
        action="store_true",
        help="Disable fallback to the best symbolic option when the LLM output is not a valid choice letter.",
    )
    args = parser.parse_args()

    print(f"[TraceFlow] Building train-only process index from {args.train_file}")
    agent = TraceFlowAgent.from_files(
        raw_file=args.raw_file,
        train_file=args.train_file,
        model_name=args.model,
        top_k=args.top_k,
        plan_max_new_tokens=args.plan_max_new_tokens,
        answer_max_new_tokens=args.answer_max_new_tokens,
        load_in_4bit=args.load_in_4bit,
        use_retrieval=not args.disable_retrieval,
        use_symbolic=not args.disable_symbolic,
        use_planning=not args.disable_planning,
        use_symbolic_fallback=not args.disable_symbolic_fallback,
    )

    print(f"[TraceFlow] Loading benchmark records from {args.data_file}")
    records = load_jsonl(args.data_file)
    if args.limit is not None:
        records = records[: args.limit]

    outputs = []
    for idx, record in enumerate(records, start=1):
        outputs.append(agent.answer_record(record))
        if idx % 25 == 0:
            print(f"[TraceFlow] Processed {idx} records")

    write_jsonl(args.save_file, outputs)
    print(f"[TraceFlow] Wrote {len(outputs)} records to {args.save_file}")


if __name__ == "__main__":
    main()
