from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .agent import DEFAULT_MODEL, TraceFlowAgent, load_jsonl
from ..eval import compute_mcq_accuracy, read_jsonl


TASK_FAMILIES: dict[str, list[str]] = {
    "route_operation": [
        "A1_route_retrieval",
        "A2_missing_step",
        "A3_next_activity",
    ],
    "attribute_inference": [
        "B1_condition_prediction",
        "B2_full_condition_set",
        "C1_tool_selection",
    ],
    "causal_ordering": [
        "D1_process_ordering",
    ],
}


ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "no_symbolic": {
        "use_symbolic": False,
        "use_symbolic_fallback": False,
    },
    "no_retrieval": {
        "use_retrieval": False,
        "top_k": 0,
    },
    "no_planning": {
        "use_planning": False,
    },
    "llm_decision_only": {
        "use_retrieval": False,
        "use_symbolic": False,
        "use_planning": False,
        "use_symbolic_fallback": False,
        "top_k": 0,
    },
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_result_file(result_file: Path, model_name: str) -> dict[str, Any]:
    results, preds, gts, _questions = read_jsonl(str(result_file))
    mcq = compute_mcq_accuracy(results)
    metrics = {
        "model": model_name,
        "mode": "mcq",
        "result_file": str(result_file),
        "total": len(results),
        "MCQ_accuracy": mcq["accuracy"],
        "MCQ_correct": mcq["correct"],
        "MCQ_total": mcq["total"],
    }
    per_task = {}
    task_buckets: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(results):
        task_buckets[(rec.get("task") or "unknown").strip() or "unknown"].append(idx)
    for task_name, indices in sorted(task_buckets.items()):
        sub_results = [results[j] for j in indices]
        sub_mcq = compute_mcq_accuracy(sub_results)
        per_task[task_name] = {
            "total": len(sub_results),
            "MCQ_accuracy": sub_mcq["accuracy"],
            "MCQ_correct": sub_mcq["correct"],
            "MCQ_total": sub_mcq["total"],
        }
    metrics["per_task"] = per_task
    return metrics


def compute_task_family_metrics(result_file: Path) -> dict[str, dict[str, Any]]:
    results, _preds, _gts, _questions = read_jsonl(str(result_file))
    out: dict[str, dict[str, Any]] = {}
    for family_name, tasks in TASK_FAMILIES.items():
        family_results: list[dict[str, Any]] = []
        for rec in results:
            if rec.get("task") in tasks:
                family_results.append(rec)
        family_mcq = compute_mcq_accuracy(family_results)
        out[family_name] = {
            "total": len(family_results),
            "MCQ_accuracy": family_mcq["accuracy"],
            "MCQ_correct": family_mcq["correct"],
            "MCQ_total": family_mcq["total"],
        }
    return out


def run_one_configuration(
    records: list[dict[str, Any]],
    raw_file: str,
    train_file: str,
    output_dir: Path,
    model_name: str,
    base_kwargs: dict[str, Any],
    config_name: str,
    config_overrides: dict[str, Any],
) -> dict[str, Any]:
    config_dir = output_dir / config_name
    result_file = config_dir / "results.jsonl"
    eval_file = config_dir / "results_eval.json"
    family_file = config_dir / "task_family_eval.json"

    agent_kwargs = dict(base_kwargs)
    agent_kwargs.update(config_overrides)

    print("")
    print("======================================================================")
    print(f"[TraceFlow Experiment] {config_name}")
    print(f"model={model_name}")
    print(f"overrides={config_overrides}")
    print("======================================================================")

    agent = TraceFlowAgent.from_files(
        raw_file=raw_file,
        train_file=train_file,
        model_name=model_name,
        top_k=agent_kwargs["top_k"],
        plan_max_new_tokens=agent_kwargs["plan_max_new_tokens"],
        answer_max_new_tokens=agent_kwargs["answer_max_new_tokens"],
        load_in_4bit=agent_kwargs["load_in_4bit"],
        use_retrieval=agent_kwargs["use_retrieval"],
        use_symbolic=agent_kwargs["use_symbolic"],
        use_planning=agent_kwargs["use_planning"],
        use_symbolic_fallback=agent_kwargs["use_symbolic_fallback"],
    )

    outputs: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        outputs.append(agent.answer_record(record))
        if idx % 25 == 0:
            print(f"[TraceFlow Experiment] {config_name}: processed {idx}")

    write_jsonl(result_file, outputs)

    metrics = evaluate_result_file(result_file, model_name=model_name)
    task_family_metrics = compute_task_family_metrics(result_file)
    metrics["experiment_name"] = config_name
    metrics["config"] = config_overrides
    metrics["task_family"] = task_family_metrics

    eval_file.parent.mkdir(parents=True, exist_ok=True)
    eval_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    family_file.write_text(json.dumps(task_family_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_name",
        "MCQ_accuracy",
        "MCQ_correct",
        "MCQ_total",
        "top_k",
        "use_retrieval",
        "use_symbolic",
        "use_planning",
        "use_symbolic_fallback",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            config = row.get("config", {})
            writer.writerow(
                {
                    "experiment_name": row.get("experiment_name", ""),
                    "MCQ_accuracy": row.get("MCQ_accuracy", 0.0),
                    "MCQ_correct": row.get("MCQ_correct", 0),
                    "MCQ_total": row.get("MCQ_total", 0),
                    "top_k": config.get("top_k", ""),
                    "use_retrieval": config.get("use_retrieval", ""),
                    "use_symbolic": config.get("use_symbolic", ""),
                    "use_planning": config.get("use_planning", ""),
                    "use_symbolic_fallback": config.get("use_symbolic_fallback", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TraceFlow ablations and additional experiments.")
    parser.add_argument("--data_file", required=True, help="Evaluation JSONL file.")
    parser.add_argument("--train_file", required=True, help="Training JSONL file.")
    parser.add_argument("--raw_file", required=True, help="Raw MatPROV JSONL file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save all experiment outputs.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Backbone model.")
    parser.add_argument("--top_k", type=int, default=8, help="Default top-k retrieval depth.")
    parser.add_argument("--plan_max_new_tokens", type=int, default=96, help="Planning token budget.")
    parser.add_argument("--answer_max_new_tokens", type=int, default=48, help="Answer token budget.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke tests.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit mode.")
    parser.add_argument(
        "--suite",
        choices=["full", "ablations_only", "topk_only"],
        default="full",
        help="Which experiment suite to run.",
    )
    parser.add_argument(
        "--topk_values",
        default="1,2,4,8,16",
        help="Comma-separated top-k values for the top-k sweep.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = load_jsonl(args.data_file)
    if args.limit is not None:
        records = records[: args.limit]

    base_kwargs = {
        "top_k": args.top_k,
        "plan_max_new_tokens": args.plan_max_new_tokens,
        "answer_max_new_tokens": args.answer_max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "use_retrieval": True,
        "use_symbolic": True,
        "use_planning": True,
        "use_symbolic_fallback": True,
    }

    all_metrics: list[dict[str, Any]] = []

    if args.suite in {"full", "ablations_only"}:
        for config_name, overrides in ABLATION_CONFIGS.items():
            effective_overrides = dict(base_kwargs)
            effective_overrides.update(overrides)
            metrics = run_one_configuration(
                records=records,
                raw_file=args.raw_file,
                train_file=args.train_file,
                output_dir=output_dir / "ablations",
                model_name=args.model,
                base_kwargs=base_kwargs,
                config_name=config_name,
                config_overrides=effective_overrides,
            )
            all_metrics.append(metrics)

    if args.suite in {"full", "topk_only"}:
        topk_values = [int(v.strip()) for v in args.topk_values.split(",") if v.strip()]
        for top_k in topk_values:
            overrides = dict(base_kwargs)
            overrides["top_k"] = top_k
            metrics = run_one_configuration(
                records=records,
                raw_file=args.raw_file,
                train_file=args.train_file,
                output_dir=output_dir / "topk",
                model_name=args.model,
                base_kwargs=base_kwargs,
                config_name=f"topk_{top_k}",
                config_overrides=overrides,
            )
            all_metrics.append(metrics)

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary_csv, all_metrics)
    print("")
    print("======================================================================")
    print(f"[TraceFlow Experiment] complete. Summary saved to {summary_json}")
    print("======================================================================")


if __name__ == "__main__":
    main()
