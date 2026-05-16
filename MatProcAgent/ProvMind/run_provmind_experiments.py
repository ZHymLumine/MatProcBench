"""
run_provmind_experiments.py — ProvMind ablation experiments for MatProcBench.

Ablation groups
---------------
references  Baselines: pure LLM, TraceFlow (heuristic+symbolic), ProvMind full system
retrieval   Decompose retrieval views (text / struct / heuristic and combinations)
scoring     Scoring mode sweep (symbolic / neural / hybrid weight variants)
modules     Agent module ablations (planning, symbolic fallback, symbolic scoring)
fusion      Retrieval fusion weight sweep (α text, β struct, γ heuristic)
topk        Top-k retrieval depth sweep

Suite options: references | retrieval | scoring | modules | fusion | topk | full

Usage
-----
  python -m MatProcAgent.ProvMind.run_provmind_experiments \\
      --data_file   data/processed/additional_split/test.jsonl \\
      --train_file  data/processed/additional_split/train.jsonl \\
      --raw_file    data/raw_data/MatPROV.jsonl \\
      --output_dir  MatProcAgent/ProvMind/results/provmind_experiments/MODEL/SPLIT \\
      --model       Qwen/Qwen2.5-7B-Instruct \\
      --suite       full
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .agent import DEFAULT_MODEL, TraceFlowAgent
from .dual_view_agent import DualViewAgent
from .utils import load_jsonl
from ..eval import compute_mcq_accuracy, read_jsonl


# ── Task family groupings (shared with run_experiments.py) ────────────────────
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


# ── Ablation config groups ────────────────────────────────────────────────────
#
# Each entry is a dict of overrides applied on top of base_kwargs (ProvMind full
# system).  The special key "_agent" selects which agent class to instantiate:
#   "traceflow"  → TraceFlowAgent  (no neural retrieval; used for LLM/TraceFlow baselines)
#   "dualview"   → DualViewAgent   (default; ProvMind)

# Group 0 – Reference baselines
REFERENCE_CONFIGS: dict[str, dict[str, Any]] = {
    # Pure LLM: no retrieval, no symbolic scoring, no planning
    "llm_only": {
        "_agent": "traceflow",
        "use_retrieval": False,
        "use_symbolic": False,
        "use_planning": False,
        "use_symbolic_fallback": False,
    },
    # TraceFlow (heuristic retrieval + symbolic scoring + planning)
    "traceflow": {
        "_agent": "traceflow",
    },
    # ProvMind full system (dual retrieval + hybrid scoring + planning)
    "provmind_full": {},
}

# Group 1 – Retrieval view decomposition
# All configs keep scoring_mode="hybrid" and use_planning=True to isolate retrieval.
# alpha=text weight, beta=struct (GAT) weight, gamma=heuristic weight.
RETRIEVAL_CONFIGS: dict[str, dict[str, Any]] = {
    # Single-view configs
    "retrieval_text_only": {
        "retrieval_mode": "text_only",
        "alpha": 1.0, "beta": 0.0, "gamma": 0.0,
    },
    "retrieval_struct_only": {
        "retrieval_mode": "dual",
        "alpha": 0.0, "beta": 1.0, "gamma": 0.0,
    },
    "retrieval_heuristic_only": {
        "retrieval_mode": "dual",
        "alpha": 0.0, "beta": 0.0, "gamma": 1.0,
    },
    # Two-view combinations (pairwise ablations of each view)
    "retrieval_text_struct": {      # text + GAT, no heuristic
        "retrieval_mode": "dual",
        "alpha": 0.5, "beta": 0.5, "gamma": 0.0,
    },
    "retrieval_text_heuristic": {   # text + heuristic, no GAT
        "retrieval_mode": "dual",
        "alpha": 0.5, "beta": 0.0, "gamma": 0.5,
    },
    "retrieval_struct_heuristic": { # GAT + heuristic, no text
        "retrieval_mode": "dual",
        "alpha": 0.0, "beta": 0.5, "gamma": 0.5,
    },
    # Full three-view (= provmind_full, repeated here for direct comparison)
    "retrieval_full_default": {
        "retrieval_mode": "dual",
        "alpha": 0.4, "beta": 0.3, "gamma": 0.3,
    },
}

# Group 2 – Scoring mode sweep
# All configs use full dual retrieval and planning to isolate scoring.
SCORING_CONFIGS: dict[str, dict[str, Any]] = {
    "score_symbolic": {
        "scoring_mode": "symbolic",
    },
    "score_neural": {
        "scoring_mode": "neural",
    },
    "score_hybrid_50_50": {
        "scoring_mode": "hybrid",
        "scoring_sym_weight": 0.5,
        "scoring_neu_weight": 0.5,
    },
    "score_hybrid_sym70_neu30": {
        "scoring_mode": "hybrid",
        "scoring_sym_weight": 0.7,
        "scoring_neu_weight": 0.3,
    },
    "score_hybrid_sym30_neu70": {
        "scoring_mode": "hybrid",
        "scoring_sym_weight": 0.3,
        "scoring_neu_weight": 0.7,
    },
}

# Group 3 – Agent module ablations
# All configs use full dual retrieval and hybrid scoring to isolate each module.
MODULE_CONFIGS: dict[str, dict[str, Any]] = {
    "no_planning": {
        "use_planning": False,
    },
    "no_symbolic_fallback": {
        "use_symbolic_fallback": False,
    },
    "no_symbolic": {
        "use_symbolic": False,
        "use_symbolic_fallback": False,
    },
    "no_planning_no_fallback": {
        "use_planning": False,
        "use_symbolic_fallback": False,
    },
}

# Group 4 – Fusion weight sweep
# Vary the relative contribution of each retrieval view.
# All configs use scoring_mode="hybrid" and planning to isolate fusion.
FUSION_CONFIGS: dict[str, dict[str, Any]] = {
    "fusion_equal": {
        "alpha": 0.33, "beta": 0.33, "gamma": 0.34,
    },
    "fusion_text_heavy": {
        "alpha": 0.6, "beta": 0.2, "gamma": 0.2,
    },
    "fusion_struct_heavy": {
        "alpha": 0.2, "beta": 0.6, "gamma": 0.2,
    },
    "fusion_heuristic_heavy": {
        "alpha": 0.2, "beta": 0.2, "gamma": 0.6,
    },
}

SUITE_MAP: dict[str, dict[str, dict[str, Any]]] = {
    "references": REFERENCE_CONFIGS,
    "retrieval":  RETRIEVAL_CONFIGS,
    "scoring":    SCORING_CONFIGS,
    "modules":    MODULE_CONFIGS,
    "fusion":     FUSION_CONFIGS,
}


# ── Evaluation helpers ────────────────────────────────────────────────────────

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_result_file(result_file: Path, model_name: str) -> dict[str, Any]:
    results, _preds, _gts, _questions = read_jsonl(str(result_file))
    mcq = compute_mcq_accuracy(results)
    metrics: dict[str, Any] = {
        "model": model_name,
        "result_file": str(result_file),
        "total": len(results),
        "MCQ_accuracy": mcq["accuracy"],
        "MCQ_correct": mcq["correct"],
        "MCQ_total": mcq["total"],
    }
    per_task: dict[str, Any] = {}
    task_buckets: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(results):
        task_buckets[(rec.get("task") or "unknown").strip() or "unknown"].append(idx)
    for task_name, indices in sorted(task_buckets.items()):
        sub = [results[j] for j in indices]
        sub_mcq = compute_mcq_accuracy(sub)
        per_task[task_name] = {
            "total": len(sub),
            "MCQ_accuracy": sub_mcq["accuracy"],
            "MCQ_correct": sub_mcq["correct"],
            "MCQ_total": sub_mcq["total"],
        }
    metrics["per_task"] = per_task
    return metrics


def compute_task_family_metrics(result_file: Path) -> dict[str, dict[str, Any]]:
    results, _p, _g, _q = read_jsonl(str(result_file))
    out: dict[str, dict[str, Any]] = {}
    for family_name, tasks in TASK_FAMILIES.items():
        subset = [r for r in results if r.get("task") in tasks]
        family_mcq = compute_mcq_accuracy(subset)
        out[family_name] = {
            "total": len(subset),
            "MCQ_accuracy": family_mcq["accuracy"],
            "MCQ_correct": family_mcq["correct"],
            "MCQ_total": family_mcq["total"],
        }
    return out


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_name",
        "agent_type",
        "MCQ_accuracy",
        "MCQ_correct",
        "MCQ_total",
        # ProvMind-specific config fields
        "retrieval_mode",
        "alpha", "beta", "gamma",
        "scoring_mode",
        "scoring_sym_weight",
        "scoring_neu_weight",
        "use_symbolic",
        "use_planning",
        "use_symbolic_fallback",
        "top_k",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cfg = row.get("config", {})
            writer.writerow({
                "experiment_name":       row.get("experiment_name", ""),
                "agent_type":            row.get("agent_type", "dualview"),
                "MCQ_accuracy":          row.get("MCQ_accuracy", 0.0),
                "MCQ_correct":           row.get("MCQ_correct", 0),
                "MCQ_total":             row.get("MCQ_total", 0),
                "retrieval_mode":        cfg.get("retrieval_mode", ""),
                "alpha":                 cfg.get("alpha", ""),
                "beta":                  cfg.get("beta", ""),
                "gamma":                 cfg.get("gamma", ""),
                "scoring_mode":          cfg.get("scoring_mode", ""),
                "scoring_sym_weight":    cfg.get("scoring_sym_weight", ""),
                "scoring_neu_weight":    cfg.get("scoring_neu_weight", ""),
                "use_symbolic":          cfg.get("use_symbolic", ""),
                "use_planning":          cfg.get("use_planning", ""),
                "use_symbolic_fallback": cfg.get("use_symbolic_fallback", ""),
                "top_k":                 cfg.get("top_k", ""),
            })


# ── Agent instantiation ───────────────────────────────────────────────────────

def _build_agent(
    agent_type: str,
    raw_file: str,
    train_file: str,
    model_name: str,
    text_encoder: str,
    cache_dir: str | None,
    kwargs: dict[str, Any],
) -> TraceFlowAgent | DualViewAgent:
    if agent_type == "traceflow":
        return TraceFlowAgent.from_files(
            raw_file=raw_file,
            train_file=train_file,
            model_name=model_name,
            top_k=kwargs["top_k"],
            plan_max_new_tokens=kwargs["plan_max_new_tokens"],
            answer_max_new_tokens=kwargs["answer_max_new_tokens"],
            load_in_4bit=kwargs["load_in_4bit"],
            use_retrieval=kwargs.get("use_retrieval", True),
            use_symbolic=kwargs["use_symbolic"],
            use_planning=kwargs["use_planning"],
            use_symbolic_fallback=kwargs["use_symbolic_fallback"],
        )
    return DualViewAgent.from_files(
        raw_file=raw_file,
        train_file=train_file,
        model_name=model_name,
        top_k=kwargs["top_k"],
        retrieval_mode=kwargs["retrieval_mode"],
        text_encoder_name=text_encoder,
        alpha=kwargs["alpha"],
        beta=kwargs["beta"],
        gamma=kwargs["gamma"],
        cache_dir=cache_dir,
        plan_max_new_tokens=kwargs["plan_max_new_tokens"],
        answer_max_new_tokens=kwargs["answer_max_new_tokens"],
        load_in_4bit=kwargs["load_in_4bit"],
        use_symbolic=kwargs["use_symbolic"],
        use_planning=kwargs["use_planning"],
        use_symbolic_fallback=kwargs["use_symbolic_fallback"],
        scoring_mode=kwargs["scoring_mode"],
        scoring_sym_weight=kwargs["scoring_sym_weight"],
        scoring_neu_weight=kwargs["scoring_neu_weight"],
    )


# ── Single configuration runner ───────────────────────────────────────────────

def run_one_configuration(
    records: list[dict[str, Any]],
    raw_file: str,
    train_file: str,
    output_dir: Path,
    model_name: str,
    text_encoder: str,
    cache_dir: str | None,
    base_kwargs: dict[str, Any],
    config_name: str,
    config_overrides: dict[str, Any],
) -> dict[str, Any]:
    config_dir = output_dir / config_name
    result_file = config_dir / "results.jsonl"
    eval_file = config_dir / "results_eval.json"
    family_file = config_dir / "task_family_eval.json"

    # Merge base kwargs with overrides (strip leading-underscore meta keys)
    kwargs = dict(base_kwargs)
    kwargs.update({k: v for k, v in config_overrides.items() if not k.startswith("_")})
    agent_type = config_overrides.get("_agent", "dualview")

    print("")
    print("=" * 70)
    print(f"[ProvMind Experiment] {config_name}  (agent={agent_type})")
    print(f"  model={model_name}")
    visible = {k: v for k, v in config_overrides.items() if not k.startswith("_")}
    print(f"  overrides={visible}")
    print("=" * 70)

    agent = _build_agent(
        agent_type=agent_type,
        raw_file=raw_file,
        train_file=train_file,
        model_name=model_name,
        text_encoder=text_encoder,
        cache_dir=cache_dir,
        kwargs=kwargs,
    )

    outputs: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        outputs.append(agent.answer_record(record))
        if idx % 25 == 0:
            print(f"[ProvMind Experiment] {config_name}: {idx}/{len(records)}")

    write_jsonl(result_file, outputs)

    metrics = evaluate_result_file(result_file, model_name=model_name)
    task_family_metrics = compute_task_family_metrics(result_file)
    metrics["experiment_name"] = config_name
    metrics["agent_type"] = agent_type
    metrics["config"] = {k: v for k, v in config_overrides.items() if not k.startswith("_")}
    metrics["task_family"] = task_family_metrics

    eval_file.parent.mkdir(parents=True, exist_ok=True)
    eval_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    family_file.write_text(json.dumps(task_family_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  MCQ accuracy: {metrics['MCQ_accuracy']:.4f}  "
          f"({metrics['MCQ_correct']}/{metrics['MCQ_total']})")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ProvMind ablation experiments on MatProcBench."
    )
    # Data paths
    parser.add_argument("--data_file",  required=True, help="Evaluation JSONL (test split).")
    parser.add_argument("--train_file", required=True, help="Training JSONL (retrieval index).")
    parser.add_argument("--raw_file",   required=True, help="Raw MatPROV JSONL.")
    parser.add_argument("--output_dir", required=True, help="Root directory for experiment outputs.")

    # Model
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Backbone LLM.")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--plan_max_new_tokens",   type=int, default=96)
    parser.add_argument("--answer_max_new_tokens", type=int, default=48)

    # Retrieval defaults (used as base for all DualViewAgent configs)
    parser.add_argument("--top_k",        type=int,   default=8)
    parser.add_argument("--text_encoder", type=str,   default="all-mpnet-base-v2")
    parser.add_argument("--alpha",        type=float, default=0.4,
                        help="Default fusion weight for text similarity.")
    parser.add_argument("--beta",         type=float, default=0.3,
                        help="Default fusion weight for structure (GAT) similarity.")
    parser.add_argument("--gamma",        type=float, default=0.3,
                        help="Default fusion weight for heuristic score.")
    parser.add_argument("--cache_dir",    type=str,   default=None,
                        help="Directory for caching encoder embeddings.")

    # Scoring defaults
    parser.add_argument("--scoring_mode",        default="hybrid",
                        choices=["symbolic", "neural", "hybrid"])
    parser.add_argument("--scoring_sym_weight",  type=float, default=0.5)
    parser.add_argument("--scoring_neu_weight",  type=float, default=0.5)

    # Suite / top-k sweep
    parser.add_argument(
        "--suite",
        default="full",
        help=(
            "Experiment suite(s) to run. "
            "Options: references | retrieval | scoring | modules | fusion | topk | full. "
            "Comma-separated to run multiple, e.g. 'references,retrieval'."
        ),
    )
    parser.add_argument("--topk_values", default="1,2,4,8,16",
                        help="Comma-separated top-k values for the topk suite.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit records for smoke tests.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = load_jsonl(args.data_file)
    if args.limit is not None:
        records = records[: args.limit]
    print(f"[ProvMind] Loaded {len(records)} test records from {args.data_file}")

    # Base kwargs — ProvMind full system defaults
    base_kwargs: dict[str, Any] = {
        "top_k":                  args.top_k,
        "retrieval_mode":         "dual",
        "alpha":                  args.alpha,
        "beta":                   args.beta,
        "gamma":                  args.gamma,
        "scoring_mode":           args.scoring_mode,
        "scoring_sym_weight":     args.scoring_sym_weight,
        "scoring_neu_weight":     args.scoring_neu_weight,
        "use_symbolic":           True,
        "use_planning":           True,
        "use_symbolic_fallback":  True,
        "use_retrieval":          True,
        "plan_max_new_tokens":    args.plan_max_new_tokens,
        "answer_max_new_tokens":  args.answer_max_new_tokens,
        "load_in_4bit":           args.load_in_4bit,
    }

    requested_suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    if "full" in requested_suites:
        requested_suites = list(SUITE_MAP.keys()) + ["topk"]

    all_metrics: list[dict[str, Any]] = []

    # ── Run ablation suites ───────────────────────────────────────────────────
    for suite_name in requested_suites:
        if suite_name == "topk":
            continue  # handled separately below
        if suite_name not in SUITE_MAP:
            print(f"[ProvMind] Unknown suite '{suite_name}', skipping.")
            continue

        suite_configs = SUITE_MAP[suite_name]
        suite_dir = output_dir / suite_name
        print(f"\n{'='*70}")
        print(f" Suite: {suite_name}  ({len(suite_configs)} configurations)")
        print(f"{'='*70}")

        for config_name, overrides in suite_configs.items():
            metrics = run_one_configuration(
                records=records,
                raw_file=args.raw_file,
                train_file=args.train_file,
                output_dir=suite_dir,
                model_name=args.model,
                text_encoder=args.text_encoder,
                cache_dir=args.cache_dir,
                base_kwargs=base_kwargs,
                config_name=config_name,
                config_overrides=overrides,
            )
            all_metrics.append(metrics)

    # ── Top-k sweep ───────────────────────────────────────────────────────────
    if "topk" in requested_suites:
        topk_values = [int(v.strip()) for v in args.topk_values.split(",") if v.strip()]
        topk_dir = output_dir / "topk"
        print(f"\n{'='*70}")
        print(f" Suite: topk  (k = {topk_values})")
        print(f"{'='*70}")
        for k in topk_values:
            metrics = run_one_configuration(
                records=records,
                raw_file=args.raw_file,
                train_file=args.train_file,
                output_dir=topk_dir,
                model_name=args.model,
                text_encoder=args.text_encoder,
                cache_dir=args.cache_dir,
                base_kwargs=base_kwargs,
                config_name=f"topk_{k}",
                config_overrides={"top_k": k},
            )
            all_metrics.append(metrics)

    # ── Save summaries ────────────────────────────────────────────────────────
    summary_json = output_dir / "summary.json"
    summary_csv  = output_dir / "summary.csv"
    summary_json.write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_csv(summary_csv, all_metrics)

    print("")
    print("=" * 70)
    print(f"[ProvMind] All experiments complete.")
    print(f"  Summary JSON : {summary_json}")
    print(f"  Summary CSV  : {summary_csv}")
    print(f"  Total configs: {len(all_metrics)}")
    print("")
    print(f"  {'Experiment':<40} {'Accuracy':>10}  {'Correct/Total':>14}")
    print(f"  {'-'*40} {'-'*10}  {'-'*14}")
    for m in all_metrics:
        acc_str = f"{m['MCQ_accuracy']:.4f}"
        ct_str  = f"{m['MCQ_correct']}/{m['MCQ_total']}"
        print(f"  {m['experiment_name']:<40} {acc_str:>10}  {ct_str:>14}")
    print("=" * 70)


if __name__ == "__main__":
    main()
