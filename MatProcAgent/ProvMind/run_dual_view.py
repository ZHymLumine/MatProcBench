"""
run_dual_view.py — CLI entry point for DualView inference on MatProcBench.

Three modes (--mode):
  dual          (default) dual-view retrieval (text + GAT + heuristic) + symbolic scoring
  encoder_only  dual-view retrieval only, NO symbolic scoring
  symbolic_only heuristic retrieval + symbolic scoring (original TraceFlow behaviour)

Output JSONL is compatible with MatProcAgent/eval.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import DEFAULT_MODEL, TraceFlowAgent
from .dual_view_agent import DualViewAgent
from .neural_scoring import ScoringMode
from .utils import load_jsonl

_MODES         = ("dual", "encoder_only", "symbolic_only")
_SCORING_MODES = ("symbolic", "neural", "hybrid")


def _write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run DualView (text encoder + 2-layer GAT + symbolic) on MatProcBench."
    )
    # ── Data paths ──────────────────────────────────────────────────────────
    p.add_argument("--data_file",  required=True, help="Evaluation JSONL (test split).")
    p.add_argument("--train_file", required=True, help="Training JSONL (retrieval index).")
    p.add_argument("--raw_file",   required=True, help="MatPROV.jsonl (raw provenance).")
    p.add_argument("--save_file",  required=True, help="Output JSONL path.")

    # ── Mode ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--mode",
        choices=_MODES,
        default="dual",
        help=(
            "dual          : dual-view retrieval + symbolic scoring (default). "
            "encoder_only  : dual-view retrieval only, no symbolic scoring. "
            "symbolic_only : heuristic retrieval + symbolic scoring (original TraceFlow)."
        ),
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    p.add_argument("--model",                default=DEFAULT_MODEL)
    p.add_argument("--plan_max_new_tokens",  type=int, default=96)
    p.add_argument("--answer_max_new_tokens", type=int, default=48)
    p.add_argument("--load_in_4bit",         action="store_true")
    p.add_argument("--disable_planning",     action="store_true")
    p.add_argument("--disable_symbolic_fallback", action="store_true")

    # ── Retrieval ────────────────────────────────────────────────────────────
    p.add_argument("--top_k",          type=int,   default=8,
                   help="Number of retrieved train processes.")
    p.add_argument("--text_encoder",   default="all-mpnet-base-v2",
                   help="SentenceTransformer model for text + node features.")
    p.add_argument("--alpha",          type=float, default=0.4,
                   help="Fusion weight for text similarity.")
    p.add_argument("--beta",           type=float, default=0.3,
                   help="Fusion weight for structure similarity (GAT).")
    p.add_argument("--gamma",          type=float, default=0.3,
                   help="Fusion weight for heuristic score.")

    # ── GAT ─────────────────────────────────────────────────────────────────
    p.add_argument("--gat_hidden_dim", type=int, default=256)
    p.add_argument("--gat_heads",      type=int, default=4)
    p.add_argument("--gat_out_dim",    type=int, default=256)

    # ── Cache ────────────────────────────────────────────────────────────────
    p.add_argument("--cache_dir", default=None,
                   help="Directory for caching encoder embeddings (speeds up re-runs).")

    # ── Scoring ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--scoring_mode",
        choices=_SCORING_MODES,
        default="hybrid",
        help=(
            "symbolic : original string-matching scorer. "
            "neural   : embedding NN over step/process embeddings. "
            "hybrid   : weighted blend of symbolic + neural (default)."
        ),
    )
    p.add_argument("--scoring_sym_weight", type=float, default=0.5,
                   help="Symbolic weight in hybrid scoring (default 0.5).")
    p.add_argument("--scoring_neu_weight", type=float, default=0.5,
                   help="Neural weight in hybrid scoring (default 0.5).")

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N records (smoke test).")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    print("=" * 70)
    print(f" DualView  |  mode={args.mode}  |  model={args.model}")
    if args.mode not in ("symbolic_only", "encoder_only"):
        print(f" scoring   = {args.scoring_mode}"
              + (f"  (sym={args.scoring_sym_weight}, neu={args.scoring_neu_weight})"
                 if args.scoring_mode == "hybrid" else ""))
    print(f" train={args.train_file}")
    print(f" data ={args.data_file}")
    print(f" save ={args.save_file}")
    print("=" * 70)

    # ── Build agent ──────────────────────────────────────────────────────────
    if args.mode == "symbolic_only":
        # Original TraceFlow — no neural encoder needed
        print("[DualView] Mode=symbolic_only → building standard TraceFlow agent.")
        agent = TraceFlowAgent.from_files(
            raw_file=args.raw_file,
            train_file=args.train_file,
            model_name=args.model,
            top_k=args.top_k,
            plan_max_new_tokens=args.plan_max_new_tokens,
            answer_max_new_tokens=args.answer_max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            use_retrieval=True,
            use_symbolic=True,
            use_planning=not args.disable_planning,
            use_symbolic_fallback=not args.disable_symbolic_fallback,
        )
    else:
        retrieval_mode = "dual"          # both "dual" and "encoder_only" use dual-view retrieval
        use_symbolic   = (args.mode != "encoder_only")
        # encoder_only disables all scoring; otherwise use the requested scoring_mode
        scoring_mode: ScoringMode = args.scoring_mode if use_symbolic else "symbolic"
        print(f"[DualView] retrieval_mode={retrieval_mode}  use_symbolic={use_symbolic}  "
              f"scoring_mode={scoring_mode}")
        agent = DualViewAgent.from_files(
            raw_file=args.raw_file,
            train_file=args.train_file,
            model_name=args.model,
            top_k=args.top_k,
            retrieval_mode=retrieval_mode,
            text_encoder_name=args.text_encoder,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            gat_hidden_dim=args.gat_hidden_dim,
            gat_heads=args.gat_heads,
            gat_out_dim=args.gat_out_dim,
            cache_dir=args.cache_dir,
            plan_max_new_tokens=args.plan_max_new_tokens,
            answer_max_new_tokens=args.answer_max_new_tokens,
            load_in_4bit=args.load_in_4bit,
            use_symbolic=use_symbolic,
            use_planning=not args.disable_planning,
            use_symbolic_fallback=not args.disable_symbolic_fallback,
            scoring_mode=scoring_mode,
            scoring_sym_weight=args.scoring_sym_weight,
            scoring_neu_weight=args.scoring_neu_weight,
        )

    # ── Run inference ────────────────────────────────────────────────────────
    print(f"\n[DualView] Loading benchmark records from {args.data_file}")
    records = load_jsonl(args.data_file)
    if args.limit is not None:
        records = records[: args.limit]
        print(f"[DualView] Limiting to first {args.limit} records.")

    outputs: list[dict] = []
    for idx, record in enumerate(records, start=1):
        outputs.append(agent.answer_record(record))
        if idx % 25 == 0:
            print(f"[DualView] Processed {idx} / {len(records)} records.")

    _write_jsonl(args.save_file, outputs)
    print(f"\n[DualView] Done. Wrote {len(outputs)} records → {args.save_file}")


if __name__ == "__main__":
    main()
