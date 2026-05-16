"""
eval_cross_split.py – Cross-Split Evaluation for MatProcBench SFT Models
=========================================================================

Evaluates a fine-tuned model checkpoint across multiple test splits in one
pass to measure OOD generalisation under three types of distribution shift:

  additional_split → paper-level OOD   (unseen papers)
  type_split       → material-type OOD (novel material chemistries)
  year_split       → temporal OOD      (newer publications)

Clean cross-split pairs (0% DOI contamination):
  Train: additional → Test: additional, type, year
  Train: type       → Test: additional, type
  Train: year       → Test: additional, year

Evaluation modes
----------------
  zero_shot      Model sees the question only (no evidence/context)
  rag            Model gets top-k evidence retrieved from the train corpus

Usage
-----
  # Evaluate additional_split model across all OOD splits (zero-shot)
  python eval_cross_split.py \\
      --checkpoint checkpoints/sft_additional/best_checkpoint \\
      --train_split additional_split \\
      --test_splits additional_split,type_split,year_split \\
      --eval_mode zero_shot \\
      --output_dir results/cross_split/additional

  # RAG mode (retrieves from train corpus)
  python eval_cross_split.py \\
      --checkpoint checkpoints/sft_additional/best_checkpoint \\
      --train_split additional_split \\
      --test_splits additional_split,type_split,year_split \\
      --eval_mode rag \\
      --rag_k 3 \\
      --output_dir results/cross_split/additional

  # Run both modes sequentially
  python eval_cross_split.py \\
      --checkpoint checkpoints/sft_additional/best_checkpoint \\
      --train_split additional_split \\
      --test_splits additional_split,type_split,year_split \\
      --eval_mode all \\
      --output_dir results/cross_split/additional

Dependencies
------------
  pip install peft transformers sentence-transformers faiss-cpu tqdm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse prompt-building and evidence formatting from train_sft.py
sys.path.insert(0, str(Path(__file__).parent))
from train_sft import build_training_prompt, load_jsonl, format_evidence, format_choices_block

# ── Contamination table (computed from data; hard-coded for fast startup) ─────
# Value: fraction of test DOIs that appear in train DOIs
# Source: cross-split DOI analysis (see analysis in research notes)
_CONTAMINATION: dict[tuple[str, str], float] = {
    ("additional_split", "additional_split"): 0.00,
    ("additional_split", "type_split"):       0.00,
    ("additional_split", "year_split"):        0.00,
    ("additional_split", "random_split"):      0.72,
    ("type_split",       "additional_split"): 0.00,
    ("type_split",       "type_split"):        0.00,
    ("type_split",       "year_split"):        0.27,
    ("type_split",       "random_split"):      0.91,
    ("year_split",       "additional_split"): 0.00,
    ("year_split",       "type_split"):        0.27,
    ("year_split",       "year_split"):        0.00,
    ("year_split",       "random_split"):      0.74,
    ("random_split",     "additional_split"): 0.99,
    ("random_split",     "type_split"):        0.99,
    ("random_split",     "year_split"):        0.99,
    ("random_split",     "random_split"):      1.00,
}

_WARN_THRESHOLD = 0.05  # warn if >5% of test DOIs were seen in train


def _contamination_warning(train_split: str, test_split: str) -> str | None:
    pct = _CONTAMINATION.get((train_split, test_split))
    if pct is None or pct <= _WARN_THRESHOLD:
        return None
    return (f"WARNING: {pct*100:.0f}% of {test_split}.test DOIs appear in "
            f"{train_split}.train — results are NOT OOD-clean.")


# ── MCQ accuracy helpers (standalone, no HF evaluate dependency) ──────────────

def _extract_letter(text: str) -> str:
    text = (text or "").strip()
    m = re.match(r'^([A-D])[\s\.\)\:\,\-]', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.match(r'^([A-D])$', text, re.IGNORECASE):
        return text[0].upper()
    for pat in [
        r'the\s+answer\s+is\s+([A-D])\b',
        r'\banswer[:\s]+([A-D])\b',
        r'\boption\s+([A-D])\b',
        r'\b([A-D])\s+is\s+(?:correct|right)\b',
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(r'\b([A-D])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ""


def _compute_accuracy(records: list[dict]) -> dict:
    correct = sum(
        1 for r in records
        if r.get("model_answer", "") == r.get("gt_answer", "")
    )
    total = len(records)
    return {"accuracy": correct / total if total else 0.0,
            "correct": correct, "total": total}


def _per_task_accuracy(records: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r.get("task", "unknown")].append(r)
    return {
        task: _compute_accuracy(recs)
        for task, recs in sorted(groups.items())
    }


# ── Inference engines ─────────────────────────────────────────────────────────

def _run_zero_shot(
    test_records: list[dict],
    model,
    tokenizer,
    batch_size: int,
) -> list[dict]:
    """Run inference with question only (no evidence injected)."""
    results = []
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(test_records), batch_size), desc="  zero_shot"):
        batch = test_records[i: i + batch_size]
        # Build prompts without evidence (blank out evidence field)
        prompts = []
        for r in batch:
            r_no_ev = {**r, "evidence": {}}
            prompts.append(build_training_prompt(r_no_ev, tokenizer, include_answer=False))

        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        input_len = enc["input_ids"].shape[1]
        for j, rec in enumerate(batch):
            raw = tokenizer.decode(out[j][input_len:], skip_special_tokens=True).strip()
            pred = _extract_letter(raw)
            gt = rec.get("answer", "")
            results.append({
                "qid":          rec.get("qid", ""),
                "task":         rec.get("task", ""),
                "eval_mode":    "zero_shot",
                "model_answer": pred,
                "gt_answer":    gt,
                "choices":      [rec["choices"].get(l, "") for l in "ABCD"],
                "answer_index": ord(gt) - ord("A") if gt in "ABCD" else -1,
            })
    return results



def _run_rag(
    test_records: list[dict],
    train_records: list[dict],
    model,
    tokenizer,
    batch_size: int,
    rag_k: int,
    embedder_name: str,
    emb_cache: str | None,
) -> list[dict]:
    """
    Run inference with RAG-retrieved evidence from the train corpus.
    Retrieved records' evidence is formatted and prepended to the question.
    """
    from retriever import MatProcRetriever

    _SYS_EVIDENCE = (
        "You are an expert assistant in materials science and materials synthesis. "
        "You are provided with factual synthesis process information relevant to the question. "
        "Use this information to answer the multiple-choice question. "
        "Output ONLY the letter (A, B, C, or D) of the correct option. "
        "Do not include any explanation or extra text."
    )
    _ANSWER_PROMPT = "\n\nAnswer with only the letter (A, B, C, or D):"

    print(f"  [rag] Building retrieval index over {len(train_records)} train records ...")
    retriever = MatProcRetriever(
        records=train_records,
        embedder_name=embedder_name,
        cache_path=emb_cache,
    )

    results = []
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(test_records), batch_size), desc="  rag"):
        batch = test_records[i: i + batch_size]
        prompts = []
        for rec in batch:
            # Retrieve top-k records by question similarity (task-filtered)
            hits = retriever.search_by_task(
                rec["question"], task=rec.get("task", ""), k=rag_k
            )
            # Format retrieved evidence blocks
            ev_parts = []
            for hit in hits:
                ev_text = format_evidence(hit.get("task", ""), hit.get("evidence", {}))
                if ev_text:
                    ev_parts.append(ev_text)
            ev_block = "\n\n---\n\n".join(ev_parts) if ev_parts else ""

            q_block = (f"{rec['question']}\n\nOptions:\n"
                       f"{format_choices_block(rec['choices'])}")
            if ev_block:
                user_content = (f"Retrieved synthesis process information:\n\n{ev_block}"
                                f"\n\n---\n\n{q_block}{_ANSWER_PROMPT}")
            else:
                user_content = f"{q_block}{_ANSWER_PROMPT}"

            messages = [
                {"role": "system", "content": _SYS_EVIDENCE},
                {"role": "user",   "content": user_content},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        input_len = enc["input_ids"].shape[1]
        for j, rec in enumerate(batch):
            raw = tokenizer.decode(out[j][input_len:], skip_special_tokens=True).strip()
            pred = _extract_letter(raw)
            gt = rec.get("answer", "")
            results.append({
                "qid":          rec.get("qid", ""),
                "task":         rec.get("task", ""),
                "eval_mode":    "rag",
                "model_answer": pred,
                "gt_answer":    gt,
                "choices":      [rec["choices"].get(l, "") for l in "ABCD"],
                "answer_index": ord(gt) - ord("A") if gt in "ABCD" else -1,
            })
    return results


# ── ASCII result table ────────────────────────────────────────────────────────

TASKS = [
    "A1_route_retrieval",
    "A2_missing_step",
    "A3_next_activity",
    "B1_condition_prediction",
    "B2_full_condition_set",
    "C1_tool_selection",
    "D1_process_ordering",
]


def _print_table(matrix: dict[str, dict], eval_mode: str, train_split: str) -> None:
    test_splits = list(matrix.keys())
    col_w = 10

    header_label = f"Train: {train_split} | Mode: {eval_mode}"
    print()
    print("=" * (32 + col_w * len(test_splits)))
    print(header_label)
    print("=" * (32 + col_w * len(test_splits)))

    # Header row
    print(f"{'Task':<32}", end="")
    for ts in test_splits:
        label = ts.replace("_split", "")
        print(f"{label:>{col_w}}", end="")
    print()
    print("-" * (32 + col_w * len(test_splits)))

    # Per-task rows
    for task in TASKS:
        print(f"{task:<32}", end="")
        for ts in test_splits:
            per_task = matrix[ts].get("per_task", {}).get(task, {})
            acc = per_task.get("accuracy")
            n = per_task.get("total", 0)
            if acc is not None and n > 0:
                print(f"{acc:>{col_w-2}.3f} ({n:3d})"[:col_w], end="")
                print(f"{acc:>{col_w}.3f}", end="")
            else:
                print(f"{'—':>{col_w}}", end="")
        print()

    # Overall row
    print("-" * (32 + col_w * len(test_splits)))
    print(f"{'OVERALL':<32}", end="")
    for ts in test_splits:
        acc = matrix[ts].get("accuracy")
        n = matrix[ts].get("total", 0)
        if acc is not None:
            print(f"{acc:>{col_w}.3f}", end="")
        else:
            print(f"{'—':>{col_w}}", end="")
    print()
    print()


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_cross_split(args: argparse.Namespace) -> None:
    processed_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    test_splits = [s.strip() for s in args.test_splits.split(",") if s.strip()]
    eval_modes = (
        ["zero_shot", "rag"]
        if args.eval_mode == "all"
        else [args.eval_mode]
    )

    # ── Print contamination warnings ─────────────────────────────────────────
    print()
    print("=== Contamination check ===")
    for ts in test_splits:
        warn = _contamination_warning(args.train_split, ts)
        if warn:
            print(f"  [!] {warn}")
        else:
            pct = _CONTAMINATION.get((args.train_split, ts), None)
            pct_str = f"{pct*100:.0f}%" if pct is not None else "unknown"
            print(f"  [ok] {args.train_split}.train → {ts}.test: {pct_str} contamination")
    print()

    # ── Load model (once for all modes) ──────────────────────────────────────
    print(f"[cross_eval] Loading tokenizer: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[cross_eval] Loading model: {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = True
    model.eval()

    # ── Load train corpus (needed for RAG) ───────────────────────────────────
    train_records: list[dict] = []
    if "rag" in eval_modes:
        train_file = processed_root / args.train_split / "train.jsonl"
        print(f"[cross_eval] Loading train corpus for RAG: {train_file}")
        train_records = load_jsonl(str(train_file))
        print(f"[cross_eval] Train corpus size: {len(train_records)}")

    # ── Evaluate per mode, per test split ────────────────────────────────────
    all_results: dict[str, dict[str, dict]] = {mode: {} for mode in eval_modes}

    for eval_mode in eval_modes:
        print(f"\n{'='*60}")
        print(f" Eval mode: {eval_mode}")
        print(f"{'='*60}")

        emb_cache_dir = output_root / "rag_cache"
        emb_cache_dir.mkdir(exist_ok=True)

        for test_split in test_splits:
            test_file = processed_root / test_split / "test.jsonl"
            if not test_file.exists():
                print(f"  [skip] {test_file} not found")
                continue

            print(f"\n  Test split: {test_split}  ({test_file})")
            test_records = load_jsonl(str(test_file))
            print(f"  Records: {len(test_records)}")

            # Run inference
            if eval_mode == "zero_shot":
                results = _run_zero_shot(
                    test_records, model, tokenizer, args.batch_size
                )
            elif eval_mode == "rag":
                emb_cache = str(
                    emb_cache_dir / f"{args.train_split}_emb.pkl"
                )
                results = _run_rag(
                    test_records, train_records, model, tokenizer,
                    args.batch_size, args.rag_k, args.embedder, emb_cache,
                )
            else:
                raise ValueError(f"Unknown eval_mode: {eval_mode}")

            # Compute metrics
            overall = _compute_accuracy(results)
            per_task = _per_task_accuracy(results)

            all_results[eval_mode][test_split] = {
                **overall,
                "per_task": per_task,
                "train_split": args.train_split,
                "test_split":  test_split,
                "eval_mode":   eval_mode,
                "contamination": _CONTAMINATION.get(
                    (args.train_split, test_split), None
                ),
            }

            # Save per-(mode, split) JSONL
            split_out_dir = output_root / eval_mode / test_split
            split_out_dir.mkdir(parents=True, exist_ok=True)
            result_file = split_out_dir / "results.jsonl"
            with open(result_file, "w") as fh:
                for r in results:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

            print(f"  Overall accuracy: {overall['accuracy']:.4f} "
                  f"({overall['correct']}/{overall['total']})")
            print(f"  Results saved to: {result_file}")

    # ── Print summary tables ──────────────────────────────────────────────────
    for eval_mode in eval_modes:
        if all_results[eval_mode]:
            _print_table(all_results[eval_mode], eval_mode, args.train_split)

    # ── Save full summary JSON ────────────────────────────────────────────────
    summary = {
        "checkpoint":  args.checkpoint,
        "train_split": args.train_split,
        "test_splits": test_splits,
        "eval_modes":  eval_modes,
        "results":     all_results,
    }
    summary_path = output_root / "cross_split_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[cross_eval] Full summary saved to: {summary_path}")

    # ── Print compact cross-split accuracy table ──────────────────────────────
    print()
    print("=" * 70)
    print(" CROSS-SPLIT SUMMARY  (overall MCQ accuracy)")
    print(f" Checkpoint: {args.checkpoint}")
    print("=" * 70)
    col_w = 14
    print(f"{'mode':<18}", end="")
    for ts in test_splits:
        print(f"{ts.replace('_split',''):>{col_w}}", end="")
    print()
    print("-" * (18 + col_w * len(test_splits)))
    for mode in eval_modes:
        print(f"{mode:<18}", end="")
        for ts in test_splits:
            acc = all_results[mode].get(ts, {}).get("accuracy")
            if acc is not None:
                print(f"{acc:>{col_w}.4f}", end="")
            else:
                print(f"{'—':>{col_w}}", end="")
        print()
    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-split OOD evaluation for MatProcBench SFT models"
    )
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to fine-tuned model checkpoint.")
    parser.add_argument("--train_split",  type=str, required=True,
                        choices=["additional_split", "type_split",
                                 "year_split", "random_split"],
                        help="Split used during training (for contamination check and RAG corpus).")
    parser.add_argument("--test_splits",  type=str,
                        default="additional_split,type_split,year_split",
                        help="Comma-separated list of splits to evaluate on.")
    parser.add_argument("--eval_mode",    type=str,
                        default="zero_shot",
                        choices=["zero_shot", "rag", "all"],
                        help="Inference mode. 'all' runs zero_shot and rag sequentially.")
    parser.add_argument("--data_root",    type=str,
                        default=None,
                        help="Root directory of processed data splits. "
                             "Auto-detected relative to this script if not set.")
    parser.add_argument("--output_dir",  type=str, required=True,
                        help="Directory to save results and summary JSON.")
    parser.add_argument("--batch_size",  type=int, default=8,
                        help="Inference batch size.")
    # RAG-specific
    parser.add_argument("--rag_k",       type=int, default=3,
                        help="Number of records to retrieve (RAG mode only).")
    parser.add_argument("--embedder",    type=str, default="all-mpnet-base-v2",
                        help="Sentence-transformer embedder for RAG retrieval.")

    args = parser.parse_args()

    # Auto-detect data root relative to this script
    if args.data_root is None:
        script_dir = Path(__file__).parent
        args.data_root = str(script_dir.parent.parent / "data" / "processed")

    if not Path(args.data_root).exists():
        sys.exit(f"Data root not found: {args.data_root}\n"
                 "Pass --data_root explicitly.")

    evaluate_cross_split(args)


if __name__ == "__main__":
    main()
