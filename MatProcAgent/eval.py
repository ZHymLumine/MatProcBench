"""
Unified evaluation script for material synthesis QA results.

Supports two modes:
  - free : EM, BLEU, Rouge-1/2/L/Lsum  (default)
  - mcq  : Accuracy (letter-match) + EM/BLEU/Rouge on gt_answer text

Input: a JSONL file where each line has at least:
  - model_answer  (str)
  - gt_answer     (str)
  - question      (str)

Usage:
  python eval.py --result_file results.jsonl --model Qwen2.5-7B
  python eval.py --result_file results_mcq.jsonl --model Qwen2.5-7B --mode mcq
"""

import argparse
import json
import re
import string
from collections import defaultdict
from typing import Dict, List, Any

import evaluate as hf_evaluate


# ── IO ────────────────────────────────────────────────────────────────────────

def read_jsonl(path: str):
    results, preds, gts, questions = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            results.append(rec)
            preds.append(rec.get("model_answer") or "")
            gts.append(rec.get("gt_answer") or "")
            questions.append(rec.get("question", ""))
    return results, preds, gts, questions


# ── Metric wrappers ──────────────────────────────────────────────────────────

def compute_exact_match(predictions, references):
    metric = hf_evaluate.load("exact_match")
    return metric.compute(predictions=predictions, references=references)


def compute_bleu(predictions, references):
    metric = hf_evaluate.load("bleu")
    # HF BLEU may raise ZeroDivisionError when all references are empty.
    preds = ["" if p is None else str(p) for p in predictions]
    refs = ["" if r is None else str(r) for r in references]
    if not refs or all(not r.strip() for r in refs):
        return {
            "bleu": 0.0,
            "precisions": [0.0, 0.0, 0.0, 0.0],
            "brevity_penalty": 0.0,
            "length_ratio": 0.0,
            "translation_length": 0,
            "reference_length": 0,
        }
    try:
        return metric.compute(predictions=preds, references=refs)
    except ZeroDivisionError:
        return {
            "bleu": 0.0,
            "precisions": [0.0, 0.0, 0.0, 0.0],
            "brevity_penalty": 0.0,
            "length_ratio": 0.0,
            "translation_length": 0,
            "reference_length": 0,
        }


def compute_rouge(predictions, references):
    metric = hf_evaluate.load("rouge")
    return metric.compute(predictions=predictions, references=references)


# ── MCQ accuracy ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    Lowercase and collapse whitespace for fuzzy text matching.

    Punctuation handling: keep characters that carry numeric/scientific meaning
    (decimal point, percent sign, comparison operators, plus/minus, slash) so
    that values like '99.9 %', '>99.99 %', and '≥98.0 %' remain distinct after
    normalization.  Strip all other punctuation.
    """
    keep = set("0123456789abcdefghijklmnopqrstuvwxyz .%><=+-/")
    text = text.lower().strip()
    text = "".join(ch if ch in keep else " " for ch in text)
    return " ".join(text.split())


_MCQ_PHRASE_PATTERNS = [
    re.compile(r'the\s+answer\s+is\s+([A-D])\b', re.IGNORECASE),
    re.compile(r'\bselect\s+([A-D])\b', re.IGNORECASE),
    re.compile(r'\bchoose\s+([A-D])\b', re.IGNORECASE),
    re.compile(r'\boption\s+([A-D])\b', re.IGNORECASE),
    re.compile(r'\banswer[:\s]+([A-D])\b', re.IGNORECASE),
    re.compile(r'\b([A-D])\s+is\s+(?:correct|right)\b', re.IGNORECASE),
]


def _choices_text_match(raw: str, choices: list) -> str:
    """
    Match normalized prediction text against choice options.

    Strategy (in order):
      1. Exact normalized match (highest confidence).
      2. The prediction is fully contained within a choice text (pred ⊆ choice).
      3. A choice text is a substring of the prediction, but only if that
         choice covers >= 60% of the prediction length (length guard).
    Returns the option letter of the best match, or "" if none found.
    """
    if not choices:
        return ""
    norm_pred = _normalize(raw)
    if not norm_pred:
        return ""

    # Pass 1: exact match
    for idx, choice in enumerate(choices):
        if _normalize(str(choice)) == norm_pred:
            return chr(65 + idx)

    # Pass 2: prediction is a substring of a choice.
    # The extra prefix/suffix in the choice must be non-alphanumeric AND must not
    # be a numeric comparison operator (>, >=, <, <=, !, ~) which would change
    # the meaning of the value (e.g., ">99.99 %" ≠ "99.99 %").
    _alnum = re.compile(r'[a-z0-9]', re.IGNORECASE)
    _numeric_op = re.compile(r'^[><=!~]+$')
    best_letter, best_score = "", 0
    for idx, choice in enumerate(choices):
        norm_choice = _normalize(str(choice))
        if not norm_choice or norm_pred not in norm_choice:
            continue
        pos = norm_choice.find(norm_pred)
        prefix = norm_choice[:pos].strip()
        suffix = norm_choice[pos + len(norm_pred):].strip()
        # Reject if extra chars are purely numeric operators or contain alnum
        if _alnum.search(prefix) or _alnum.search(suffix):
            continue
        if _numeric_op.match(prefix) if prefix else False:
            continue
        if len(norm_pred) > best_score:
            best_score = len(norm_pred)
            best_letter = chr(65 + idx)
    if best_letter:
        return best_letter

    # Pass 3: choice is a substring of prediction, with length guard
    best_letter, best_score = "", 0
    for idx, choice in enumerate(choices):
        norm_choice = _normalize(str(choice))
        if not norm_choice:
            continue
        if norm_choice in norm_pred and len(norm_choice) > best_score:
            if len(norm_choice) >= max(3, int(len(norm_pred) * 0.6)):
                best_score = len(norm_choice)
                best_letter = chr(65 + idx)
    return best_letter


def extract_mcq_letter(text, choices: list = None) -> str:
    """
    Robustly extract the predicted option letter (A-D) from model output.

    Priority order:
      1. Leading letter token: "A", "B.", "C)", "D:" at string start
      2. Explicit phrasing: "the answer is X", "option X", "choose X", etc.
      3. Choices text match (exact / containment / long-substring)
      4. Isolated single letter (only when full text is essentially one letter)
      5. Return "" if none of the above succeeds
    """
    raw = "" if text is None else str(text)
    raw = raw.strip()

    # 1. Leading letter token
    m = re.match(r'^([A-D])[\s\.\)\:\,\-]', raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.match(r'^([A-D])$', raw, re.IGNORECASE):
        return raw[0].upper()

    # 2. Explicit natural-language phrasing (applied to raw text, case-insensitive)
    for pat in _MCQ_PHRASE_PATTERNS:
        m = pat.search(raw)
        if m:
            return m.group(1).upper()

    # 3. Choices text match (before isolated-letter fallback to avoid false hits)
    letter = _choices_text_match(raw, choices)
    if letter:
        return letter

    # 4. Isolated letter (only if entire prediction reduces to a single A-D letter)
    alpha_only = re.sub(r'[^A-Za-z]', '', raw)
    if len(alpha_only) == 1 and alpha_only.upper() in "ABCD":
        return alpha_only.upper()

    return ""


def _gold_letter(rec: dict) -> str:
    """Derive the gold option letter from a record."""
    gold = (rec.get("gt_answer") or "").strip()
    if len(gold) == 1 and gold.upper() in "ABCD":
        return gold.upper()
    if "answer_index" in rec and isinstance(rec["answer_index"], int) and rec["answer_index"] >= 0:
        return chr(65 + rec["answer_index"])
    # Last resort: try to match gold text against choices
    choices = rec.get("choices", [])
    if choices:
        norm_gold = _normalize(gold)
        for idx, choice in enumerate(choices):
            if _normalize(str(choice)) == norm_gold:
                return chr(65 + idx)
    return gold.upper()


def _to_index(value):
    """Convert model output to option index when possible.

    Accepts:
      - int/float (e.g., 2, 2.0) -> 2
      - numeric strings (e.g., "2", "2.0") -> 2
    Returns None if conversion is not possible.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        f = float(s)
    except Exception:
        return None
    if f.is_integer():
        return int(f)
    return None


def compute_mcq_accuracy(records: List[dict]) -> Dict[str, float]:
    """Compute accuracy by matching predicted letter to gold letter."""
    correct, total = 0, len(records)
    for rec in records:
        # 1) Numeric prediction path: compare directly with answer_index.
        pred_idx = _to_index(rec.get("model_answer"))
        if pred_idx is not None and isinstance(rec.get("answer_index"), int):
            gold_idx = rec["answer_index"]
            # Support both 0-based (A=0) and 1-based (A=1) model outputs.
            if pred_idx == gold_idx or pred_idx - 1 == gold_idx:
                correct += 1
                continue

        # 2) Text prediction path: parse predicted option letter.
        choices = rec.get("choices") or []
        pred_letter = extract_mcq_letter(rec.get("model_answer"), choices)
        gold_letter = _gold_letter(rec)
        if pred_letter and pred_letter == gold_letter:
            correct += 1
    acc = correct / total if total else 0.0
    return {"accuracy": acc, "correct": correct, "total": total}


def compute_metrics_for_records(
    results: List[dict],
    preds: List[str],
    gts: List[str],
    is_mcq: bool,
) -> Dict[str, Any]:
    """Compute EM, Bleu, Rouge (and MCQ accuracy if is_mcq) for given lists."""
    n = len(results)
    if n == 0:
        return {"total": 0}
    em_score = compute_exact_match(preds, gts)
    bleu_score = compute_bleu(preds, gts)
    rouge_score = compute_rouge(preds, gts)
    out = {
        "total":    n,
        "EM":       em_score["exact_match"],
        "Bleu":     bleu_score["bleu"],
        "Rouge1":   rouge_score["rouge1"],
        "Rouge2":   rouge_score["rouge2"],
        "RougeL":   rouge_score["rougeL"],
        "RougeLSum": rouge_score["rougeLsum"],
    }
    if is_mcq:
        m = compute_mcq_accuracy(results)
        out["MCQ_accuracy"] = m["accuracy"]
        out["MCQ_correct"] = m["correct"]
        out["MCQ_total"]   = m["total"]
    return out


def compute_per_task_metrics(
    results: List[dict],
    preds: List[str],
    gts: List[str],
    is_mcq: bool,
) -> Dict[str, Dict[str, Any]]:
    """Group records by 'task' and compute metrics for each subtask."""
    groups = defaultdict(list)  # task_name -> list of indices
    for i, rec in enumerate(results):
        task_name = (rec.get("task") or "unknown").strip() or "unknown"
        groups[task_name].append(i)
    per_task = {}
    for task_name in sorted(groups.keys()):
        indices = groups[task_name]
        sub_results = [results[j] for j in indices]
        sub_preds   = [preds[j] for j in indices]
        sub_gts     = [gts[j] for j in indices]
        per_task[task_name] = compute_metrics_for_records(
            sub_results, sub_preds, sub_gts, is_mcq
        )
    return per_task


def compute_per_difficulty_metrics(
    results: List[dict],
    preds: List[str],
    gts: List[str],
    is_mcq: bool,
) -> Dict[str, Dict[str, Any]]:
    """Group records by 'difficulty' and compute metrics for each level.

    Supports the MatProcBench difficulty field (easy / medium / hard).
    Records lacking a difficulty field are grouped under 'unknown'.
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(results):
        difficulty = (rec.get("difficulty") or "unknown").strip() or "unknown"
        groups[difficulty].append(i)
    per_diff: Dict[str, Dict[str, Any]] = {}
    for difficulty in sorted(groups.keys()):
        indices = groups[difficulty]
        sub_results = [results[j] for j in indices]
        sub_preds   = [preds[j]   for j in indices]
        sub_gts     = [gts[j]     for j in indices]
        per_diff[difficulty] = compute_metrics_for_records(
            sub_results, sub_preds, sub_gts, is_mcq
        )
    return per_diff


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Unified evaluation for material synthesis QA")
    parser.add_argument("--result_file", type=str, required=True,
                        help="Path to .jsonl result file")
    parser.add_argument("--model", type=str, default="unknown",
                        help="Model name (for display only)")
    parser.add_argument("--mode", type=str, choices=["free", "mcq", "auto"],
                        default="auto",
                        help="Evaluation mode: free | mcq | auto (default: auto)")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Path to save metrics as JSON (e.g. results_eval.json)")
    parser.add_argument("--per_task", action="store_true",
                        help="Print per-task metrics to console")
    parser.add_argument("--per_difficulty", action="store_true",
                        help="Print per-difficulty metrics to console (easy/medium/hard)")
    args = parser.parse_args()

    results, preds, gts, questions = read_jsonl(args.result_file)
    if not results:
        print("No records found.")
        return

    # Auto-detect mode
    if args.mode == "auto":
        args.mode = "mcq" if "choices" in results[0] else "free"

    preds = [p if p is not None else "" for p in preds]
    gts   = [g if g is not None else "" for g in gts]
    is_mcq = args.mode == "mcq"

    # Text-based metrics (always computed)
    em_score    = compute_exact_match(preds, gts)
    bleu_score  = compute_bleu(preds, gts)
    rouge_score = compute_rouge(preds, gts)

    # Build metrics dict for saving
    metrics = {
        "model":       args.model,
        "mode":        args.mode,
        "result_file": args.result_file,
        "total":       len(results),
        "EM":          em_score["exact_match"],
        "Bleu":        bleu_score["bleu"],
        "Rouge1":      rouge_score["rouge1"],
        "Rouge2":      rouge_score["rouge2"],
        "RougeL":      rouge_score["rougeL"],
        "RougeLSum":   rouge_score["rougeLsum"],
    }
    if is_mcq:
        mcq = compute_mcq_accuracy(results)
        metrics["MCQ_accuracy"] = mcq["accuracy"]
        metrics["MCQ_correct"] = mcq["correct"]
        metrics["MCQ_total"]   = mcq["total"]

    # Per-task metrics (grouped by field "task")
    per_task = compute_per_task_metrics(results, preds, gts, is_mcq)
    metrics["per_task"] = per_task

    # Per-difficulty metrics (grouped by field "difficulty")
    per_difficulty = compute_per_difficulty_metrics(results, preds, gts, is_mcq)
    metrics["per_difficulty"] = per_difficulty

    line = (
        f"{args.model} || "
        f"EM: {metrics['EM']:.4f} | "
        f"Bleu: {metrics['Bleu']:.4f} | "
        f"Rouge1: {metrics['Rouge1']:.4f} | "
        f"Rouge2: {metrics['Rouge2']:.4f} | "
        f"RougeL: {metrics['RougeL']:.4f} | "
        f"RougeLSum: {metrics['RougeLSum']:.4f}"
    )
    if is_mcq:
        line += f" | MCQ-Acc: {metrics['MCQ_accuracy']:.4f} ({metrics['MCQ_correct']}/{metrics['MCQ_total']})"

    print(line)

    if args.per_task and per_task:
        print("\n--- Per-task metrics ---")
        for task_name in sorted(per_task.keys()):
            t = per_task[task_name]
            if t.get("total", 0) == 0:
                continue
            row = f"  {task_name}: n={t['total']} | EM={t['EM']:.4f} | RougeL={t['RougeL']:.4f}"
            if is_mcq and "MCQ_accuracy" in t:
                row += f" | Acc={t['MCQ_accuracy']:.4f}"
            print(row)
        print("")

    if args.per_difficulty and per_difficulty:
        print("\n--- Per-difficulty metrics ---")
        for diff_level in ["easy", "medium", "hard", "unknown"]:
            d = per_difficulty.get(diff_level)
            if d is None or d.get("total", 0) == 0:
                continue
            row = f"  {diff_level}: n={d['total']} | EM={d['EM']:.4f} | RougeL={d['RougeL']:.4f}"
            if is_mcq and "MCQ_accuracy" in d:
                row += f" | Acc={d['MCQ_accuracy']:.4f}"
            print(row)
        print("")

    if args.save_json:
        import os
        d = os.path.dirname(os.path.abspath(args.save_json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {args.save_json}")


if __name__ == "__main__":
    main()
