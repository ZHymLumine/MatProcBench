"""
run_vllm_zeroshot.py – Multi-method evaluation via a vLLM OpenAI-compatible server
====================================================================================
Supports zero_shot, few_shot, rag, and graph_rag via an OpenAI-compatible vLLM
server.  Designed for large models (32B, 72B) that are too big for local HF
pipeline inference.

Assumes the server is already running, e.g.:
    vllm serve Qwen/Qwen2.5-72B-Instruct \\
        --port 8000 \\
        --tensor-parallel-size 4 \\
        --max-model-len 8192

Usage
-----
  # Zero-shot
  python run_vllm_zeroshot.py \\
      --method zero_shot \\
      --data_file /path/to/test.jsonl \\
      --save_file results/zero_shot.jsonl

  # Few-shot
  python run_vllm_zeroshot.py \\
      --method few_shot --n_shots 3 \\
      --data_file /path/to/test.jsonl \\
      --train_file /path/to/train.jsonl \\
      --save_file results/few_shot.jsonl

  # RAG (document-level, recommended when MatPROV.jsonl is available)
  python run_vllm_zeroshot.py \\
      --method rag \\
      --data_file /path/to/test.jsonl \\
      --train_file /path/to/train.jsonl \\
      --raw_file /path/to/MatPROV.jsonl \\
      --save_file results/rag.jsonl

  # Graph-RAG
  python run_vllm_zeroshot.py \\
      --method graph_rag \\
      --data_file /path/to/test.jsonl \\
      --train_file /path/to/train.jsonl \\
      --raw_file /path/to/MatPROV.jsonl \\
      --save_file results/graph_rag.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

# ── Import shared utilities from co-located modules ───────────────────────────

_BASELINES_DIR = Path(__file__).parent
sys.path.insert(0, str(_BASELINES_DIR))

from run_baselines import (
    load_jsonl,
    write_jsonl,
    choices_to_list,
    letter_to_index,
    format_question_block,
    format_evidence_context,
    build_fewshot_pool,
    _SYS_DIRECT,
    _SYS_RAG,
    _ANSWER_PROMPT,
)

# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """
    Extract a single MCQ letter (A-D) from model output.

    Handles plain letters, leading tokens, explicit phrasing, and
    thinking-model output (reasoning in <think> blocks before final answer).
    """
    text = text.strip()

    m = re.match(r'^([A-D])[\s\.\)\:\,\-]', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.match(r'^([A-D])$', text, re.IGNORECASE):
        return text[0].upper()

    patterns = [
        r'[Aa]nswer\s*:\s*([A-D])',
        r'[Tt]he\s+answer\s+is\s+([A-D])',
        r'[Oo]ption\s+([A-D])',
        r'\b([A-D])\s+is\s+(?:correct|right)\b',
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    alpha = re.sub(r'[^A-Za-z]', '', text)
    if len(alpha) == 1 and alpha.upper() in "ABCD":
        return alpha.upper()

    return text


# ── vLLM inference ────────────────────────────────────────────────────────────

def run_inference_vllm(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> str:
    """
    Call the vLLM OpenAI-compatible endpoint and return the raw assistant text.

    temperature=0.0 → greedy decoding (recommended for MCQ reproducibility).
    enable_thinking=False suppresses Qwen3-style <think> blocks; this
    extra_body field is silently ignored by models that don't support it.
    """
    extra: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if temperature == 0.0:
        extra["top_k"] = -1  # disable top-k for greedy

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0 if temperature == 0.0 else 0.8,
            extra_body=extra,
        )
    except Exception:
        # Some vLLM versions reject unknown extra_body keys; retry without it
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0 if temperature == 0.0 else 0.8,
        )

    msg = response.choices[0].message
    return (msg.content or getattr(msg, "reasoning_content", None) or "").strip()


# ── Message builders ──────────────────────────────────────────────────────────

def _build_zero_shot(rec: dict) -> list[dict]:
    return [
        {"role": "system", "content": _SYS_DIRECT},
        {"role": "user",   "content": format_question_block(rec) + _ANSWER_PROMPT},
    ]


def _build_few_shot(rec: dict, examples: list[dict]) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _SYS_DIRECT}]
    for ex in examples:
        messages.append({"role": "user",      "content": format_question_block(ex) + _ANSWER_PROMPT})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": format_question_block(rec) + _ANSWER_PROMPT})
    return messages


def _build_rag(rec: dict, retrieved: list[dict]) -> list[dict]:
    """Question-level RAG: evidence fields from similar training records."""
    context = format_evidence_context(retrieved)
    user_content = (
        f"Relevant synthesis process information from the literature:\n\n"
        f"{context}\n\n---\n\n"
        f"{format_question_block(rec)}{_ANSWER_PROMPT}"
    )
    return [
        {"role": "system", "content": _SYS_RAG},
        {"role": "user",   "content": user_content},
    ]


def _build_doc_rag(rec: dict, docs: list[dict]) -> list[dict]:
    """Document-level RAG: process descriptions built from MatPROV."""
    from retriever import format_doc_context
    context = format_doc_context(docs)
    user_content = (
        f"Relevant synthesis process information from the literature:\n\n"
        f"{context}\n\n---\n\n"
        f"{format_question_block(rec)}{_ANSWER_PROMPT}"
    )
    return [
        {"role": "system", "content": _SYS_RAG},
        {"role": "user",   "content": user_content},
    ]


def _build_graph_rag(rec: dict, neighbourhoods: list[dict]) -> list[dict]:
    from graph_rag_baseline import format_graph_context, _SYS_GRAPH_RAG
    context = format_graph_context(neighbourhoods)
    user_content = (
        f"Synthesis process knowledge retrieved from graph:\n\n"
        f"{context}\n\n---\n\n"
        f"{format_question_block(rec)}{_ANSWER_PROMPT}"
    )
    return [
        {"role": "system", "content": _SYS_GRAPH_RAG},
        {"role": "user",   "content": user_content},
    ]


# ── Output record ─────────────────────────────────────────────────────────────

def make_output_record(item: dict, raw: str, method: str) -> dict:
    choices_dict = item.get("choices", {})
    gold_letter  = item.get("answer", "")
    return {
        "qid":          item.get("qid", ""),
        "task":         item.get("task", ""),
        "method":       method,
        "question":     item["question"],
        "raw_output":   raw,
        "model_answer": extract_answer(raw),
        "gt_answer":    gold_letter,
        "choices":      choices_to_list(choices_dict),
        "answer_index": letter_to_index(gold_letter),
    }


# ── Inline evaluation ─────────────────────────────────────────────────────────

_EVAL_PATTERNS = [
    re.compile(r'the\s+answer\s+is\s+([A-D])\b',        re.IGNORECASE),
    re.compile(r'\banswer[:\s]+([A-D])\b',               re.IGNORECASE),
    re.compile(r'\boption\s+([A-D])\b',                  re.IGNORECASE),
    re.compile(r'\b([A-D])\s+is\s+(?:correct|right)\b', re.IGNORECASE),
]


def _extract_letter(text: str) -> str:
    raw = (text or "").strip()
    m = re.match(r'^([A-D])[\s\.\)\:\,\-]', raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.match(r'^([A-D])$', raw, re.IGNORECASE):
        return raw[0].upper()
    for pat in _EVAL_PATTERNS:
        m = pat.search(raw)
        if m:
            return m.group(1).upper()
    alpha = re.sub(r'[^A-Za-z]', '', raw)
    if len(alpha) == 1 and alpha.upper() in "ABCD":
        return alpha.upper()
    return ""


def evaluate(results: list[dict]) -> dict:
    def _acc(recs):
        correct = sum(
            1 for r in recs
            if _extract_letter(r["model_answer"]) == r["gt_answer"].strip().upper()
        )
        return {"correct": correct, "total": len(recs),
                "accuracy": correct / len(recs) if recs else 0.0}

    by_task: dict[str, list] = defaultdict(list)
    for r in results:
        by_task[r.get("task", "unknown")].append(r)

    return {
        "overall":  _acc(results),
        "per_task": {t: _acc(v) for t, v in sorted(by_task.items())},
    }


def print_metrics(metrics: dict, model_name: str, method: str) -> None:
    ov = metrics["overall"]
    print(f"\n{'='*60}")
    print(f"Model : {model_name}  |  Method: {method}")
    print(f"Overall accuracy: {ov['accuracy']:.4f}  ({ov['correct']}/{ov['total']})")
    print("\n--- Per task ---")
    for task, d in metrics["per_task"].items():
        if d["total"] > 0:
            print(f"  {task:35s}: {d['accuracy']:.4f}  ({d['correct']}/{d['total']})")
    print("="*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MatProcBench multi-method evaluation via vLLM server"
    )
    # Core
    parser.add_argument("--method",    default="zero_shot",
                        choices=["zero_shot", "few_shot", "rag", "graph_rag"])
    parser.add_argument("--data_file", required=True,  help="Test split JSONL")
    parser.add_argument("--save_file", required=True,  help="Output JSONL path")
    # Server
    parser.add_argument("--base_url",       default="http://localhost:8000/v1")
    parser.add_argument("--model",          default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--max_new_tokens", type=int,   default=32)
    parser.add_argument("--temperature",    type=float, default=0.0,
                        help="0.0 = greedy (recommended for MCQ).")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable Qwen3-style <think> blocks (not recommended for MCQ).")
    # Method-specific
    parser.add_argument("--train_file", default=None,
                        help="Training JSONL (required for few_shot / rag / graph_rag).")
    parser.add_argument("--raw_file",   default=None,
                        help="MatPROV.jsonl raw provenance file. "
                             "Enables document-level RAG and richer graph construction.")
    parser.add_argument("--n_shots",    type=int, default=3)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--rag_k",      type=int, default=3)
    parser.add_argument("--graph_k",    type=int, default=3)
    parser.add_argument("--graph_hops", type=int, default=1,
                        help="Neighbourhood depth (1 or 2) for graph_rag.")
    parser.add_argument("--embedder",   default="all-mpnet-base-v2")
    parser.add_argument("--emb_cache",  default=None,
                        help="Path to cache/load FAISS embeddings.")
    parser.add_argument("--save_metrics", default=None,
                        help="Optional path to write evaluation JSON.")
    args = parser.parse_args()

    if args.method in ("few_shot", "rag", "graph_rag") and not args.train_file:
        parser.error(f"--train_file is required for method='{args.method}'.")

    # ── Connect to server ─────────────────────────────────────────────────────
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    try:
        served = [m.id for m in client.models.list().data]
        print(f"[vllm] Served models: {served}")
        if args.model not in served:
            print(f"[warn] '{args.model}' not in served list; proceeding anyway.")
    except Exception as e:
        print(f"[warn] Cannot reach server ({e}). Proceeding — will fail if unreachable.")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"[run] Loading test : {args.data_file}")
    test_records = load_jsonl(args.data_file)
    print(f"[run] {len(test_records)} test records.")

    train_records: list[dict] = []
    if args.train_file:
        print(f"[run] Loading train: {args.train_file}")
        train_records = load_jsonl(args.train_file)
        print(f"[run] {len(train_records)} train records.")

    # ── Method-specific setup ─────────────────────────────────────────────────
    fewshot_pool:    dict[str, list] = {}
    retriever        = None
    doc_mode: bool   = False
    graph_retriever  = None

    if args.method == "few_shot":
        fewshot_pool = build_fewshot_pool(train_records, args.n_shots, args.seed)
        print(f"[run] Few-shot pool sizes: { {t: len(v) for t, v in fewshot_pool.items()} }")

    elif args.method == "rag":
        raw_path = Path(args.raw_file) if args.raw_file else None
        if raw_path and raw_path.exists():
            from retriever import MatProcDocRetriever
            train_dois = {r.get("evidence", {}).get("doi", "")
                          for r in train_records if r.get("evidence", {}).get("doi")}
            print(f"[run] Building document-level RAG corpus ...")
            retriever = MatProcDocRetriever.from_raw_matprov(
                raw_records=load_jsonl(str(raw_path)),
                train_dois=train_dois,
                embedder_name=args.embedder,
                cache_path=args.emb_cache,
            )
            doc_mode = True
        else:
            from retriever import MatProcRetriever
            if args.raw_file:
                print(f"[warn] --raw_file not found; falling back to question-level RAG.")
            retriever = MatProcRetriever(
                records=train_records,
                embedder_name=args.embedder,
                cache_path=args.emb_cache,
            )
            doc_mode = False
        print(f"[run] RAG ready (k={args.rag_k}, doc_mode={doc_mode}).")

    elif args.method == "graph_rag":
        from graph_rag_baseline import ProcessKnowledgeGraph, GraphRetriever
        train_dois = {r.get("evidence", {}).get("doi", "")
                      for r in train_records if r.get("evidence", {}).get("doi")}
        raw_path = Path(args.raw_file) if args.raw_file else None
        if raw_path and raw_path.exists():
            print(f"[run] Building graph from MatPROV ...")
            graph = ProcessKnowledgeGraph.build_from_raw(
                load_jsonl(str(raw_path)), train_dois)
            graph.ingest_qa_records(train_records)
        else:
            if args.raw_file:
                print(f"[warn] --raw_file not found; using QA evidence only.")
            graph = ProcessKnowledgeGraph()
            graph.ingest_qa_records(train_records)
        print(f"[run] Graph stats: {graph.stats()}")
        graph_retriever = GraphRetriever(
            graph=graph, embedder_name=args.embedder, cache_path=args.emb_cache)
        print(f"[run] Graph retriever ready (k={args.graph_k}, hops={args.graph_hops}).")

    # ── Inference loop ────────────────────────────────────────────────────────
    results: list[dict] = []

    for item in tqdm(test_records, desc=f"[{args.method}]"):
        task = item.get("task", "unknown")
        qid  = item.get("qid", "")
        raw  = ""

        try:
            if args.method == "zero_shot":
                messages = _build_zero_shot(item)

            elif args.method == "few_shot":
                messages = _build_few_shot(item, fewshot_pool.get(task, []))

            elif args.method == "rag":
                if doc_mode:
                    doi  = item.get("evidence", {}).get("doi", None)
                    docs = retriever.search(query=item["question"], k=args.rag_k,
                                            exclude_doi=doi)
                    messages = _build_doc_rag(item, docs)
                else:
                    retrieved = retriever.search_by_task(
                        query=item["question"], task=task, k=args.rag_k,
                        exclude_qid=qid or None, dedup_doi=True,
                    )
                    messages = _build_rag(item, retrieved)

            else:  # graph_rag
                doi = item.get("evidence", {}).get("doi", None)
                nbs = graph_retriever.search(
                    query=item["question"], k=args.graph_k,
                    exclude_doi=doi, hops=args.graph_hops,
                )
                messages = _build_graph_rag(item, nbs)

            raw = run_inference_vllm(
                client, args.model, messages, args.max_new_tokens,
                temperature=args.temperature,
                enable_thinking=args.enable_thinking,
            )

        except Exception as exc:
            print(f"\n[error] qid={qid}: {exc}")

        results.append(make_output_record(item, raw, args.method))

    # ── Save & evaluate ───────────────────────────────────────────────────────
    write_jsonl(args.save_file, results)
    print(f"\n[run] Saved {len(results)} records → {args.save_file}")

    metrics = evaluate(results)
    print_metrics(metrics, args.model, args.method)

    if args.save_metrics:
        Path(args.save_metrics).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_metrics, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        print(f"[run] Metrics → {args.save_metrics}")


if __name__ == "__main__":
    main()
