"""
run_all_splits.py – Batch evaluator across all splits and methods via vLLM
===========================================================================
Runs zero_shot, few_shot, rag, and graphrag evaluations on every dataset
split (random_split, year_split, type_split, additional_split) using a
vLLM OpenAI-compatible server that must already be running, e.g.:

  CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-27B \\
      --port 8000 --tensor-parallel-size 1 \\
      --max-model-len auto --reasoning-parser qwen3 \\
      --language-model-only

Results are written to:
  <out_root>/<split>/<method>.jsonl          (predictions)
  <out_root>/<split>/<method>_metrics.json   (per-task accuracy)

Usage
-----
  # Evaluate all splits × all methods with defaults
  python run_all_splits.py

  # Only specific splits and methods
  python run_all_splits.py \\
      --splits random_split year_split \\
      --methods zero_shot few_shot

  # Custom paths
  python run_all_splits.py \\
      --model Qwen/Qwen3.5-27B \\
      --vllm_url http://localhost:8000/v1 \\
      --data_root ../../data/processed \\
      --raw_file ../../data/raw_data/MatPROV.jsonl \\
      --out_root results/qwen35_27b
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

# ── Locate sibling modules ────────────────────────────────────────────────────

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

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
from retriever import format_doc_context
from graph_rag_baseline import format_graph_context, _SYS_GRAPH_RAG
from retriever import MatProcDocRetriever, MatProcRetriever
from graph_rag_baseline import ProcessKnowledgeGraph, GraphRetriever


# ── Constants ─────────────────────────────────────────────────────────────────

ALL_SPLITS  = ["random_split", "year_split", "type_split", "additional_split"]
ALL_METHODS = ["zero_shot", "few_shot", "rag", "graphrag"]

# Per-method defaults for max_new_tokens; can be overridden via --max_tokens
_DEFAULT_TOKENS: dict[str, int] = {
    "zero_shot": 32,
    "few_shot":  32,
    "rag":       32,
    "graphrag":  32,
}

# ── Answer extraction ─────────────────────────────────────────────────────────

_LETTER_PATS = [
    re.compile(r'[Aa]nswer\s*:\s*([A-D])',                 re.IGNORECASE),
    re.compile(r'[Tt]he\s+answer\s+is\s+([A-D])\b',        re.IGNORECASE),
    re.compile(r'[Oo]ption\s+([A-D])\b',                   re.IGNORECASE),
    re.compile(r'\b([A-D])\s+is\s+(?:correct|right)\b',    re.IGNORECASE),
]


def extract_answer(text: str) -> str:
    """
    Extract a single MCQ letter (A-D) from raw model output.

    Priority:
      1. Leading letter token (A/B/C/D followed by punctuation or end)
      2. Explicit phrasing patterns
      3. Single-letter response
    Falls back to returning the full text if nothing matches.
    """
    text = text.strip()

    m = re.match(r'^([A-D])[\s\.\)\:\,\-]', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r'[A-D]', text, re.IGNORECASE):
        return text[0].upper()

    for pat in _LETTER_PATS:
        hits = pat.findall(text)
        if hits:
            return hits[-1].upper()

    alpha = re.sub(r'[^A-Za-z]', '', text)
    if len(alpha) == 1 and alpha.upper() in "ABCD":
        return alpha.upper()

    return text


# ── vLLM inference ────────────────────────────────────────────────────────────

def run_inference(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> str:
    """
    Call the vLLM OpenAI-compatible chat completions endpoint.

    temperature=0 → greedy decoding (recommended for reproducible MCQ).
    enable_thinking controls Qwen3-style <think> blocks via extra_body;
    silently ignored by models that do not support it.
    Returns the raw assistant content string.
    """
    extra: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if temperature == 0.0:
        extra["top_k"] = -1  # disable top-k for greedy

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0 if temperature == 0.0 else 0.8,
            extra_body=extra,
        )
    except Exception:
        # Older vLLM versions may reject unknown extra_body keys; retry bare
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0 if temperature == 0.0 else 0.8,
        )

    msg = resp.choices[0].message
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


def _build_rag_qa(rec: dict, retrieved: list[dict]) -> list[dict]:
    """Question-level RAG: format evidence fields of similar training records."""
    context = format_evidence_context(retrieved)
    user = (
        f"Relevant synthesis process information from the literature:\n\n"
        f"{context}\n\n---\n\n"
        f"{format_question_block(rec)}{_ANSWER_PROMPT}"
    )
    return [{"role": "system", "content": _SYS_RAG},
            {"role": "user",   "content": user}]


def _build_rag_doc(rec: dict, docs: list[dict]) -> list[dict]:
    """Document-level RAG: process descriptions built from MatPROV."""
    context = format_doc_context(docs)
    user = (
        f"Relevant synthesis process information from the literature:\n\n"
        f"{context}\n\n---\n\n"
        f"{format_question_block(rec)}{_ANSWER_PROMPT}"
    )
    return [{"role": "system", "content": _SYS_RAG},
            {"role": "user",   "content": user}]


def _build_graphrag(rec: dict, neighbourhoods: list[dict]) -> list[dict]:
    context = format_graph_context(neighbourhoods)
    user = (
        f"Synthesis process knowledge retrieved from graph:\n\n"
        f"{context}\n\n---\n\n"
        f"{format_question_block(rec)}{_ANSWER_PROMPT}"
    )
    return [{"role": "system", "content": _SYS_GRAPH_RAG},
            {"role": "user",   "content": user}]


# ── Evaluation helpers ────────────────────────────────────────────────────────

def make_record(item: dict, raw: str, method: str) -> dict:
    choices_dict = item.get("choices", {})
    gold = item.get("answer", "")
    return {
        "qid":          item.get("qid", ""),
        "task":         item.get("task", ""),
        "method":       method,
        "question":     item["question"],
        "raw_output":   raw,
        "model_answer": extract_answer(raw),
        "gt_answer":    gold,
        "choices":      choices_to_list(choices_dict),
        "answer_index": letter_to_index(gold),
    }


def compute_metrics(results: list[dict]) -> dict:
    def _acc(recs: list[dict]) -> dict:
        correct = sum(
            1 for r in recs
            if extract_answer(r["model_answer"]) == r["gt_answer"].strip().upper()
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


def print_metrics(metrics: dict, tag: str) -> None:
    ov = metrics["overall"]
    print(f"\n{'─'*60}")
    print(f"{tag}")
    print(f"Overall accuracy: {ov['accuracy']:.4f}  ({ov['correct']}/{ov['total']})")
    for task, d in metrics["per_task"].items():
        if d["total"] > 0:
            print(f"  {task:40s}: {d['accuracy']:.4f}  ({d['correct']}/{d['total']})")
    print(f"{'─'*60}")


# ── Per-split runner ──────────────────────────────────────────────────────────

def run_split(
    split: str,
    methods: list[str],
    *,
    client: OpenAI,
    model: str,
    data_root: Path,
    raw_file: Path | None,
    out_root: Path,
    n_shots: int,
    rag_k: int,
    graph_k: int,
    graph_hops: int,
    embedder: str,
    max_tokens: int | None,
    temperature: float,
    enable_thinking: bool,
    seed: int,
    skip_existing: bool,
) -> dict[str, dict]:
    """
    Run every requested method on one dataset split.
    Returns {method: metrics_dict}.
    """
    split_dir  = data_root / split
    train_file = split_dir / "train.jsonl"
    test_file  = split_dir / "test.jsonl"
    out_dir    = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # FAISS embedding caches are shared across methods within a split
    rag_cache   = str(out_dir / "_rag_emb_cache.pkl")
    graph_cache = str(out_dir / "_graph_emb_cache.pkl")

    print(f"\n{'='*60}")
    print(f"SPLIT: {split}")
    print(f"  test : {test_file}")
    print(f"  train: {train_file}")

    test_records = load_jsonl(str(test_file))
    train_records = load_jsonl(str(train_file)) if train_file.exists() else []
    print(f"  test={len(test_records)}, train={len(train_records)}")

    # Lazy-initialise retrievers only when a method needs them
    _retrievers: dict[str, Any] = {}

    def get_rag_retriever(doc_mode: bool):
        key = "rag_doc" if doc_mode else "rag_qa"
        if key not in _retrievers:
            if doc_mode:
                train_dois = {r.get("evidence", {}).get("doi", "")
                              for r in train_records if r.get("evidence", {}).get("doi")}
                _retrievers[key] = MatProcDocRetriever.from_raw_matprov(
                    raw_records=load_jsonl(str(raw_file)),
                    train_dois=train_dois,
                    embedder_name=embedder,
                    cache_path=rag_cache,
                )
            else:
                _retrievers[key] = MatProcRetriever(
                    records=train_records,
                    embedder_name=embedder,
                    cache_path=rag_cache,
                )
        return _retrievers[key]

    def get_graph_retriever():
        key = "graph"
        if key not in _retrievers:
            train_dois = {r.get("evidence", {}).get("doi", "")
                          for r in train_records if r.get("evidence", {}).get("doi")}
            if raw_file and raw_file.exists():
                graph = ProcessKnowledgeGraph.build_from_raw(
                    load_jsonl(str(raw_file)), train_dois)
                graph.ingest_qa_records(train_records)
            else:
                graph = ProcessKnowledgeGraph()
                graph.ingest_qa_records(train_records)
            print(f"  Graph stats: {graph.stats()}")
            _retrievers[key] = GraphRetriever(
                graph=graph, embedder_name=embedder, cache_path=graph_cache)
        return _retrievers[key]

    split_metrics: dict[str, dict] = {}

    for method in methods:
        out_file     = out_dir / f"{method}.jsonl"
        metrics_file = out_dir / f"{method}_metrics.json"

        if skip_existing and out_file.exists():
            print(f"\n[{split}/{method}] Skipping — {out_file} already exists.")
            if metrics_file.exists():
                split_metrics[method] = json.loads(metrics_file.read_text())
            continue

        tokens = max_tokens if max_tokens is not None else _DEFAULT_TOKENS[method]
        print(f"\n[{split}/{method}] Starting ({len(test_records)} items, max_tokens={tokens})")

        # ── Method setup ──────────────────────────────────────────────────────
        fewshot_pool:  dict[str, list] = {}
        retriever      = None
        doc_mode       = False
        graph_retriever = None

        if method == "few_shot":
            fewshot_pool = build_fewshot_pool(train_records, n_shots, seed)

        elif method == "rag":
            use_doc = raw_file is not None and raw_file.exists()
            retriever = get_rag_retriever(use_doc)
            doc_mode  = use_doc
            print(f"  RAG doc_mode={doc_mode}, k={rag_k}")

        elif method == "graphrag":
            graph_retriever = get_graph_retriever()
            print(f"  GraphRAG k={graph_k}, hops={graph_hops}")

        # ── Inference loop ────────────────────────────────────────────────────
        results: list[dict] = []

        for item in tqdm(test_records, desc=f"  {method}", leave=False):
            task = item.get("task", "unknown")
            qid  = item.get("qid", "")
            raw  = ""

            try:
                if method == "zero_shot":
                    msgs = _build_zero_shot(item)

                elif method == "few_shot":
                    msgs = _build_few_shot(item, fewshot_pool.get(task, []))

                elif method == "rag":
                    if doc_mode:
                        doi  = item.get("evidence", {}).get("doi", None)
                        docs = retriever.search(query=item["question"], k=rag_k,
                                                exclude_doi=doi)
                        msgs = _build_rag_doc(item, docs)
                    else:
                        retrieved = retriever.search_by_task(
                            query=item["question"], task=task, k=rag_k,
                            exclude_qid=qid or None, dedup_doi=True,
                        )
                        msgs = _build_rag_qa(item, retrieved)

                else:  # graphrag
                    doi = item.get("evidence", {}).get("doi", None)
                    nbs = graph_retriever.search(
                        query=item["question"], k=graph_k,
                        exclude_doi=doi, hops=graph_hops,
                    )
                    msgs = _build_graphrag(item, nbs)

                raw = run_inference(client, model, msgs, tokens,
                                    temperature=temperature,
                                    enable_thinking=enable_thinking)

            except Exception as exc:
                print(f"\n  [error] qid={qid}: {exc}")

            results.append(make_record(item, raw, method))

        # ── Save results ──────────────────────────────────────────────────────
        write_jsonl(str(out_file), results)
        print(f"  Saved {len(results)} records → {out_file}")

        metrics = compute_metrics(results)
        metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"  Metrics → {metrics_file}")

        print_metrics(metrics, tag=f"{split}/{method}")
        split_metrics[method] = metrics

    return split_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MatProcBench: evaluate all splits × all methods via vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server
    parser.add_argument("--model",           default="Qwen/Qwen3.5-27B",
                        help="Model ID as registered in vLLM.")
    parser.add_argument("--vllm_url",        default="http://localhost:8000/v1",
                        help="Base URL of the vLLM OpenAI-compatible server.")
    parser.add_argument("--temperature",     type=float, default=0.0,
                        help="0.0 = greedy decoding (recommended for MCQ).")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable Qwen3-style <think> blocks (off by default).")
    parser.add_argument("--max_tokens",      type=int, default=None,
                        help="Max new tokens. Defaults to 32 for all methods.")

    # What to run
    parser.add_argument("--splits",  nargs="+", default=ALL_SPLITS,
                        choices=ALL_SPLITS, metavar="SPLIT",
                        help="Splits to evaluate.")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, metavar="METHOD",
                        help="Methods to run.")

    # Paths (use relative paths from this script's location)
    _script_dir = Path(__file__).resolve().parent
    _repo_root  = _script_dir.parent.parent   # MatProcBench/

    parser.add_argument("--data_root",
                        default=str(_repo_root / "data" / "processed"),
                        help="Directory containing split subdirectories.")
    parser.add_argument("--raw_file",
                        default=str(_repo_root / "data" / "raw_data" / "MatPROV.jsonl"),
                        help="MatPROV.jsonl raw provenance file (enables doc-RAG and richer graph).")
    parser.add_argument("--out_root",
                        default=str(_script_dir / "results" / "vllm_eval"),
                        help="Root output directory.")

    # Method hyper-parameters
    parser.add_argument("--n_shots",    type=int, default=3,
                        help="Number of few-shot examples.")
    parser.add_argument("--rag_k",      type=int, default=3,
                        help="Number of retrieved documents for RAG.")
    parser.add_argument("--graph_k",    type=int, default=3,
                        help="Number of retrieved neighbourhoods for GraphRAG.")
    parser.add_argument("--graph_hops", type=int, default=1,
                        help="Neighbourhood depth for GraphRAG (1 or 2).")
    parser.add_argument("--embedder",   default="all-mpnet-base-v2",
                        help="Sentence-transformer model for FAISS retrieval.")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for few-shot sampling.")

    # Misc
    parser.add_argument("--force", action="store_true",
                        help="Re-run and overwrite even if the output file already exists.")
    parser.add_argument("--summary_file",  default=None,
                        help="Optional JSON file to write the full metrics summary.")

    args = parser.parse_args()

    raw_path = Path(args.raw_file)
    if not raw_path.exists():
        print(f"[warn] --raw_file not found: {raw_path}")
        print("       RAG will fall back to question-level; GraphRAG will use QA evidence only.")
        raw_path = None

    # ── Connect to vLLM server ────────────────────────────────────────────────
    client = OpenAI(base_url=args.vllm_url, api_key="EMPTY")
    print(f"[setup] Connecting to vLLM at {args.vllm_url}")
    try:
        served = [m.id for m in client.models.list().data]
        print(f"[setup] Served models: {served}")
        if args.model not in served:
            print(f"[warn] '{args.model}' not in served list; proceeding anyway.")
    except Exception as e:
        print(f"[warn] Cannot reach vLLM server ({e}). Will fail if still unreachable during inference.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Output root: {out_root}")
    print(f"[setup] Splits : {args.splits}")
    print(f"[setup] Methods: {args.methods}")

    # ── Run all combinations ──────────────────────────────────────────────────
    all_metrics: dict[str, dict[str, dict]] = {}

    for split in args.splits:
        split_metrics = run_split(
            split=split,
            methods=args.methods,
            client=client,
            model=args.model,
            data_root=Path(args.data_root),
            raw_file=raw_path,
            out_root=out_root,
            n_shots=args.n_shots,
            rag_k=args.rag_k,
            graph_k=args.graph_k,
            graph_hops=args.graph_hops,
            embedder=args.embedder,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
            seed=args.seed,
            skip_existing=not args.force,
        )
        all_metrics[split] = split_metrics

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print("FINAL SUMMARY")
    print(f"{'#'*60}")
    header = f"{'Split':<20}" + "".join(f"{m:>12}" for m in args.methods)
    print(header)
    print("-" * len(header))
    for split, sm in all_metrics.items():
        row = f"{split:<20}"
        for method in args.methods:
            if method in sm:
                acc = sm[method]["overall"]["accuracy"]
                row += f"{acc:>12.4f}"
            else:
                row += f"{'N/A':>12}"
        print(row)

    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
        print(f"\n[done] Full metrics → {args.summary_file}")


if __name__ == "__main__":
    main()
