"""
run_baselines.py – MatProcBench MCQ Baseline Runner
====================================================
Runs one of four inference methods on the MatProcBench dataset:

  zero_shot  – direct MCQ answer; no examples, no retrieval.
  few_shot   – k task-stratified in-context examples sampled from the
               training split; multi-turn user/assistant messages.
  cot        – zero-shot chain-of-thought; model reasons step-by-step
               then emits "Answer: X".
  rag        – k evidence records retrieved by FAISS semantic search;
               formatted as process-level context before the question.

Output JSONL is fully compatible with ../eval.py:
  gt_answer = gold letter  (A/B/C/D)
  choices   = list of option texts  [choice_A, choice_B, choice_C, choice_D]
  answer_index = 0-based index of the correct option

Usage
-----
  # Zero-shot
  python run_baselines.py \\
      --method zero_shot --model Qwen/Qwen2.5-7B-Instruct \\
      --data_file /path/to/test.jsonl --save_file results/zero_shot.jsonl

  # Few-shot  (requires --train_file)
  python run_baselines.py \\
      --method few_shot --n_shots 3 \\
      --data_file /path/to/test.jsonl --train_file /path/to/train.jsonl \\
      --save_file results/few_shot_3.jsonl

  # Chain-of-Thought
  python run_baselines.py \\
      --method cot --max_new_tokens 512 \\
      --data_file /path/to/test.jsonl --save_file results/cot.jsonl

  # RAG  (requires --train_file)
  python run_baselines.py \\
      --method rag --rag_k 3 \\
      --data_file /path/to/test.jsonl --train_file /path/to/train.jsonl \\
      --save_file results/rag_k3.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline


# ── Supported models ──────────────────────────────────────────────────────────

SUPPORTED_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-8B",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
]

# ── System prompts ────────────────────────────────────────────────────────────

_SYS_DIRECT = (
    "You are an expert assistant in materials science and materials synthesis. "
    "Answer the following multiple-choice question by outputting ONLY the letter "
    "(A, B, C, or D) of the correct option. "
    "Do not include any explanation or extra text."
)

_SYS_COT = (
    "You are an expert assistant in materials science and materials synthesis. "
    "Think step by step to reason through the problem before answering. "
    "After your reasoning, output your final answer on a new line in the format:\n"
    "Answer: X\n"
    "where X is the letter (A, B, C, or D) of the correct option."
)

_SYS_RAG = (
    "You are an expert assistant in materials science and materials synthesis. "
    "You are provided with relevant synthesis process information retrieved from "
    "the literature to help you answer the question. "
    "Answer the multiple-choice question by outputting ONLY the letter "
    "(A, B, C, or D) of the correct option. "
    "Do not include any explanation or extra text."
)


_ANSWER_PROMPT = "\n\nAnswer with only the letter (A, B, C, or D):"


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def choices_to_list(choices_dict: dict) -> list[str]:
    """Convert {'A': '...', 'B': '...', ...} → ['...', '...', '...', '...']."""
    return [choices_dict[l] for l in "ABCD" if l in choices_dict]


def letter_to_index(letter: str) -> int:
    """'A' → 0, 'B' → 1, etc. Returns -1 on failure."""
    letter = letter.strip().upper()
    if letter in "ABCD":
        return ord(letter) - ord("A")
    return -1


# ── Prompt formatters ─────────────────────────────────────────────────────────

def format_choices_block(choices_dict: dict) -> str:
    """Format choices dict as 'A: ...\nB: ...\nC: ...\nD: ...'."""
    return "\n".join(f"{l}: {choices_dict[l]}" for l in "ABCD" if l in choices_dict)


def format_question_block(rec: dict) -> str:
    """Combine question text with formatted choices."""
    return f"{rec['question']}\n\nOptions:\n{format_choices_block(rec['choices'])}"


def format_fewshot_example(rec: dict) -> str:
    """Single-line representation used inside few-shot assistant turns."""
    return rec["answer"]  # just the letter; question/choices go in the user turn


def format_evidence_context(retrieved: list[dict], max_chars: int = 3000) -> str:
    """
    Format the evidence fields of retrieved records as process-level context.
    Only task-relevant fields are emitted; internal IDs are skipped.

    Returns a string that is prepended to the question in the RAG user message.
    """
    _SKIP_KEYS = {"doi", "activity_id", "process_label"}
    _LIST_KEYS = {"precursors", "products", "correct_sequence",
                  "activity_sequence", "tools", "fresh_precursors",
                  "flowing_intermediates", "process_labels"}
    _DICT_KEYS = {"all_conditions", "conditions"}

    parts: list[str] = []
    total_chars = 0

    for idx, rec in enumerate(retrieved, 1):
        ev = rec.get("evidence", {})
        task = rec.get("task", "")
        lines = [f"[Retrieved Process {idx}] (task: {task})"]

        for key, val in ev.items():
            if key in _SKIP_KEYS or not val:
                continue
            label = key.replace("_", " ").capitalize()
            if key in _LIST_KEYS and isinstance(val, list):
                lines.append(f"  {label}: {' → '.join(str(v) for v in val)}"
                             if "sequence" in key or "route" in key
                             else f"  {label}: {', '.join(str(v) for v in val)}")
            elif key in _DICT_KEYS and isinstance(val, dict):
                cond_str = "; ".join(f"{k}: {v}" for k, v in val.items() if v)
                lines.append(f"  {label}: {cond_str}")
            else:
                lines.append(f"  {label}: {val}")

        block = "\n".join(lines)
        if total_chars + len(block) > max_chars:
            break
        parts.append(block)
        total_chars += len(block)

    return "\n\n".join(parts)


# ── Message builders ──────────────────────────────────────────────────────────

def build_messages_zero_shot(rec: dict) -> list[dict]:
    """System + single user message with question and choices."""
    user_content = format_question_block(rec) + _ANSWER_PROMPT
    return [
        {"role": "system", "content": _SYS_DIRECT},
        {"role": "user", "content": user_content},
    ]


def build_messages_few_shot(rec: dict, examples: list[dict]) -> list[dict]:
    """
    Multi-turn few-shot: system → [user / assistant] × k → user (test).
    Each assistant turn contains only the gold letter.
    """
    messages: list[dict] = [{"role": "system", "content": _SYS_DIRECT}]
    for ex in examples:
        messages.append({
            "role": "user",
            "content": format_question_block(ex) + _ANSWER_PROMPT,
        })
        messages.append({
            "role": "assistant",
            "content": ex["answer"],
        })
    # Append the test question as the final user turn
    messages.append({
        "role": "user",
        "content": format_question_block(rec) + _ANSWER_PROMPT,
    })
    return messages


def build_messages_cot(rec: dict) -> list[dict]:
    """Zero-shot CoT: system with reasoning instruction + user with question."""
    user_content = format_question_block(rec)
    return [
        {"role": "system", "content": _SYS_COT},
        {"role": "user", "content": user_content},
    ]


def build_messages_rag(rec: dict, retrieved: list[dict]) -> list[dict]:
    """
    RAG: system → user message with retrieved evidence context + question.
    Evidence is formatted as process-level descriptions (not Q&A pairs).
    Used when retrieval is over training QA records (MatProcRetriever).
    """
    context = format_evidence_context(retrieved)
    user_content = (
        f"Relevant synthesis process information from the literature:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"{format_question_block(rec)}"
        f"{_ANSWER_PROMPT}"
    )
    return [
        {"role": "system", "content": _SYS_RAG},
        {"role": "user", "content": user_content},
    ]


def build_messages_doc_rag(rec: dict, docs: list[dict]) -> list[dict]:
    """
    Document-level RAG: system → user message with retrieved process documents.
    Used when retrieval is over MatPROV process documents (MatProcDocRetriever).
    """
    from retriever import format_doc_context
    context = format_doc_context(docs)
    user_content = (
        f"Relevant synthesis process information from the literature:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"{format_question_block(rec)}"
        f"{_ANSWER_PROMPT}"
    )
    return [
        {"role": "system", "content": _SYS_RAG},
        {"role": "user", "content": user_content},
    ]


# ── Few-shot example pool ─────────────────────────────────────────────────────

def build_fewshot_pool(
    train_records: list[dict],
    n_shots: int,
    seed: int,
) -> dict[str, list[dict]]:
    """
    Pre-sample n_shots examples per task type from the training split.
    Returns a dict: task_type → list[dict].
    The same pool is reused for all test records of that task type so
    results are reproducible and examples are not leaked from the test set.
    """
    rng = random.Random(seed)
    task_groups: dict[str, list[dict]] = {}
    for rec in train_records:
        task = rec.get("task", "unknown")
        task_groups.setdefault(task, []).append(rec)

    pool: dict[str, list[dict]] = {}
    for task, group in task_groups.items():
        rng.shuffle(group)
        pool[task] = group[:n_shots]
    return pool


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(pipe, messages: list[dict], max_new_tokens: int) -> str:
    """Run the HuggingFace pipeline and return the assistant's raw text."""
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,   # override model's default; suppresses do_sample warning
        top_p=1.0,
        top_k=50,
        num_return_sequences=1,
    )
    # pipeline returns generated_text as a list of messages when given a list
    generated = outputs[0]["generated_text"]
    if isinstance(generated, list):
        # Chat-template output: last message is the assistant reply
        return generated[-1]["content"].strip()
    # Fallback: plain string
    return generated.strip()


def extract_cot_answer(full_text: str) -> str:
    """
    Extract the final answer letter from a CoT response.
    Looks for 'Answer: X' pattern (last occurrence wins).
    Falls back to returning the full text if no match found.
    """
    matches = re.findall(r'[Aa]nswer\s*:\s*([A-Da-d])', full_text)
    if matches:
        return matches[-1].upper()
    return full_text


def make_output_record(
    item: dict,
    model_answer: str,
    method: str,
    cot_trace: str | None = None,
) -> dict:
    """Build the standardised output dict compatible with eval.py."""
    choices_dict = item.get("choices", {})
    gold_letter = item.get("answer", "")
    record = {
        "qid":          item.get("qid", ""),
        "task":         item.get("task", ""),
        "method":       method,
        "question":     item["question"],
        "model_answer": model_answer,
        "gt_answer":    gold_letter,                       # single letter
        "choices":      choices_to_list(choices_dict),     # list for eval.py
        "answer_index": letter_to_index(gold_letter),      # 0-based
    }
    if cot_trace is not None:
        record["cot_trace"] = cot_trace
    return record


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MatProcBench MCQ baseline runner"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["zero_shot", "few_shot", "cot", "rag"],
        help="Inference method.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the evaluation JSONL (e.g. test.jsonl).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to training JSONL (required for few_shot and rag).",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        required=True,
        help="Output JSONL path.",
    )
    # few-shot options
    parser.add_argument(
        "--n_shots",
        type=int,
        default=3,
        help="Number of in-context examples (few_shot only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for example sampling.",
    )
    # RAG options
    parser.add_argument(
        "--rag_k",
        type=int,
        default=3,
        help="Number of retrieved records/documents (rag only).",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="all-mpnet-base-v2",
        help="Sentence-transformer model for RAG retrieval.",
    )
    parser.add_argument(
        "--emb_cache",
        type=str,
        default=None,
        help="Path to cache/load FAISS embeddings (rag only).",
    )
    parser.add_argument(
        "--raw_file",
        type=str,
        default=None,
        help=(
            "Path to MatPROV.jsonl raw provenance file (rag only). "
            "When provided, enables document-level RAG: one text document per "
            "synthesis process is indexed instead of training question texts, "
            "matching the standard RAG paradigm of indexing knowledge rather "
            "than questions."
        ),
    )
    # Generation options
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help=(
            "Max new tokens to generate. "
            "Defaults: zero_shot/few_shot=16, cot=512, rag=16."
        ),
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization (requires bitsandbytes). Recommended for 32B+ models.",
    )
    args = parser.parse_args()

    # Validate method-specific requirements
    if args.method in ("few_shot", "rag") and not args.train_file:
        parser.error(f"--train_file is required for method='{args.method}'.")

    # Default max_new_tokens per method
    default_tokens = {"zero_shot": 16, "few_shot": 16, "cot": 1024, "rag": 16}
    max_new_tokens = args.max_new_tokens or default_tokens[args.method]

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[run_baselines] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # LLaMat-2-Chat (and other Llama-2-based models) ship without a chat_template.
    # Apply the standard Llama-2 chat template so apply_chat_template() works.
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% set messages = messages[1:] %}"
            "{% else %}"
            "{% set system_message = '' %}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<s>[INST] ' }}"
            "{% if loop.first and system_message %}"
            "{{ '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' }}"
            "{% endif %}"
            "{{ message['content'] + ' [/INST] ' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + ' </s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )

    quantization_config = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        if args.load_in_4bit else None
    )
    pipe = pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer,
        torch_dtype=None if args.load_in_4bit else torch.bfloat16,
        model_kwargs={"quantization_config": quantization_config} if quantization_config else {},
        device_map="auto",
        trust_remote_code=True,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"[run_baselines] Loading test data: {args.data_file}")
    test_records = load_jsonl(args.data_file)
    print(f"[run_baselines] {len(test_records)} test records.")

    train_records: list[dict] = []
    if args.train_file:
        print(f"[run_baselines] Loading training data: {args.train_file}")
        train_records = load_jsonl(args.train_file)
        print(f"[run_baselines] {len(train_records)} training records.")

    # ── Method-specific setup ─────────────────────────────────────────────────
    fewshot_pool: dict[str, list[dict]] = {}
    retriever = None

    if args.method == "few_shot":
        fewshot_pool = build_fewshot_pool(train_records, args.n_shots, args.seed)
        task_counts = {t: len(v) for t, v in fewshot_pool.items()}
        print(f"[run_baselines] Few-shot pool: {task_counts}")

    elif args.method == "rag":
        sys.path.insert(0, str(Path(__file__).parent))

        raw_file_path = Path(args.raw_file) if args.raw_file else None
        if raw_file_path and raw_file_path.exists():
            # ── Document-level RAG (standard): index MatPROV process docs ──────
            from retriever import MatProcDocRetriever
            train_dois: set[str] = {
                r.get("evidence", {}).get("doi", "")
                for r in train_records if r.get("evidence", {}).get("doi")
            }
            print(f"[run_baselines] Building document-level RAG corpus from {raw_file_path}")
            raw_records_rag = load_jsonl(str(raw_file_path))
            retriever = MatProcDocRetriever.from_raw_matprov(
                raw_records=raw_records_rag,
                train_dois=train_dois,
                embedder_name=args.embedder,
                cache_path=args.emb_cache,
            )
            print(f"[run_baselines] Doc-RAG retriever ready (k={args.rag_k}).")
        else:
            # ── Question-level RAG (legacy): index training question texts ─────
            from retriever import MatProcRetriever
            if args.raw_file:
                print(f"[run_baselines] Warning: --raw_file not found ({args.raw_file}); "
                      "falling back to question-level RAG.")
            retriever = MatProcRetriever(
                records=train_records,
                embedder_name=args.embedder,
                cache_path=args.emb_cache,
            )
            print(f"[run_baselines] Question-RAG retriever ready (k={args.rag_k}).")

    # ── Inference loop ────────────────────────────────────────────────────────
    results: list[dict] = []

    for item in tqdm(test_records, desc=f"[{args.method}]"):
        task = item.get("task", "unknown")
        qid  = item.get("qid", "")

        if args.method == "zero_shot":
            messages = build_messages_zero_shot(item)

        elif args.method == "few_shot":
            examples = fewshot_pool.get(task, [])
            messages = build_messages_few_shot(item, examples)

        elif args.method == "cot":
            messages = build_messages_cot(item)

        else:  # rag
            from retriever import MatProcDocRetriever
            if isinstance(retriever, MatProcDocRetriever):
                # Document-level RAG: retrieve by DOI exclusion
                doi = item.get("evidence", {}).get("doi", None)
                docs = retriever.search(
                    query=item["question"],
                    k=args.rag_k,
                    exclude_doi=doi,
                )
                messages = build_messages_doc_rag(item, docs)
            else:
                # Question-level RAG: task-stratified retrieval with DOI dedup
                retrieved = retriever.search_by_task(
                    query=item["question"],
                    task=task,
                    k=args.rag_k,
                    exclude_qid=qid if qid else None,
                    dedup_doi=True,
                )
                messages = build_messages_rag(item, retrieved)

        raw_output = run_inference(pipe, messages, max_new_tokens)

        if args.method == "cot":
            cot_trace = raw_output
            model_answer = extract_cot_answer(raw_output)
        else:
            cot_trace = None
            model_answer = raw_output

        results.append(make_output_record(item, model_answer, args.method, cot_trace))

    # ── Save ──────────────────────────────────────────────────────────────────
    write_jsonl(args.save_file, results)
    print(f"[run_baselines] Saved {len(results)} records to: {args.save_file}")
    if results:
        print(f"[run_baselines] Sample output: {results[0]}")


if __name__ == "__main__":
    main()
