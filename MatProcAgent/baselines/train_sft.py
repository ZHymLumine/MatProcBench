"""
train_sft.py – Evidence-Augmented SFT with LoRA for MatProcBench MCQ
=====================================================================

Training strategy:
  - During training: inject gold `evidence` fields as structured context → answer
  - At inference: use RAG-retrieved evidence as context (see run_baselines.py rag method)
  - The model learns HOW to reason from process information, not memorize answers
  - This improves OOD generalisation vs. pure prompt-based methods

Key design decisions:
  - LoRA fine-tuning with attention + MLP layers (parameter-efficient)
  - Completion-only loss: loss computed on assistant reply tokens only
  - Task-balanced sampling (sqrt strategy prevents B1 domination)
  - Evidence formatted with full contextual fields per task type
  - Evidence dropout for train/inference distribution robustness
  - Dev-set MCQ accuracy callback with early stopping

Usage
-----
  # Basic run on additional_split training data
  python train_sft.py \\
      --model   Qwen/Qwen2.5-7B-Instruct \\
      --train_file  data/processed/additional_split/train.jsonl \\
      --dev_file    data/processed/additional_split/dev.jsonl \\
      --output_dir  checkpoints/sft_additional

  # With task balancing and evidence dropout
  python train_sft.py \\
      --model  Qwen/Qwen2.5-7B-Instruct \\
      --train_file  data/processed/additional_split/train.jsonl \\
      --dev_file    data/processed/additional_split/dev.jsonl \\
      --output_dir  checkpoints/sft_additional \\
      --balance_strategy sqrt \\
      --evidence_dropout 0.15 \\
      --max_steps 3000

  # Evaluate the fine-tuned model (zero-shot with evidence-aware prompting)
  python train_sft.py \\
      --eval_only \\
      --model   checkpoints/sft_additional/best_checkpoint \\
      --test_file   data/processed/additional_split/test.jsonl \\
      --output_dir  checkpoints/sft_additional

Dependencies
------------
  pip install peft transformers datasets trl accelerate bitsandbytes
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Evidence formatters (one per task type) ───────────────────────────────────

def _fmt_list(items: list, sep: str = " → ") -> str:
    return sep.join(str(x) for x in items) if items else ""


def _fmt_conditions(conds: dict) -> str:
    if not conds:
        return ""
    return "; ".join(f"{k}: {v}" for k, v in conds.items() if v)


def _fmt_materials(mats: list | dict) -> str:
    if isinstance(mats, list):
        return ", ".join(str(m) for m in mats) if mats else ""
    if isinstance(mats, dict):
        return "; ".join(f"{k}: {v}" for k, v in mats.items() if v)
    return str(mats) if mats else ""


def _drop_fields(ev: dict, dropout: float, rng: random.Random) -> dict:
    """Randomly zero out evidence fields to simulate imperfect RAG retrieval."""
    if dropout <= 0.0:
        return ev
    return {k: (v if rng.random() > dropout else ([] if isinstance(v, list) else ({} if isinstance(v, dict) else "")))
            for k, v in ev.items()}


def format_evidence(task: str, ev: dict) -> str:
    """
    Convert the `evidence` dict into a human-readable context string.
    Includes full contextual fields per task type to maximise reasoning signal.
    """
    if not ev:
        return ""

    pl = ev.get("process_label", "")
    header = f"Process: {pl}" if pl else ""

    # Step position context (available in most tasks)
    def _step_ctx() -> str:
        si = ev.get("step_index")
        tot = ev.get("total_steps")
        if si is not None and tot:
            return f"Step {si + 1} of {tot}"
        return ""

    if task == "A1_route_retrieval":
        seq = _fmt_list(ev.get("correct_sequence", []))
        precs = ", ".join(ev.get("precursors", []))
        prods = ", ".join(ev.get("products", []))
        parts = [header]
        if seq:
            parts.append(f"Activity sequence: {seq}")
        if precs:
            parts.append(f"Precursors: {precs}")
        if prods:
            parts.append(f"Products: {prods}")
        tot = ev.get("total_steps")
        if tot:
            parts.append(f"Total steps: {tot}")
        return "\n".join(p for p in parts if p)

    elif task == "A2_missing_step":
        parts = [header]
        step_ctx = _step_ctx()
        if step_ctx:
            parts.append(step_ctx)
        # missing_activity is the gold answer label
        missing = ev.get("missing_activity", "")
        if missing:
            parts.append(f"Missing activity: {missing}")
        in_form = ev.get("input_form", "")
        out_form = ev.get("output_form", "")
        if in_form:
            parts.append(f"Input form: {in_form}")
        if out_form:
            parts.append(f"Output form: {out_form}")
        in_mats = ev.get("input_materials") or ev.get("anonymized_inputs", [])
        out_mats = ev.get("output_materials") or ev.get("anonymized_outputs", [])
        if in_mats:
            parts.append(f"Input materials: {_fmt_materials(in_mats)}")
        if out_mats:
            parts.append(f"Output materials: {_fmt_materials(out_mats)}")
        conds = ev.get("conditions", {})
        if conds:
            parts.append(f"Conditions: {_fmt_conditions(conds)}")
        return "\n".join(p for p in parts if p)

    elif task == "A3_next_activity":
        parts = [header]
        step_ctx = _step_ctx()
        if step_ctx:
            parts.append(step_ctx)
        prefix = ev.get("prefix_activities", [])
        nxt = ev.get("next_activity", "")
        if prefix:
            parts.append(f"Completed prefix: {_fmt_list(prefix)}")
        if nxt:
            parts.append(f"Next activity: {nxt}")
        # Current material state
        cur_mats = ev.get("anonymized_current_materials") or ev.get("raw_current_materials", [])
        if cur_mats:
            parts.append(f"Current materials: {_fmt_materials(cur_mats)}")
        next_in = ev.get("anonymized_next_inputs") or ev.get("raw_next_inputs", [])
        next_out = ev.get("anonymized_next_outputs") or ev.get("raw_next_outputs", [])
        if next_in:
            parts.append(f"Next step inputs: {_fmt_materials(next_in)}")
        if next_out:
            parts.append(f"Next step outputs: {_fmt_materials(next_out)}")
        nxt_conds = ev.get("next_conditions", {})
        if nxt_conds:
            parts.append(f"Next step conditions: {_fmt_conditions(nxt_conds)}")
        return "\n".join(p for p in parts if p)

    elif task in ("B1_condition_prediction", "B2_full_condition_set"):
        parts = [header]
        step_ctx = _step_ctx()
        if step_ctx:
            parts.append(step_ctx)
        act = ev.get("activity_label", "")
        if act:
            parts.append(f"Activity: {act}")
        # Full synthesis route for this process (crucial reasoning context)
        route = ev.get("route", [])
        if route:
            parts.append(f"Full synthesis route: {_fmt_list(route)}")
        in_mats = ev.get("input_materials", [])
        out_mats = ev.get("output_materials", [])
        if in_mats:
            parts.append(f"Input materials: {_fmt_materials(in_mats)}")
        if out_mats:
            parts.append(f"Output materials: {_fmt_materials(out_mats)}")
        # Gold conditions (answer-containing for training)
        conds = ev.get("all_conditions") or ev.get("conditions") or {}
        if conds:
            parts.append(f"Conditions: {_fmt_conditions(conds)}")
        return "\n".join(p for p in parts if p)

    elif task == "C1_tool_selection":
        parts = [header]
        step_ctx = _step_ctx()
        if step_ctx:
            parts.append(step_ctx)
        act = ev.get("activity_label", "")
        if act:
            parts.append(f"Activity: {act}")
        route = ev.get("route", [])
        if route:
            parts.append(f"Full synthesis route: {_fmt_list(route)}")
        in_mats = ev.get("input_materials", [])
        out_mats = ev.get("output_materials", [])
        if in_mats:
            parts.append(f"Input materials: {_fmt_materials(in_mats)}")
        if out_mats:
            parts.append(f"Output materials: {_fmt_materials(out_mats)}")
        conds = ev.get("conditions", {})
        if conds:
            parts.append(f"Conditions: {_fmt_conditions(conds)}")
        tools = ev.get("tools", [])
        if tools:
            parts.append(f"Equipment: {', '.join(tools)}")
        return "\n".join(p for p in parts if p)

    elif task == "D1_process_ordering":
        parts = [header]
        seq = ev.get("correct_sequence", [])
        if seq:
            parts.append(f"Correct sequence: {_fmt_list(seq)}")
        # Operation-level I/O details
        op_io = ev.get("operation_io", {})
        if op_io:
            io_lines = []
            for op, details in op_io.items():
                if isinstance(details, dict):
                    ins = _fmt_materials(details.get("inputs", []))
                    outs = _fmt_materials(details.get("outputs", []))
                    if ins or outs:
                        io_lines.append(f"  {op}: {ins} → {outs}")
                else:
                    io_lines.append(f"  {op}: {details}")
            if io_lines:
                parts.append("Operation I/O:\n" + "\n".join(io_lines))
        return "\n".join(p for p in parts if p)

    # Fallback: dump all non-id fields
    parts = [header]
    skip = {"doi", "activity_id", "process_label"}
    for k, v in ev.items():
        if k in skip or not v:
            continue
        if isinstance(v, list):
            parts.append(f"{k}: {', '.join(str(x) for x in v)}")
        elif isinstance(v, dict):
            parts.append(f"{k}: {_fmt_conditions(v)}")
        else:
            parts.append(f"{k}: {v}")
    return "\n".join(p for p in parts if p)


# ── Prompt construction ───────────────────────────────────────────────────────

_SYS_EVIDENCE = (
    "You are an expert assistant in materials science and materials synthesis. "
    "You are provided with factual synthesis process information relevant to the question. "
    "Use this information to answer the multiple-choice question. "
    "Output ONLY the letter (A, B, C, or D) of the correct option. "
    "Do not include any explanation or extra text."
)

_ANSWER_PROMPT = "\n\nAnswer with only the letter (A, B, C, or D):"


def format_choices_block(choices: dict) -> str:
    return "\n".join(f"{l}: {choices[l]}" for l in "ABCD" if l in choices)


def build_training_prompt(
    rec: dict,
    tokenizer,
    include_answer: bool = True,
    evidence_dropout: float = 0.0,
    rng: random.Random | None = None,
) -> str:
    """
    Build the full text for one training example using chat template.
    Evidence context is injected before the question.
    Returns the complete text (prompt + optional answer) for SFT.
    """
    task = rec.get("task", "")
    ev = rec.get("evidence", {})

    if evidence_dropout > 0.0 and rng is not None and include_answer:
        ev = _drop_fields(ev, evidence_dropout, rng)

    ev_text = format_evidence(task, ev)

    q_block = f"{rec['question']}\n\nOptions:\n{format_choices_block(rec['choices'])}"

    if ev_text:
        user_content = (
            f"Synthesis process information:\n\n{ev_text}\n\n"
            f"---\n\n{q_block}{_ANSWER_PROMPT}"
        )
    else:
        user_content = f"{q_block}{_ANSWER_PROMPT}"

    messages = [
        {"role": "system",    "content": _SYS_EVIDENCE},
        {"role": "user",      "content": user_content},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": rec["answer"]})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not include_answer,
    )


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataset(
    records: list[dict],
    balance_strategy: str = "none",
    upsample: dict[str, int] | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Build the training list with optional task balancing and manual upsampling.

    balance_strategy:
      - "none"  : original distribution, just shuffle
      - "sqrt"  : target ∝ sqrt(count) — moderate rebalancing, keeps large-task signal
      - "equal" : equal examples per task (down-samples large tasks)
      - "cap"   : cap each task at median count (light rebalancing)

    upsample: manual per-task repeat factors applied AFTER balancing
    """
    rng = random.Random(seed)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r.get("task", "unknown")].append(r)

    if balance_strategy == "none":
        out = list(records)
    else:
        counts = {t: len(g) for t, g in by_task.items()}
        if balance_strategy == "sqrt":
            max_c = max(counts.values())
            sqrt_max = math.sqrt(max_c)
            targets = {t: max(1, round(sqrt_max * math.sqrt(c)))
                       for t, c in counts.items()}
        elif balance_strategy == "equal":
            min_c = min(counts.values())
            targets = {t: min_c for t in counts}
        elif balance_strategy == "cap":
            sorted_counts = sorted(counts.values())
            median = sorted_counts[len(sorted_counts) // 2]
            targets = {t: min(c, median) for t, c in counts.items()}
        else:
            raise ValueError(f"Unknown balance_strategy: {balance_strategy!r}")

        out = []
        for task, group in by_task.items():
            n = targets[task]
            if n <= len(group):
                out.extend(rng.sample(group, n))
            else:
                # oversample with repetition
                full_repeats = n // len(group)
                remainder = n % len(group)
                out.extend(group * full_repeats)
                out.extend(rng.sample(group, remainder))

    # Manual per-task upsampling (applied on top of balancing)
    if upsample:
        extra: list[dict] = []
        task_groups: dict[str, list[dict]] = defaultdict(list)
        for r in out:
            task_groups[r.get("task", "unknown")].append(r)
        for task, factor in upsample.items():
            if task in task_groups and factor > 1:
                extra.extend(task_groups[task] * (factor - 1))
        out.extend(extra)

    rng.shuffle(out)
    return out


def parse_upsample_arg(s: str) -> dict[str, int]:
    """Parse 'B1_condition_prediction:2,D1_process_ordering:2' → dict."""
    result: dict[str, int] = {}
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            task, factor = part.rsplit(":", 1)
            result[task.strip()] = int(factor.strip())
    return result


def _detect_response_template(tokenizer) -> str:
    """
    Heuristically detect the assistant-turn start token for completion-only loss.
    Falls back to None if unrecognised (disables completion-only mode).
    """
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    if "im_start" in chat_template:          # Qwen2.5 / ChatML
        return "<|im_start|>assistant\n"
    if "start_header_id" in chat_template:   # LLaMA-3
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if "[/INST]" in chat_template:           # Mistral
        return "[/INST]"
    return ""


# ── Dev-set MCQ accuracy callback ────────────────────────────────────────────

def make_dev_accuracy_callback(dev_records, tokenizer, best_ckpt_dir,
                               eval_steps, patience, batch_size):
    """
    TrainerCallback that evaluates MCQ accuracy on dev set every `eval_steps`
    steps, saves the best checkpoint, and triggers early stopping after
    `patience` consecutive evaluations without improvement.
    """
    from transformers import TrainerCallback

    state = {
        "best_acc": -1.0,
        "no_improve": 0,
        "last_eval_step": 0,
    }

    def _run_inference(model) -> float:
        model.eval()
        correct = 0
        device = next(model.parameters()).device

        # Decoder-only models need left-padding for batched generation.
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        for i in range(0, len(dev_records), batch_size):
            batch = dev_records[i: i + batch_size]
            prompts = [
                build_training_prompt(r, tokenizer, include_answer=False)
                for r in batch
            ]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len = enc["input_ids"].shape[1]
            for j, rec in enumerate(batch):
                text = tokenizer.decode(out[j][input_len:], skip_special_tokens=True).strip()
                m = re.search(r'[A-Da-d]', text)
                pred = m.group(0).upper() if m else ""
                if pred == rec.get("answer", ""):
                    correct += 1

        tokenizer.padding_side = original_padding_side
        model.train()
        return correct / len(dev_records) if dev_records else 0.0

    def _maybe_evaluate(step, trainer_state, control, model):
        if step == state["last_eval_step"]:
            return
        state["last_eval_step"] = step

        print(f"\n[dev_callback] Evaluating dev set at step {step} ...")
        acc = _run_inference(model)
        best = state["best_acc"]
        print(f"[dev_callback] Dev MCQ accuracy: {acc:.4f}  (best so far: {best:.4f})")

        trainer_state.log_history.append({"step": step, "dev_mcq_accuracy": acc})

        if acc > state["best_acc"]:
            state["best_acc"] = acc
            state["no_improve"] = 0
            print(f"[dev_callback] New best! Saving to: {best_ckpt_dir}")
            Path(best_ckpt_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_ckpt_dir)
            tokenizer.save_pretrained(best_ckpt_dir)
            with open(f"{best_ckpt_dir}/best_dev_info.json", "w") as fh:
                json.dump({"best_step": step, "dev_mcq_accuracy": acc}, fh, indent=2)
        else:
            state["no_improve"] += 1
            print(f"[dev_callback] No improvement for {state['no_improve']}/{patience} evals.")
            if state["no_improve"] >= patience:
                print("[dev_callback] Early stopping triggered.")
                control.should_training_stop = True

    class _DevAccuracyCallback(TrainerCallback):
        def on_step_end(self, args, trainer_state, control, model=None, **kwargs):  # noqa: ARG002
            step = trainer_state.global_step
            if step > 0 and step % eval_steps == 0:
                _maybe_evaluate(step, trainer_state, control, model)

        def on_train_end(self, args, trainer_state, control, model=None, **kwargs):  # noqa: ARG002
            step = trainer_state.global_step
            if step != state["last_eval_step"]:
                _maybe_evaluate(step, trainer_state, control, model)

    return _DevAccuracyCallback()


# ── Completion-only collator (trl 1.x removed this class) ────────────────────

class DataCollatorForCompletionOnlyLM:
    """Mask prompt tokens from loss; only train on completion tokens.

    Drop-in replacement for the class removed in trl >= 1.0.
    """

    def __init__(self, response_template, tokenizer, ignore_index: int = -100):
        from transformers import DataCollatorForLanguageModeling  # noqa: PLC0415
        self._base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        self.ignore_index = ignore_index
        if isinstance(response_template, str):
            self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        else:
            self.response_token_ids = list(response_template)

    def __call__(self, features):
        import torch  # noqa: PLC0415
        batch = self._base_collator(features)
        tpl = self.response_token_ids
        for i in range(batch["labels"].shape[0]):
            seq = batch["labels"][i].tolist()
            response_start = None
            for j in range(len(seq) - len(tpl), -1, -1):
                if seq[j : j + len(tpl)] == tpl:
                    response_start = j + len(tpl)
                    break
            if response_start is not None:
                batch["labels"][i, :response_start] = self.ignore_index
            else:
                batch["labels"][i] = torch.full_like(batch["labels"][i], self.ignore_index)
        return batch


# ── LoRA fine-tuning ──────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        sys.exit(
            f"Missing dependency: {e}\n"
            "Install with: pip install peft trl datasets accelerate bitsandbytes"
        )

    print(f"[sft] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Quantisation (optional 4-bit for GPU memory savings) ─────────────────
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[sft] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not args.load_in_4bit else None,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA configuration ────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Load dev set ──────────────────────────────────────────────────────────
    dev_records: list[dict] = []
    if args.dev_file:
        print(f"[sft] Loading dev data: {args.dev_file}")
        dev_records = load_jsonl(args.dev_file)
        print(f"[sft] Dev examples: {len(dev_records)}")
    else:
        print("[sft] No --dev_file provided; skipping dev monitoring and early stopping.")

    # ── Build dataset ─────────────────────────────────────────────────────────
    print(f"[sft] Loading training data: {args.train_file}")
    records = load_jsonl(args.train_file)

    upsample = parse_upsample_arg(args.upsample_tasks) if args.upsample_tasks else {}
    train_records = build_dataset(
        records,
        balance_strategy=args.balance_strategy,
        upsample=upsample,
        seed=args.seed,
    )
    print(f"[sft] Training examples after balancing: {len(train_records)}")

    # Per-task count after balancing
    task_counts: dict[str, int] = defaultdict(int)
    for r in train_records:
        task_counts[r.get("task", "unknown")] += 1
    for t, c in sorted(task_counts.items()):
        print(f"  {t}: {c}")

    # Set truncation length via tokenizer (TRL 1.x reads model_max_length)
    tokenizer.model_max_length = args.max_seq_len

    # ── Format prompts (with optional evidence dropout) ───────────────────────
    ev_rng = random.Random(args.seed)
    texts = [
        build_training_prompt(
            r, tokenizer,
            include_answer=True,
            evidence_dropout=args.evidence_dropout,
            rng=ev_rng,
        )
        for r in train_records
    ]
    hf_dataset = Dataset.from_dict({"text": texts})

    # ── Completion-only data collator ─────────────────────────────────────────
    # Compute loss only on the assistant's reply tokens, not the prompt.
    response_template = args.response_template or _detect_response_template(tokenizer)
    data_collator = None
    if response_template:
        print(f"[sft] Completion-only loss enabled. Response template: {response_template!r}")
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )
    else:
        print("[sft] WARNING: Could not detect response template. "
              "Training on full text (prompt + answer). "
              "Pass --response_template explicitly to enable completion-only loss.")

    # ── SFT Training ──────────────────────────────────────────────────────────
    best_ckpt_dir = str(Path(args.output_dir) / "best_checkpoint")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        bf16=True,
        dataset_text_field="text",
        seed=args.seed,
        report_to="none",
    )

    callbacks = []
    if dev_records:
        callbacks.append(make_dev_accuracy_callback(
            dev_records=dev_records,
            tokenizer=tokenizer,
            best_ckpt_dir=best_ckpt_dir,
            eval_steps=args.eval_steps,
            patience=args.early_stopping_patience,
            batch_size=args.dev_batch_size,
        ))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=hf_dataset,
        args=sft_config,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )

    print("[sft] Starting training ...")
    trainer.train()

    if dev_records and Path(best_ckpt_dir).exists():
        print(f"[sft] Best checkpoint already saved to: {best_ckpt_dir}")
        print(f"[sft] Tip: pass --model {best_ckpt_dir} --eval_only for evaluation.")
    else:
        print(f"[sft] Saving final model to: {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    print("[sft] Done.")


# ── Zero-shot evaluation with evidence-aware prompt ──────────────────────────
def evaluate(args: argparse.Namespace) -> None:
    """
    Run evaluation on test.jsonl using the fine-tuned model.
    Uses zero-shot evidence-aware prompting (no RAG retrieval here —
    for RAG evaluation run run_baselines.py rag with the fine-tuned model path).
    """
    from transformers import pipeline as hf_pipeline

    print(f"[eval] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = hf_pipeline(
        "text-generation",
        model=args.model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    test_records = load_jsonl(args.test_file)
    print(f"[eval] {len(test_records)} test records")

    results: list[dict] = []
    correct = 0

    for item in tqdm(test_records, desc="[eval]"):
        prompt = build_training_prompt(item, tokenizer, include_answer=False)
        out = pipe(
            prompt,
            max_new_tokens=8,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        generated = out[0]["generated_text"]
        answer_text = generated[len(prompt):].strip()
        m = re.search(r'[A-Da-d]', answer_text)
        pred = m.group(0).upper() if m else answer_text[:1].upper()

        gt = item.get("answer", "")
        if pred == gt:
            correct += 1

        results.append({
            "qid":          item.get("qid", ""),
            "task":         item.get("task", ""),
            "method":       "sft_evidence",
            "question":     item["question"],
            "model_answer": pred,
            "gt_answer":    gt,
            "choices":      [item["choices"].get(l, "") for l in "ABCD"],
            "answer_index": ord(gt) - ord("A") if gt in "ABCD" else -1,
        })

    acc = correct / len(results) if results else 0.0
    print(f"[eval] MCQ Accuracy: {acc:.4f}  ({correct}/{len(results)})")

    by_task: dict[str, list] = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r["model_answer"] == r["gt_answer"])
    print("[eval] Per-task accuracy:")
    for t, hits in sorted(by_task.items()):
        print(f"  {t:<35} {sum(hits)/len(hits):.4f}  (n={len(hits)})")

    save_path = Path(args.output_dir) / "sft_eval_results.jsonl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[eval] Saved to: {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-Augmented SFT for MatProcBench")
    parser.add_argument("--model",       type=str, required=True,
                        help="HuggingFace model ID or path to fine-tuned checkpoint.")
    parser.add_argument("--train_file",  type=str, default=None,
                        help="Training JSONL (required unless --eval_only).")
    parser.add_argument("--dev_file",    type=str, default=None,
                        help="Dev JSONL for MCQ accuracy monitoring and early stopping.")
    parser.add_argument("--test_file",   type=str, default=None,
                        help="Test JSONL for evaluation.")
    parser.add_argument("--output_dir",  type=str, required=True,
                        help="Directory for checkpoints and eval results.")
    parser.add_argument("--eval_only",   action="store_true",
                        help="Skip training; run evaluation only.")

    # Training hyperparameters (step-based)
    parser.add_argument("--max_steps",    type=int,   default=3000,
                        help="Total number of training steps. "
                             "Default 3000 ≈ 2-3 epochs on additional_split train set "
                             "(effective batch 16, 19k examples).")
    parser.add_argument("--eval_steps",   type=int,   default=200,
                        help="Run dev MCQ evaluation every N steps; also saves checkpoint.")
    parser.add_argument("--warmup_steps", type=int,   default=200,
                        help="Linear warmup over this many steps.")
    parser.add_argument("--logging_steps",type=int,   default=20,
                        help="Log training loss every N steps.")
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--dev_batch_size", type=int, default=8,
                        help="Batch size for dev-set MCQ inference during training.")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Stop after this many consecutive evals without dev accuracy improvement.")
    parser.add_argument("--grad_accum",   type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum).")
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--max_seq_len",  type=int,   default=1024)
    parser.add_argument("--seed",         type=int,   default=42)

    # Task balancing
    parser.add_argument("--balance_strategy", type=str, default="sqrt",
                        choices=["none", "sqrt", "equal", "cap"],
                        help="Task-count balancing strategy. "
                             "'sqrt': target ∝ sqrt(count) — recommended. "
                             "'equal': same N per task (drops large-task data). "
                             "'cap': cap at median count. "
                             "'none': original distribution.")
    parser.add_argument("--upsample_tasks", type=str, default=None,
                        help="Manual per-task repeat factors applied after balancing, "
                             "e.g. 'B1_condition_prediction:2,D1_process_ordering:2'")

    # Evidence dropout
    parser.add_argument("--evidence_dropout", type=float, default=0.0,
                        help="Fraction of evidence fields to randomly zero out during training "
                             "(0.0 = disabled). Helps robustness to imperfect RAG retrieval. "
                             "Recommended range: 0.10–0.20.")

    # Completion-only loss
    parser.add_argument("--response_template", type=str, default=None,
                        help="Assistant-turn start token for completion-only loss. "
                             "Auto-detected from tokenizer if not set. "
                             "Example for Qwen2.5: '<|im_start|>assistant\\n'")

    # LoRA hyperparameters
    parser.add_argument("--lora_r",              type=int,   default=32,
                        help="LoRA rank. Higher = more capacity, more memory.")
    parser.add_argument("--lora_alpha",          type=int,   default=64)
    parser.add_argument("--lora_dropout",        type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated LoRA target module names. "
                             "Includes MLP layers by default for better coverage.")
    parser.add_argument("--load_in_4bit",        action="store_true",
                        help="Load model in 4-bit (QLoRA) for memory efficiency.")

    args = parser.parse_args()

    if not args.eval_only:
        if not args.train_file:
            parser.error("--train_file is required for training.")
        train(args)

    if args.test_file:
        evaluate(args)


if __name__ == "__main__":
    main()
