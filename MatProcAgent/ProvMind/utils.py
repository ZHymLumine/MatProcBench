from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def base_label(text: str) -> str:
    text = norm(text)
    text = re.sub(r"[.,;:]+$", "", text)
    text = re.sub(r"\s+#\d+$", "", text)
    return text


def norm_choice(text: str) -> str:
    return norm(text).replace(" ,", ",")


def split_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def split_sequence(text: str) -> list[str]:
    chunks = re.split(r"\s*(?:->|→)\s*", text.strip())
    return [base_label(chunk) for chunk in chunks if chunk.strip()]


def render_sequence(labels: list[str]) -> str:
    return " -> ".join(labels)


def parse_triplet(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in text.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        result[norm(key)] = norm(value)
    return result


def extract_line(question: str, prefix: str) -> str:
    for line in question.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def extract_int(question: str, pattern: str) -> int | None:
    match = re.search(pattern, question)
    return int(match.group(1)) if match else None


def count_numbered_prefix(question: str) -> int | None:
    line = extract_line(question, "Completed route prefix:")
    if not line:
        return None
    hits = re.findall(r"(?:^|[→>])\s*(\d+)\.", line)
    return len(hits) if hits else None


def parse_key_values(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in line.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        result[norm(key)] = value.strip()
    return result


def tokenize_material_list(text: str) -> set[str]:
    out: set[str] = set()
    for item in split_csv(text):
        for token in re.findall(r"[A-Za-z0-9\.\-\+\(\)]+", item):
            if len(token) >= 2:
                out.add(norm(token))
    return out


def condition_key_from_question(question: str) -> str:
    lower = question.lower()
    if "reaction temperature" in lower:
        return "temperature"
    if "reaction duration" in lower:
        return "duration"
    if "required atmosphere" in lower or "gas environment" in lower:
        return "atmosphere"
    if "applied pressure" in lower:
        return "pressure"
    if "rotation / stirring speed" in lower or "rotation speed" in lower:
        return "rotation"
    if "heating or cooling rate" in lower:
        return "temperature_rate"
    raise ValueError("could not infer condition key from question")


def infer_task(record: dict[str, Any]) -> str:
    task = record.get("task", "")
    if task:
        return task
    question = record.get("question", "").lower()
    if "which operation route matches" in question:
        return "A1_route_retrieval"
    if "synthesis route with one operation masked" in question:
        return "A2_missing_step"
    if "which operation label is recorded next" in question:
        return "A3_next_activity"
    if "which complete set of process conditions" in question:
        return "B2_full_condition_set"
    if "which tool, equipment" in question:
        return "C1_tool_selection"
    if "which operation order is consistent" in question:
        return "D1_process_ordering"
    if "what is the" in question and "for this operation" in question:
        return "B1_condition_prediction"
    raise ValueError("could not infer task type")


def sequence_similarity(left: list[str], right: list[str]) -> int:
    score = 0
    for idx, label in enumerate(left):
        if idx < len(right) and label == right[idx]:
            score += 3
    score += len(set(left) & set(right))
    score -= abs(len(left) - len(right))
    return score


def extract_answer_letter(text: str) -> str:
    text = (text or "").strip()
    for pattern in [
        r"answer\s*:\s*([A-D])\b",
        r"option\s*([A-D])\b",
        r"\b([A-D])\b",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
