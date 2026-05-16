from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .compiler import ProcessState
from .utils import (
    base_label,
    extract_int,
    extract_line,
    norm,
    parse_key_values,
    split_csv,
    tokenize_material_list,
)

@dataclass
class StepPrototype:
    label: str
    prev_label: str
    next_label: str
    conditions: dict[str, str]
    tools: list[str]
    inputs: list[str]
    outputs: list[str]
    input_forms: list[str]
    output_forms: list[str]
    position: int
    total_steps: int


class ProcessMemoryIndex:
    """Train-split process memory used for analogy retrieval and symbolic scoring."""

    def __init__(self, processes: list[ProcessState]) -> None:
        self.processes = processes
        self.global_bigrams: Counter[tuple[str, str]] = Counter()
        self.prefix_next: defaultdict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.step_library: list[StepPrototype] = []
        self._build_statistics()

    @classmethod
    def from_files(cls, raw_file: str, train_file: str) -> "ProcessMemoryIndex":
        allowed_keys = cls._load_allowed_keys(train_file)
        processes: list[ProcessState] = []
        with open(raw_file, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (norm(record.get("doi", "")), norm(record.get("label", "")))
                if key not in allowed_keys:
                    continue
                processes.append(ProcessState(record))
        return cls(processes)

    @staticmethod
    def _load_allowed_keys(train_file: str) -> set[tuple[str, str]]:
        allowed: set[tuple[str, str]] = set()
        with open(train_file, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                evidence = record.get("evidence", {})
                doi = evidence.get("doi") or extract_line(record.get("question", ""), "Study DOI:")
                process = evidence.get("process_label") or extract_line(record.get("question", ""), "Process:")
                if doi and process:
                    allowed.add((norm(doi), norm(process)))
        return allowed

    def _build_statistics(self) -> None:
        for process in self.processes:
            route = [base_label(label) for label in process.route_labels(instance_aware=False)]
            for left, right in zip(route, route[1:]):
                self.global_bigrams[(left, right)] += 1
            for k in range(1, len(route)):
                self.prefix_next[tuple(route[:k])][route[k]] += 1

            ordered = process.ordered_activities()
            for idx, (activity_id, activity) in enumerate(ordered, start=1):
                prev_label = ordered[idx - 2][1].label if idx > 1 else "START"
                next_label = ordered[idx][1].label if idx < len(ordered) else "END"
                inputs = [entity.label for entity in process.activity_inputs(activity_id) if entity.label]
                outputs = [entity.label for entity in process.activity_outputs(activity_id) if entity.label]
                input_forms = [entity.form.lower() for entity in process.activity_inputs(activity_id) if entity.form]
                output_forms = [entity.form.lower() for entity in process.activity_outputs(activity_id) if entity.form]
                tools = [entity.label for entity in process.activity_tools(activity_id) if entity.label]
                self.step_library.append(
                    StepPrototype(
                        label=base_label(activity.label),
                        prev_label=base_label(prev_label),
                        next_label=base_label(next_label),
                        conditions={norm(key): value for key, value in activity.conditions.items()},
                        tools=tools,
                        inputs=inputs,
                        outputs=outputs,
                        input_forms=input_forms,
                        output_forms=output_forms,
                        position=idx,
                        total_steps=len(ordered),
                    )
                )

    def _query_activity_labels(self, record: dict[str, Any], task: str) -> list[str]:
        question = record["question"]
        labels: list[str] = []
        if task in {"A2_missing_step", "A2_missing_step_hard"}:
            # Parse all non-masked steps from the partial route line to find analogous processes.
            # Original A2 had no retrieval branch; A2_hard fixes this gap for both variants.
            route_line = extract_line(question, "Synthesis route with one operation masked:")
            chunks = re.split(r"\s*→\s*", route_line)
            for chunk in chunks:
                if "MISSING" not in chunk.upper():
                    lbl = base_label(chunk)
                    if lbl:
                        labels.append(lbl)
        if task in {"A3_next_activity", "A3_next_activity_hard"}:
            line = extract_line(question, "Completed route prefix:")
            labels.extend(base_label(chunk) for chunk in re.findall(r"\d+\.\s*([^→\n]+)", line))
        if task in {"B1_condition_prediction", "B2_full_condition_set", "C1_tool_selection"}:
            line = extract_line(question, "Target step:")
            if "," in line:
                labels.append(base_label(line.split(",", 1)[1]))
            prev_line = extract_line(question, "Previous step:")
            next_line = extract_line(question, "Next step:")
            if prev_line:
                labels.append(base_label(prev_line))
            if next_line:
                labels.append(base_label(next_line))
        if task in {"D1_process_ordering", "D1_process_ordering_hard"}:
            operation_line = extract_line(question, "Operation instances to arrange:")
            labels.extend(base_label(item) for item in split_csv(operation_line))
        return [label for label in labels if label and label not in {"start", "end"}]

    def _query_material_tokens(self, record: dict[str, Any]) -> set[str]:
        question = record["question"]
        fields = [
            extract_line(question, "Precursor materials:"),
            extract_line(question, "Starting material(s):"),
            extract_line(question, "Target inputs:"),
            extract_line(question, "Target outputs:"),
        ]
        tokens: set[str] = set()
        for field in fields:
            tokens |= tokenize_material_list(field)
        return tokens

    def retrieve_processes(self, record: dict[str, Any], task: str, top_k: int = 8) -> list[ProcessState]:
        query_labels = self._query_activity_labels(record, task)
        query_tokens = self._query_material_tokens(record)
        query_len = extract_int(record["question"], r"Recorded route length:\s*(\d+)\s+operations")
        scored: list[tuple[int, ProcessState]] = []
        for process in self.processes:
            route = [base_label(label) for label in process.route_labels(instance_aware=False)]
            score = 0
            score += 3 * len(set(query_labels) & set(route))
            if query_len is not None and len(route) == query_len:
                score += 4
            precursor_tokens: set[str] = set()
            for _entity_id, entity in process.precursors():
                precursor_tokens |= tokenize_material_list(entity.label)
            score += len(query_tokens & precursor_tokens)
            if score > 0:
                scored.append((score, process))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [process for _score, process in scored[:top_k]]


TrainProcessIndex = ProcessMemoryIndex
