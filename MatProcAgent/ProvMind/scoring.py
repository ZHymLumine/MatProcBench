from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .compiler import ProcessState
from .memory import ProcessMemoryIndex, StepPrototype
from .utils import (
    base_label,
    condition_key_from_question,
    extract_int,
    extract_line,
    norm,
    parse_key_values,
    parse_triplet,
    render_sequence,
    sequence_similarity,
    split_sequence,
    tokenize_material_list,
)

class TaskSignalScorer:
    """Task-aware symbolic scorer over train-split process memory."""

    def __init__(self, process_memory: ProcessMemoryIndex) -> None:
        self.process_memory = process_memory

    def _matching_step_contexts(self, record: dict[str, Any]) -> list[tuple[int, StepPrototype]]:
        question = record["question"]
        line = extract_line(question, "Target step:")
        target_label = base_label(line.split(",", 1)[1] if "," in line else "")
        prev_label = base_label(extract_line(question, "Previous step:"))
        next_label = base_label(extract_line(question, "Next step:"))
        conds = parse_key_values(extract_line(question, "Known target-step conditions:"))
        input_tokens = tokenize_material_list(extract_line(question, "Target inputs:"))
        output_tokens = tokenize_material_list(extract_line(question, "Target outputs:"))

        matches: list[tuple[int, StepPrototype]] = []
        for step in self.process_memory.step_library:
            if step.label != target_label:
                continue
            score = 6
            if prev_label and step.prev_label == prev_label:
                score += 2
            if next_label and step.next_label == next_label:
                score += 2
            for key, value in conds.items():
                if norm(step.conditions.get(key, "")) == norm(value):
                    score += 3
            step_input_tokens = tokenize_material_list(", ".join(step.inputs))
            step_output_tokens = tokenize_material_list(", ".join(step.outputs))
            score += len(input_tokens & step_input_tokens)
            score += len(output_tokens & step_output_tokens)
            matches.append((score, step))
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[:50]

    def score(self, task: str, record: dict[str, Any], retrieved: list[ProcessState]) -> dict[str, Any]:
        handler = getattr(self, f"_score_{task}")
        return handler(record, retrieved)

    def _score_A1_route_retrieval(self, record: dict[str, Any], retrieved: list[ProcessState]) -> dict[str, Any]:
        scores: dict[str, float] = {}
        for letter, choice in record["choices"].items():
            sequence = split_sequence(choice)
            score = 0.0
            for left, right in zip(sequence, sequence[1:]):
                score += self.process_memory.global_bigrams.get((left, right), 0)
            for process in retrieved:
                route = [base_label(label) for label in process.route_labels(instance_aware=False)]
                score += max(0, sequence_similarity(sequence, route))
            scores[letter] = score
        return {"task_hint": "Route retrieval from train-process analogs.", "option_scores": scores}

    def _score_A2_missing_step(self, record: dict[str, Any], retrieved: list[ProcessState]) -> dict[str, Any]:
        question = record["question"]
        position = extract_int(question, r"Masked operation position:\s*(\d+)\s+of") or 1
        conds = parse_key_values(extract_line(question, "Known masked-step conditions:"))
        input_forms = [form.lower() for form in re.findall(r"\(([^)]+)\)", extract_line(question, "Masked operation input node(s):"))]
        output_forms = [form.lower() for form in re.findall(r"\(([^)]+)\)", extract_line(question, "Masked operation output node(s):"))]

        label_scores: Counter[str] = Counter()
        for step in self.process_memory.step_library:
            score = 0
            if step.position == position:
                score += 2
            score += 2 * len(set(input_forms) & set(step.input_forms))
            score += 2 * len(set(output_forms) & set(step.output_forms))
            for key, value in conds.items():
                if norm(step.conditions.get(key, "")) == norm(value):
                    score += 3
            if score > 0:
                label_scores[step.label] += score

        scores = {letter: float(label_scores.get(base_label(choice), 0)) for letter, choice in record["choices"].items()}
        return {"task_hint": f"Missing step at position {position}.", "option_scores": scores}

    def _score_A3_next_activity(self, record: dict[str, Any], retrieved: list[ProcessState]) -> dict[str, Any]:
        question = record["question"]
        prefix_line = extract_line(question, "Completed route prefix:")
        prefix = [base_label(chunk) for chunk in re.findall(r"\d+\.\s*([^→\n]+)", prefix_line)]
        prefix = [label for label in prefix if label]
        label_scores: Counter[str] = Counter()

        if tuple(prefix) in self.process_memory.prefix_next:
            label_scores.update(self.process_memory.prefix_next[tuple(prefix)])

        for process in retrieved:
            route = [base_label(label) for label in process.route_labels(instance_aware=False)]
            if len(route) > len(prefix) and route[: len(prefix)] == prefix:
                label_scores[route[len(prefix)]] += 10
            else:
                for k in range(min(len(prefix), len(route) - 1), 0, -1):
                    if route[:k] == prefix[:k]:
                        label_scores[route[k]] += k
                        break

        if prefix:
            last_label = prefix[-1]
            for (left, right), count in self.process_memory.global_bigrams.items():
                if left == last_label:
                    label_scores[right] += count

        scores = {letter: float(label_scores.get(base_label(choice), 0)) for letter, choice in record["choices"].items()}
        return {"task_hint": f"Predict next activity after prefix: {render_sequence(prefix)}.", "option_scores": scores}

    def _score_B1_condition_prediction(self, record: dict[str, Any], retrieved: list[ProcessState]) -> dict[str, Any]:
        cond_key = condition_key_from_question(record["question"])
        value_scores: Counter[str] = Counter()
        for weight, step in self._matching_step_contexts(record):
            value = step.conditions.get(cond_key, "")
            if value:
                value_scores[norm(value)] += weight
        scores = {letter: float(value_scores.get(norm(choice), 0)) for letter, choice in record["choices"].items()}
        return {"task_hint": f"Condition slot to predict: {cond_key}.", "option_scores": scores}

    def _score_B2_full_condition_set(self, record: dict[str, Any], retrieved: list[ProcessState]) -> dict[str, Any]:
        triplet_scores: Counter[tuple[str, str, str]] = Counter()
        for weight, step in self._matching_step_contexts(record):
            triplet = (
                norm(step.conditions.get("temperature", "")),
                norm(step.conditions.get("duration", "")),
                norm(step.conditions.get("atmosphere", "")),
            )
            if all(triplet):
                triplet_scores[triplet] += weight
        scores: dict[str, float] = {}
        for letter, choice in record["choices"].items():
            triplet = parse_triplet(choice)
            key = (
                norm(triplet.get("temperature", "")),
                norm(triplet.get("duration", "")),
                norm(triplet.get("atmosphere", "")),
            )
            scores[letter] = float(triplet_scores.get(key, 0))
        return {"task_hint": "Choose the most plausible full condition set from train analogs.", "option_scores": scores}

    def _score_C1_tool_selection(self, record: dict[str, Any], retrieved: list[ProcessState]) -> dict[str, Any]:
        tool_scores: Counter[str] = Counter()
        for weight, step in self._matching_step_contexts(record):
            if not step.tools:
                continue
            tool_scores[norm(", ".join(sorted(step.tools)))] += weight
            for tool in step.tools:
                tool_scores[norm(tool)] += weight
        scores: dict[str, float] = {}
        for letter, choice in record["choices"].items():
            normalized = norm(choice)
            pieces = [norm(part) for part in choice.split(",") if part.strip()]
            score = float(tool_scores.get(normalized, 0))
            score += sum(tool_scores.get(piece, 0) for piece in pieces)
            scores[letter] = score
        return {"task_hint": "Choose the tool consistent with similar train steps.", "option_scores": scores}

    def _score_D1_process_ordering(self, record: dict[str, Any], retrieved: list[ProcessState]) -> dict[str, Any]:
        edges: list[tuple[str, str, str]] = []
        for line in record["question"].splitlines():
            line = line.strip()
            if not line.startswith("•"):
                continue
            match = re.match(r"•\s+(.+?)\s+produces\s+(M\d+);.+consumed by\s+(.+)", line)
            if not match:
                continue
            producer, alias, consumer = match.groups()
            edges.append((base_label(producer), alias, base_label(consumer)))

        scores: dict[str, float] = {}
        for letter, choice in record["choices"].items():
            candidate = split_sequence(choice)
            candidate_pos = {label: idx for idx, label in enumerate(candidate)}
            satisfied = sum(
                1
                for producer, _alias, consumer in edges
                if producer in candidate_pos and consumer in candidate_pos and candidate_pos[producer] < candidate_pos[consumer]
            )
            scores[letter] = float(satisfied)
        return {"task_hint": "Visible provenance edges define the legal ordering.", "option_scores": scores, "visible_edges": edges}

    def _score_A2_missing_step_hard(self, record: dict[str, Any], retrieved: list[ProcessState]) -> dict[str, Any]:
        """
        Symbolic scorer for A2_missing_step_hard.
        No form extraction from question text (leak removed). Relies on:
          - Step position in the route
          - Labels of the step immediately before and after [MISSING STEP]
        These are parsed from the partial route line, not from form-annotated node refs.
        """
        question = record["question"]
        position = extract_int(question, r"Masked operation position:\s*(\d+)\s+of") or 1

        # Parse before/after step labels from the partial route
        route_line = extract_line(question, "Synthesis route with one operation masked:")
        chunks = re.split(r"\s*→\s*", route_line)
        missing_idx = next((i for i, c in enumerate(chunks) if "MISSING" in c.upper()), None)
        before_labels = [base_label(c) for c in chunks[:missing_idx]] if missing_idx is not None else []
        after_labels  = [base_label(c) for c in chunks[missing_idx + 1:]] if missing_idx is not None else []
        before_labels = [lbl for lbl in before_labels if lbl]
        after_labels  = [lbl for lbl in after_labels if lbl]
        prev_label = before_labels[-1] if before_labels else ""
        next_label = after_labels[0]  if after_labels  else ""

        label_scores: Counter[str] = Counter()
        for step in self.process_memory.step_library:
            score = 0
            if step.position == position:
                score += 2
            if prev_label and step.prev_label == prev_label:
                score += 3
            if next_label and step.next_label == next_label:
                score += 3
            if score > 0:
                label_scores[step.label] += score

        # Also use retrieved processes: check which choice labels appear at the same
        # position in retrieved routes with matching before/after context.
        for process in retrieved:
            route = [base_label(lbl) for lbl in process.route_labels(instance_aware=False)]
            for idx, lbl in enumerate(route):
                if (
                    (not prev_label or (idx > 0 and route[idx - 1] == prev_label))
                    and (not next_label or (idx < len(route) - 1 and route[idx + 1] == next_label))
                ):
                    label_scores[lbl] += 5

        scores = {letter: float(label_scores.get(base_label(choice), 0))
                  for letter, choice in record["choices"].items()}
        return {
            "task_hint": f"Missing step at position {position} (no form hints; using route context).",
            "option_scores": scores,
        }

    def _score_A3_next_activity_hard(self, record: dict[str, Any], retrieved: list[ProcessState]) -> dict[str, Any]:
        """
        Symbolic scorer for A3_next_activity_hard.
        The original A3 scorer does not use the graph signature or conditions —
        it relies solely on prefix-next lookup and bigrams, which are both legitimate.
        This hard variant delegates directly to the original scorer.
        """
        result = self._score_A3_next_activity(record, retrieved)
        result["task_hint"] = result["task_hint"].replace(
            "Predict next activity", "Predict next activity (hard)"
        )
        return result

    def _score_D1_process_ordering_hard(self, record: dict[str, Any], retrieved: list[ProcessState]) -> dict[str, Any]:
        """
        Symbolic scorer for D1_process_ordering_hard.
        No bullet-point edges in the question to parse. Instead, score each candidate
        ordering by:
          1. Global bigram frequency from training data (how often does step A precede step B?)
          2. Sequence similarity against retrieved train-process routes
        This mirrors the A1 route-retrieval strategy and tests genuine route knowledge.
        """
        scores: dict[str, float] = {}

        reference_routes = [
            [base_label(lbl) for lbl in process.route_labels(instance_aware=False)]
            for process in retrieved
        ]

        for letter, choice in record["choices"].items():
            candidate = split_sequence(choice)
            score = 0.0
            # Bigram plausibility from corpus statistics
            for left, right in zip(candidate, candidate[1:]):
                score += self.process_memory.global_bigrams.get((left, right), 0)
            # Sequence similarity against retrieved analogous train processes
            for route in reference_routes:
                score += max(0.0, sequence_similarity(candidate, route))
            scores[letter] = score

        return {
            "task_hint": "No provenance edges; rank orderings by train-process bigrams + route similarity.",
            "option_scores": scores,
        }
