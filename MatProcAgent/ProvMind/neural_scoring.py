"""
neural_scoring.py — Neural and hybrid task scorers for ProvMind.

NeuralTaskScorer replaces string-matching in _matching_step_contexts with
embedding nearest-neighbour lookup over the pre-built step_library embeddings
(stored in DualViewIndex._step_embs).  Route/sequence tasks (A1, A3) are scored
by encoding each answer choice and measuring cosine similarity against the
process-text embedding matrix.

HybridTaskScorer fuses symbolic and neural option scores with configurable
weights.  Each scorer's output is max-normalised before blending so that
count-based (symbolic) and cosine-based (neural) scales are compatible.

ScoringMode literals:
  "symbolic" — original TaskSignalScorer only
  "neural"   — NeuralTaskScorer only
  "hybrid"   — weighted blend (default)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .compiler import ProcessState
from .scoring import TaskSignalScorer
from .utils import (
    base_label,
    condition_key_from_question,
    extract_int,
    extract_line,
    norm,
    parse_triplet,
    render_sequence,
    split_sequence,
    tokenize_material_list,
)

if TYPE_CHECKING:
    from .dual_view_retriever import DualViewIndex

ScoringMode = Literal["symbolic", "neural", "hybrid"]


# ── Query text builders ────────────────────────────────────────────────────────

def _build_step_context_query(record: dict[str, Any]) -> str:
    """Serialise the B1/B2/C1 step context into a string matching _step_to_text format."""
    question = record["question"]
    line = extract_line(question, "Target step:")
    target = base_label(line.split(",", 1)[1] if "," in line else "")
    prev = base_label(extract_line(question, "Previous step:"))
    nxt  = base_label(extract_line(question, "Next step:"))
    inp  = extract_line(question, "Target inputs:")
    out  = extract_line(question, "Target outputs:")
    parts = [f"step: {target}"]
    if prev:
        parts.append(f"prev: {prev}")
    if nxt:
        parts.append(f"next: {nxt}")
    if inp:
        parts.append(f"inputs: {inp}")
    if out:
        parts.append(f"outputs: {out}")
    return ". ".join(parts)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    max_v = max(scores.values(), default=0.0)
    if max_v <= 0.0:
        return dict(scores)
    return {k: v / max_v for k, v in scores.items()}


def _top_k_mean(sims: np.ndarray, k: int) -> float:
    k = min(k, len(sims))
    return float(np.mean(np.partition(sims, -k)[-k:])) if k > 0 else 0.0


# ── Neural scorer ──────────────────────────────────────────────────────────────

class NeuralTaskScorer(TaskSignalScorer):
    """
    Task scorer using pre-computed step and process embeddings.

    B1 / B2 / C1 : embedding NN over step_library replaces string matching.
    A2           : encode each candidate label, score by max sim to step prototypes.
    A1 / A3      : encode each choice sequence/continuation, score by mean top-k
                   cosine similarity against the process-text embedding matrix.
    D1           : unchanged (structural edge-satisfaction, no semantic gain).
    """

    def __init__(self, index: "DualViewIndex", top_k_steps: int = 50) -> None:
        super().__init__(index)
        self._encoder       = index._text_encoder
        self._step_embs     = index._step_embs          # [N_steps, D]
        self._proc_embs     = index._process_text_embs  # [N_proc,  D]
        self._top_k_steps   = top_k_steps

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        emb = self._encoder.encode(text, convert_to_numpy=True).astype(np.float32)
        nrm = np.linalg.norm(emb)
        return emb / max(nrm, 1e-8)

    # ── B1/B2/C1 shared: embedding NN over step library ───────────────────────

    def _neural_matching_step_contexts(
        self, record: dict[str, Any]
    ) -> list[tuple[float, Any]]:
        if self._step_embs.ndim < 2 or self._step_embs.shape[0] == 0:
            return []
        q = self._encode(_build_step_context_query(record))
        sims = (self._step_embs @ q + 1.0) / 2.0          # [N_steps] in [0, 1]
        k = min(self._top_k_steps, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [(float(sims[i]) * 10.0, self.process_memory.step_library[i])
                for i in top_idx]

    def _score_B1_condition_prediction(
        self, record: dict[str, Any], retrieved: list[ProcessState]
    ) -> dict[str, Any]:
        cond_key = condition_key_from_question(record["question"])
        value_scores: Counter[str] = Counter()
        for weight, step in self._neural_matching_step_contexts(record):
            value = step.conditions.get(cond_key, "")
            if value:
                value_scores[norm(value)] += weight
        scores = {
            letter: float(value_scores.get(norm(choice), 0.0))
            for letter, choice in record["choices"].items()
        }
        return {"task_hint": f"[neural] Condition slot: {cond_key}.", "option_scores": scores}

    def _score_B2_full_condition_set(
        self, record: dict[str, Any], retrieved: list[ProcessState]
    ) -> dict[str, Any]:
        triplet_scores: Counter[tuple[str, str, str]] = Counter()
        for weight, step in self._neural_matching_step_contexts(record):
            triplet = (
                norm(step.conditions.get("temperature", "")),
                norm(step.conditions.get("duration", "")),
                norm(step.conditions.get("atmosphere", "")),
            )
            if all(triplet):
                triplet_scores[triplet] += weight
        scores: dict[str, float] = {}
        for letter, choice in record["choices"].items():
            t = parse_triplet(choice)
            key = (
                norm(t.get("temperature", "")),
                norm(t.get("duration", "")),
                norm(t.get("atmosphere", "")),
            )
            scores[letter] = float(triplet_scores.get(key, 0.0))
        return {
            "task_hint": "[neural] Full condition set from embedding-matched steps.",
            "option_scores": scores,
        }

    def _score_C1_tool_selection(
        self, record: dict[str, Any], retrieved: list[ProcessState]
    ) -> dict[str, Any]:
        tool_scores: Counter[str] = Counter()
        for weight, step in self._neural_matching_step_contexts(record):
            if not step.tools:
                continue
            tool_scores[norm(", ".join(sorted(step.tools)))] += weight
            for tool in step.tools:
                tool_scores[norm(tool)] += weight
        scores: dict[str, float] = {}
        for letter, choice in record["choices"].items():
            pieces = [norm(p) for p in choice.split(",") if p.strip()]
            score = float(tool_scores.get(norm(choice), 0.0))
            score += sum(tool_scores.get(p, 0.0) for p in pieces)
            scores[letter] = score
        return {
            "task_hint": "[neural] Tool from embedding-matched steps.",
            "option_scores": scores,
        }

    # ── A2: encode each candidate label, score by max sim to step prototypes ──

    def _score_A2_missing_step(
        self, record: dict[str, Any], retrieved: list[ProcessState]
    ) -> dict[str, Any]:
        question = record["question"]
        position = extract_int(question, r"Masked operation position:\s*(\d+)\s+of") or 1
        has_embs = self._step_embs.ndim == 2 and self._step_embs.shape[0] > 0
        scores: dict[str, float] = {}
        for letter, choice in record["choices"].items():
            if has_embs:
                q = self._encode(f"step: {base_label(choice)} position: {position}")
                score = float(np.max((self._step_embs @ q + 1.0) / 2.0))
            else:
                score = 0.0
            scores[letter] = score
        return {
            "task_hint": f"[neural] Missing step at position {position}.",
            "option_scores": scores,
        }

    # ── A1: encode each candidate route, score by mean top-5 process sim ──────

    def _score_A1_route_retrieval(
        self, record: dict[str, Any], retrieved: list[ProcessState]
    ) -> dict[str, Any]:
        has_embs = self._proc_embs.ndim == 2 and self._proc_embs.shape[0] > 0
        scores: dict[str, float] = {}
        for letter, choice in record["choices"].items():
            if has_embs:
                route_text = "route: " + " → ".join(split_sequence(choice))
                q = self._encode(route_text)
                sims = (self._proc_embs @ q + 1.0) / 2.0
                score = _top_k_mean(sims, 5)
            else:
                score = 0.0
            scores[letter] = score
        return {
            "task_hint": "[neural] Route retrieval by sequence embedding.",
            "option_scores": scores,
        }

    # ── A3: encode prefix+candidate, score by mean top-5 process sim ─────────

    def _score_A3_next_activity(
        self, record: dict[str, Any], retrieved: list[ProcessState]
    ) -> dict[str, Any]:
        question = record["question"]
        prefix_line = extract_line(question, "Completed route prefix:")
        prefix = [base_label(c) for c in re.findall(r"\d+\.\s*([^→\n]+)", prefix_line)]
        prefix = [lbl for lbl in prefix if lbl]
        prefix_str = " → ".join(prefix)
        has_embs = self._proc_embs.ndim == 2 and self._proc_embs.shape[0] > 0
        scores: dict[str, float] = {}
        for letter, choice in record["choices"].items():
            if has_embs:
                continuation = f"route prefix: {prefix_str} → next: {base_label(choice)}"
                q = self._encode(continuation)
                sims = (self._proc_embs @ q + 1.0) / 2.0
                score = _top_k_mean(sims, 5)
            else:
                score = 0.0
            scores[letter] = score
        return {
            "task_hint": f"[neural] Next activity after: {render_sequence(prefix)}.",
            "option_scores": scores,
        }


# ── Hybrid scorer ──────────────────────────────────────────────────────────────

class HybridTaskScorer:
    """
    Linearly fuses symbolic and neural option scores.

    Both score vectors are max-normalised to [0, 1] before blending so that
    the very different magnitude ranges (count-based vs cosine-based) do not
    dominate the mix.
    """

    def __init__(
        self,
        symbolic: TaskSignalScorer,
        neural: NeuralTaskScorer,
        sym_weight: float = 0.5,
        neu_weight: float = 0.5,
    ) -> None:
        self._sym   = symbolic
        self._neu   = neural
        self._sym_w = sym_weight
        self._neu_w = neu_weight

    def score(
        self, task: str, record: dict[str, Any], retrieved: list[ProcessState]
    ) -> dict[str, Any]:
        sym_out = self._sym.score(task, record, retrieved)
        neu_out = self._neu.score(task, record, retrieved)

        sym_norm = _normalize_scores(sym_out["option_scores"])
        neu_norm = _normalize_scores(neu_out["option_scores"])

        letters = set(sym_norm) | set(neu_norm)
        fused = {
            ltr: self._sym_w * sym_norm.get(ltr, 0.0) + self._neu_w * neu_norm.get(ltr, 0.0)
            for ltr in letters
        }
        hint = (f"[hybrid sym={self._sym_w:.2f} neu={self._neu_w:.2f}] "
                f"{sym_out.get('task_hint', '')}")
        return {"task_hint": hint, "option_scores": fused}
