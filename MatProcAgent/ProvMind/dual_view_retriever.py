"""
dual_view_retriever.py — Dual-view retrieval index for ProvMind.

Extends ProcessMemoryIndex with two neural views:
  - Text view      : SentenceTransformer encodes a text document per process.
  - Structure view : 2-layer frozen GAT with text node features (torch_geometric).

Fusion:
  score = alpha * text_sim + beta * struct_sim + gamma * heuristic_score

retrieval_mode controls which views are active:
  "heuristic"  — original string-matching only (parent class behaviour)
  "text_only"  — text encoder only
  "dual"       — text + structure + heuristic (default)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GATConv, global_mean_pool

from .compiler import ProcessState
from .memory import ProcessMemoryIndex, StepPrototype
from .utils import (
    base_label,
    extract_int,
    extract_line,
    norm,
    split_csv,
    tokenize_material_list,
)

RetrievalMode = Literal["heuristic", "text_only", "dual"]


# ── Process → text ────────────────────────────────────────────────────────────

def _process_to_text(process: ProcessState) -> str:
    parts = [f"Process: {process.label}"]
    precursors = [e.label for _, e in process.precursors() if e.label]
    if precursors:
        parts.append(f"Precursors: {', '.join(precursors)}")
    for act_id, activity in process.ordered_activities():
        inputs  = [e.label for e in process.activity_inputs(act_id)  if e.label]
        outputs = [e.label for e in process.activity_outputs(act_id) if e.label]
        tools   = [e.label for e in process.activity_tools(act_id)   if e.label]
        step = activity.label
        if inputs:
            step += f" (in: {', '.join(inputs)})"
        if outputs:
            step += f" -> {', '.join(outputs)}"
        if activity.conditions:
            step += " [" + ", ".join(f"{k}={v}" for k, v in activity.conditions.items()) + "]"
        if tools:
            step += f" via {', '.join(tools)}"
        parts.append(step)
    return ". ".join(parts)


# ── Step prototype → text ─────────────────────────────────────────────────────

def _step_to_text(step: StepPrototype) -> str:
    parts = [f"step: {step.label}"]
    if step.prev_label and step.prev_label != "START":
        parts.append(f"prev: {step.prev_label}")
    if step.next_label and step.next_label != "END":
        parts.append(f"next: {step.next_label}")
    if step.inputs:
        parts.append(f"inputs: {', '.join(step.inputs)}")
    if step.outputs:
        parts.append(f"outputs: {', '.join(step.outputs)}")
    if step.conditions:
        parts.append("conditions: " + ", ".join(f"{k}={v}" for k, v in step.conditions.items()))
    if step.tools:
        parts.append(f"tools: {', '.join(step.tools)}")
    return ". ".join(parts)


# ── Process → graph nodes/edges ───────────────────────────────────────────────

def _process_to_graph(
    process: ProcessState,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Return (node_texts, undirected_edges) for a train process graph."""
    nodes: list[str] = []
    id_to_idx: dict[str, int] = {}

    for act_id, activity in process.ordered_activities():
        id_to_idx[act_id] = len(nodes)
        cond = " ".join(f"{k}={v}" for k, v in activity.conditions.items())
        nodes.append(f"{activity.label} {cond}".strip())

    for ent_id, entity in process.entities.items():
        id_to_idx[ent_id] = len(nodes)
        extra = " ".join(filter(None, [entity.form, entity.purity]))
        nodes.append(f"{entity.label} {extra}".strip())

    edges: list[tuple[int, int]] = []
    for ent_id, act_id in process.usage:
        if ent_id in id_to_idx and act_id in id_to_idx:
            s, d = id_to_idx[ent_id], id_to_idx[act_id]
            edges += [(s, d), (d, s)]
    for act_id, ent_id in process.generation:
        if act_id in id_to_idx and ent_id in id_to_idx:
            s, d = id_to_idx[act_id], id_to_idx[ent_id]
            edges += [(s, d), (d, s)]

    return nodes, edges


def _query_to_graph(
    record: dict[str, Any],
    task: str,
) -> tuple[list[str], list[tuple[int, int]]]:
    """Build a partial query graph from question text (task-aware)."""
    question = record["question"]
    nodes: list[str] = []
    edges: list[tuple[int, int]] = []

    act_labels: list[str] = []
    if task == "A3_next_activity":
        line = extract_line(question, "Completed route prefix:")
        act_labels = [base_label(c) for c in re.findall(r"\d+\.\s*([^→\n]+)", line)]

    elif task in {"B1_condition_prediction", "B2_full_condition_set", "C1_tool_selection"}:
        target_line = extract_line(question, "Target step:")
        target = base_label(target_line.split(",", 1)[1] if "," in target_line else target_line)
        prev   = base_label(extract_line(question, "Previous step:"))
        nxt    = base_label(extract_line(question, "Next step:"))
        act_labels = [lbl for lbl in [prev, target, nxt] if lbl]

    elif task == "D1_process_ordering":
        op_line = extract_line(question, "Operation instances to arrange:")
        act_labels = [base_label(item) for item in split_csv(op_line)]

    elif task == "A2_missing_step":
        prev = base_label(extract_line(question, "Previous recorded operation:"))
        nxt  = base_label(extract_line(question, "Next recorded operation:"))
        act_labels = [lbl for lbl in [prev, "[MASKED]", nxt] if lbl]

    if not act_labels:
        mat = (
            extract_line(question, "Precursor materials:")
            or extract_line(question, "Starting material(s):")
        )
        if mat:
            nodes.append(mat)
        return nodes, edges

    for i, label in enumerate(act_labels):
        nodes.append(label)
        if i > 0:
            edges += [(i - 1, i), (i, i - 1)]

    if task in {"B1_condition_prediction", "B2_full_condition_set", "C1_tool_selection"}:
        target_idx = min(1, len(nodes) - 1)
        for field in ["Target inputs:", "Target outputs:"]:
            mat_line = extract_line(question, field)
            if mat_line:
                m_idx = len(nodes)
                nodes.append(mat_line)
                edges += [(m_idx, target_idx), (target_idx, m_idx)]

    return nodes, edges


# ── 2-layer frozen GAT ────────────────────────────────────────────────────────

class _FrozenTwoLayerGAT(nn.Module):
    """2-layer GAT with randomly-initialised, permanently frozen weights.

    Even with random weights, neighbourhood aggregation encodes structural
    patterns on top of the text node features. Seed is fixed at construction
    so embeddings are reproducible across runs.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        heads: int = 4,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True,
                             add_self_loops=True, dropout=0.0)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False,
                             add_self_loops=True, dropout=0.0)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return global_mean_pool(x, batch)   # [num_graphs, out_dim]


# ── Main class ────────────────────────────────────────────────────────────────

class DualViewIndex(ProcessMemoryIndex):
    """
    ProcessMemoryIndex extended with dual-view neural retrieval.

    Inherits all symbolic-scoring statistics (bigrams, step_library, etc.)
    from the parent class unchanged — only retrieve_processes() is replaced.
    """

    def __init__(
        self,
        processes: list[ProcessState],
        text_encoder_name: str = "all-mpnet-base-v2",
        retrieval_mode: RetrievalMode = "dual",
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        gat_hidden_dim: int = 256,
        gat_heads: int = 4,
        gat_out_dim: int = 256,
        cache_dir: str | None = None,
        batch_size: int = 64,
    ) -> None:
        super().__init__(processes)

        self.retrieval_mode = retrieval_mode
        self._alpha = alpha
        self._beta  = beta
        self._gamma = gamma
        self._cache_dir  = Path(cache_dir) if cache_dir else None
        self._batch_size = batch_size
        self._text_encoder_name = text_encoder_name

        self._process_text_embs: np.ndarray   = np.empty((0,), dtype=np.float32)
        self._process_struct_embs: np.ndarray = np.empty((0,), dtype=np.float32)
        self._step_embs: np.ndarray           = np.empty((0,), dtype=np.float32)
        self._gat: _FrozenTwoLayerGAT | None  = None

        if retrieval_mode == "heuristic":
            return

        print(f"[DualViewIndex] Loading text encoder: {text_encoder_name}")
        self._text_encoder = SentenceTransformer(text_encoder_name)
        self._build_text_index()
        self._build_step_index()

        if retrieval_mode == "dual":
            self._build_struct_index(gat_hidden_dim, gat_heads, gat_out_dim)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_files(  # type: ignore[override]
        cls,
        raw_file: str,
        train_file: str,
        text_encoder_name: str = "all-mpnet-base-v2",
        retrieval_mode: RetrievalMode = "dual",
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        gat_hidden_dim: int = 256,
        gat_heads: int = 4,
        gat_out_dim: int = 256,
        cache_dir: str | None = None,
        batch_size: int = 64,
    ) -> "DualViewIndex":
        allowed_keys = cls._load_allowed_keys(train_file)
        processes: list[ProcessState] = []
        with open(raw_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (norm(rec.get("doi", "")), norm(rec.get("label", "")))
                if key in allowed_keys:
                    processes.append(ProcessState(rec))
        return cls(
            processes=processes,
            text_encoder_name=text_encoder_name,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            gat_hidden_dim=gat_hidden_dim,
            gat_heads=gat_heads,
            gat_out_dim=gat_out_dim,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, tag: str) -> Path | None:
        if self._cache_dir is None:
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        safe = tag.replace("/", "_").replace(" ", "_")
        return self._cache_dir / f"{safe}.npy"

    # ── Index construction ────────────────────────────────────────────────────

    def _build_text_index(self) -> None:
        enc_tag = self._text_encoder_name.replace("/", "_")
        cache = self._cache_path(f"text_{enc_tag}_{len(self.processes)}")
        if cache and cache.exists():
            print(f"[DualViewIndex] Loading text embeddings from cache: {cache}")
            self._process_text_embs = np.load(str(cache))
            return

        print(f"[DualViewIndex] Encoding {len(self.processes)} process documents …")
        texts = [_process_to_text(p) for p in self.processes]
        embs = self._text_encoder.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        self._process_text_embs = _l2_normalize(embs)

        if cache:
            np.save(str(cache), self._process_text_embs)
            print(f"[DualViewIndex] Text embeddings cached → {cache}")

    def _build_step_index(self) -> None:
        enc_tag = self._text_encoder_name.replace("/", "_")
        n_steps = len(self.step_library)
        cache = self._cache_path(f"steps_{enc_tag}_{n_steps}")
        if cache and cache.exists():
            print(f"[DualViewIndex] Loading step embeddings from cache: {cache}")
            self._step_embs = np.load(str(cache))
            return

        print(f"[DualViewIndex] Encoding {n_steps} step prototypes …")
        if n_steps == 0:
            feat_dim = self._process_text_embs.shape[1] if self._process_text_embs.ndim == 2 else 768
            self._step_embs = np.empty((0, feat_dim), dtype=np.float32)
            return

        step_texts = [_step_to_text(s) for s in self.step_library]
        embs = self._text_encoder.encode(
            step_texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        self._step_embs = _l2_normalize(embs)

        if cache:
            np.save(str(cache), self._step_embs)
            print(f"[DualViewIndex] Step embeddings cached → {cache}")

    def _build_struct_index(self, hidden_dim: int, heads: int, out_dim: int) -> None:
        in_dim = self._process_text_embs.shape[1]
        cache = self._cache_path(
            f"struct_pyg_{in_dim}_{hidden_dim}_{heads}_{out_dim}_{len(self.processes)}"
        )
        if cache and cache.exists():
            print(f"[DualViewIndex] Loading struct embeddings from cache: {cache}")
            self._process_struct_embs = np.load(str(cache))
            # GAT is still needed at query time even when train embeddings are cached
            torch.manual_seed(42)
            self._gat = _FrozenTwoLayerGAT(in_dim, hidden_dim, heads, out_dim)
            self._gat.eval()
            return

        print(f"[DualViewIndex] Building structure embeddings (2-layer GAT) "
              f"for {len(self.processes)} processes …")

        # Collect all node texts in one pass across all processes
        all_node_texts: list[str] = []
        process_ranges: list[tuple[int, int]] = []
        process_edge_lists: list[list[tuple[int, int]]] = []

        for process in self.processes:
            node_texts, edges = _process_to_graph(process)
            start = len(all_node_texts)
            all_node_texts.extend(node_texts)
            process_ranges.append((start, len(all_node_texts)))
            process_edge_lists.append(edges)

        # Encode all node texts in a single batch
        all_node_embs = self._text_encoder.encode(
            all_node_texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32) if all_node_texts else np.empty((0, in_dim), dtype=np.float32)

        torch.manual_seed(42)
        self._gat = _FrozenTwoLayerGAT(in_dim, hidden_dim, heads, out_dim)
        self._gat.eval()

        embs = self._encode_graphs_pyg(all_node_embs, process_ranges, process_edge_lists, out_dim)
        self._process_struct_embs = _l2_normalize(embs)

        if cache:
            np.save(str(cache), self._process_struct_embs)
            print(f"[DualViewIndex] Struct embeddings cached → {cache}")

    # ── Graph encoding helpers ────────────────────────────────────────────────

    def _encode_graphs_pyg(
        self,
        all_node_embs: np.ndarray,
        process_ranges: list[tuple[int, int]],
        process_edge_lists: list[list[tuple[int, int]]],
        out_dim: int,
    ) -> np.ndarray:
        results: list[np.ndarray] = []
        for (start, end), edges in zip(process_ranges, process_edge_lists):
            node_embs = all_node_embs[start:end]
            n = len(node_embs)
            if n == 0:
                results.append(np.zeros(out_dim, dtype=np.float32))
                continue

            x = torch.tensor(node_embs)
            if edges:
                src = [e[0] for e in edges]
                dst = [e[1] for e in edges]
                edge_index = torch.tensor([src, dst], dtype=torch.long)
            else:
                idx = list(range(n))
                edge_index = torch.tensor([idx, idx], dtype=torch.long)

            batch = torch.zeros(n, dtype=torch.long)
            with torch.no_grad():
                out = self._gat(x, edge_index, batch)
            results.append(out.squeeze(0).numpy())

        return (np.array(results, dtype=np.float32)
                if results else np.zeros((0, out_dim), dtype=np.float32))

    # ── Query encoding ────────────────────────────────────────────────────────

    def _encode_query_text(self, record: dict[str, Any]) -> np.ndarray:
        emb = self._text_encoder.encode(
            record["question"], convert_to_numpy=True
        ).astype(np.float32)
        return _l2_normalize(emb[None])[0]

    def _encode_query_struct(self, record: dict[str, Any], task: str) -> np.ndarray | None:
        node_texts, edges = _query_to_graph(record, task)
        if not node_texts:
            return None

        node_embs = self._text_encoder.encode(
            node_texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        n = len(node_texts)
        x = torch.tensor(node_embs)
        if edges:
            src = [e[0] for e in edges]
            dst = [e[1] for e in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            idx = list(range(n))
            edge_index = torch.tensor([idx, idx], dtype=torch.long)

        batch = torch.zeros(n, dtype=torch.long)
        with torch.no_grad():
            out = self._gat(x, edge_index, batch)
        emb = out.squeeze(0).numpy().astype(np.float32)
        return _l2_normalize(emb[None])[0]

    # ── Heuristic scores over all train processes ─────────────────────────────

    def _heuristic_scores_array(self, record: dict[str, Any], task: str) -> np.ndarray:
        query_labels = self._query_activity_labels(record, task)
        query_tokens = self._query_material_tokens(record)
        query_len    = extract_int(
            record["question"],
            r"Recorded route length:\s*(\d+)\s+operations",
        )
        scores = np.zeros(len(self.processes), dtype=np.float32)
        for i, process in enumerate(self.processes):
            route = [base_label(lbl) for lbl in process.route_labels(instance_aware=False)]
            s = 3 * len(set(query_labels) & set(route))
            if query_len is not None and len(route) == query_len:
                s += 4
            prec_tokens: set[str] = set()
            for _, entity in process.precursors():
                prec_tokens |= tokenize_material_list(entity.label)
            s += len(query_tokens & prec_tokens)
            scores[i] = float(s)
        max_s = scores.max()
        if max_s > 0:
            scores /= max_s
        return scores

    # ── Public retrieval interface ────────────────────────────────────────────

    def retrieve_processes(
        self,
        record: dict[str, Any],
        task: str,
        top_k: int = 8,
    ) -> list[ProcessState]:
        if self.retrieval_mode == "heuristic":
            return ProcessMemoryIndex.retrieve_processes(self, record, task, top_k)

        N = len(self.processes)
        if N == 0:
            return []

        combined    = np.zeros(N, dtype=np.float32)
        weight_used = 0.0

        if self.retrieval_mode in ("text_only", "dual"):
            q_text   = self._encode_query_text(record)
            text_sim = (self._process_text_embs @ q_text + 1.0) / 2.0
            combined    += self._alpha * text_sim
            weight_used += self._alpha

        if self.retrieval_mode == "dual":
            q_struct = self._encode_query_struct(record, task)
            if q_struct is not None and self._process_struct_embs.ndim == 2:
                struct_sim   = (self._process_struct_embs @ q_struct + 1.0) / 2.0
                combined    += self._beta * struct_sim
                weight_used += self._beta

            heuristic    = self._heuristic_scores_array(record, task)
            combined    += self._gamma * heuristic
            weight_used += self._gamma

        if weight_used > 0:
            combined /= weight_used

        indices = np.argsort(combined)[::-1][:top_k]
        return [self.processes[i] for i in indices]


# ── Utility ───────────────────────────────────────────────────────────────────

def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)
