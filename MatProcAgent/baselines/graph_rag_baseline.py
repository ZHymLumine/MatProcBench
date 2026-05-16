"""
graph_rag_baseline.py – GraphRAG baseline for MatProcBench
===========================================================
Builds a Process Knowledge Graph from:
  (a) raw MatPROV.jsonl  (preferred – full provenance graph with all matprov:
      attributes defined in context.jsonld)
  (b) processed train.jsonl  (fallback / supplement – QA evidence fields)

Node types
----------
  ActivityNode  – one synthesis step; carries all matprov: condition attributes
                  (temperature, duration, atmosphere, pressure, rotation,
                   temperature_rate, concentration, length_thickness,
                   length_diameter, mass, …) plus a list of tool names
  MaterialNode  – one material or tool entity; carries form, purity, mass,
                  concentration, length_diameter, length_thickness

Edge types
----------
  activity → material   (Generation: activity produced this material)
  material → activity   (Usage: activity consumed this material)
  activity → activity   (temporal ordering derived from the DAG)
  activity → tool       (Usage of a tool entity)

Retrieval
---------
Each ActivityNode is embedded using a rich text representation that includes:
  the activity label, all conditions, input/output material names+forms,
  predecessor/successor activity labels, and tool names.

The FAISS index is built over these node embeddings.  At query time the test
question is embedded and matched against activity nodes; the top-K local
neighbourhoods are retrieved and formatted as structured context.

Output JSONL is fully compatible with eval.py.

Usage
-----
  # With raw data (recommended – richer graph)
  python graph_rag_baseline.py \\
      --model  Qwen/Qwen2.5-7B-Instruct \\
      --data_file  /path/to/test.jsonl \\
      --train_file /path/to/train.jsonl \\
      --raw_file   /path/to/MatPROV.jsonl \\
      --save_file  results/graph_rag.jsonl

  # Without raw data (uses QA evidence only)
  python graph_rag_baseline.py \\
      --model  Qwen/Qwen2.5-7B-Instruct \\
      --data_file  /path/to/test.jsonl \\
      --train_file /path/to/train.jsonl \\
      --save_file  results/graph_rag.jsonl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline

sys.path.insert(0, str(Path(__file__).parent))
from run_baselines import (
    load_jsonl,
    write_jsonl,
    choices_to_list,
    letter_to_index,
    format_question_block,
    run_inference,
    _ANSWER_PROMPT,
)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYS_GRAPH_RAG = (
    "You are an expert assistant in materials science and materials synthesis. "
    "You are provided with structured synthesis process knowledge retrieved from "
    "a provenance graph of materials synthesis procedures. "
    "Each retrieved process shows the activity sequence, material flows with "
    "physical forms and purity, process conditions, and equipment used. "
    "Use this structured knowledge to answer the multiple-choice question. "
    "Output ONLY the letter (A, B, C, or D) of the correct option. "
    "Do not include any explanation or extra text."
)

# ── All matprov: condition keys (from context.jsonld + data survey) ────────────

# Activity-level condition attributes
_ACT_COND_KEYS: dict[str, str] = {
    "matprov:temperature":       "temperature",
    "matprov:temperature_start": "temperature_start",
    "matprov:temperature_end":   "temperature_end",
    "matprov:temperature_rate":  "temperature_rate",
    "matprov:duration":          "duration",
    "matprov:pressure":          "pressure",
    "matprov:pressure_start":    "pressure_start",
    "matprov:pressure_end":      "pressure_end",
    "matprov:atmosphere":        "atmosphere",
    "matprov:atmosphere_start":  "atmosphere_start",
    "matprov:rotation":          "rotation",
    "matprov:rotation_rate":     "rotation_rate",
    "matprov:concentration":     "concentration",
    "matprov:mass":              "mass",
    "matprov:length_thickness":  "length_thickness",
    "matprov:length_diameter":   "length_diameter",
    "matprov:length":            "length",
    "matprov:length_width":      "length_width",
    "matprov:length_height":     "length_height",
    "matprov:rate":              "rate",
    "matprov:heating_rate":      "heating_rate",
    "matprov:form":              "form",         # rare on activities
}

# Entity-level physical attributes
_ENT_ATTR_KEYS: dict[str, str] = {
    "matprov:form":              "form",
    "matprov:purity":            "purity",
    "matprov:mass":              "mass",
    "matprov:concentration":     "concentration",
    "matprov:length_diameter":   "length_diameter",
    "matprov:length_thickness":  "length_thickness",
    "matprov:length":            "length",
    "matprov:length_width":      "length_width",
    "matprov:length_height":     "length_height",
}


# ── Node dataclasses ──────────────────────────────────────────────────────────

class MaterialNode:
    """Physical entity (precursor, intermediate, product, or tool)."""

    __slots__ = (
        "node_id", "label", "entity_type",
        "form", "purity", "mass", "concentration",
        "length_diameter", "length_thickness", "length",
        "length_width", "length_height",
    )

    def __init__(self, node_id: str, label: str, entity_type: str = "material") -> None:
        self.node_id = node_id
        self.label = label
        self.entity_type = entity_type   # "material" | "tool"
        self.form = ""
        self.purity = ""
        self.mass = ""
        self.concentration = ""
        self.length_diameter = ""
        self.length_thickness = ""
        self.length = ""
        self.length_width = ""
        self.length_height = ""

    def update_from_dict(self, attrs: dict[str, str]) -> None:
        for k, v in attrs.items():
            if v and hasattr(self, k):
                setattr(self, k, v)

    def summary(self) -> str:
        """Compact text for embedding / context formatting."""
        parts = [self.label]
        if self.form:
            parts.append(f"form={self.form}")
        if self.purity:
            parts.append(f"purity={self.purity}")
        if self.concentration:
            parts.append(f"conc={self.concentration}")
        if self.mass:
            parts.append(f"mass={self.mass}")
        size = self.length_diameter or self.length_thickness or self.length
        if size:
            parts.append(f"size={size}")
        return " ".join(parts)


class ActivityNode:
    """One synthesis step in the process graph."""

    __slots__ = (
        "node_id", "label", "conditions",
        "process_label", "doi", "task",
        "in_mat_ids", "out_mat_ids",      # MaterialNode ids
        "tool_ids",                        # MaterialNode ids (tools)
        "pred_activities", "succ_activities",
    )

    def __init__(
        self,
        node_id: str,
        label: str,
        conditions: dict[str, str],
        process_label: str,
        doi: str,
        task: str,
    ) -> None:
        self.node_id = node_id
        self.label = label
        self.conditions: dict[str, str] = conditions
        self.process_label = process_label
        self.doi = doi
        self.task = task
        self.in_mat_ids: list[str] = []
        self.out_mat_ids: list[str] = []
        self.tool_ids: list[str] = []
        self.pred_activities: list[str] = []
        self.succ_activities: list[str] = []


# ── helpers ───────────────────────────────────────────────────────────────────

def _val(field: Any) -> str:
    """Extract string value from a JSON-LD value field."""
    if isinstance(field, list):
        for item in field:
            if isinstance(item, dict):
                v = item.get("@value", "")
                if v:
                    return str(v).strip()
            elif isinstance(item, str) and item.strip():
                return item.strip()
    if isinstance(field, dict):
        return str(field.get("@value", "")).strip()
    if isinstance(field, str):
        return field.strip()
    return ""


# ── Knowledge graph ───────────────────────────────────────────────────────────

class ProcessKnowledgeGraph:
    """
    Two-source graph builder:

    1. ``build_from_raw(raw_records, train_dois)``
       Parses MatPROV prov_jsonld graphs directly, extracting every matprov:
       attribute for both Activity and Entity nodes.  Only processes whose DOI
       is in train_dois are ingested.

    2. ``ingest_qa_records(train_qa)``
       Supplements the graph using QA evidence fields (C2 form, C3 tools, etc.)
       for any attributes not already captured from the raw graphs.
    """

    def __init__(self) -> None:
        self.act_nodes: dict[str, ActivityNode] = {}    # act_node_id → ActivityNode
        self.mat_nodes: dict[str, MaterialNode] = {}    # mat_node_id → MaterialNode

        self._act_counter = 0
        self._mat_counter = 0

        # Dedup keys
        self._proc_act_idx: dict[tuple[str, str], str] = {}  # (pl, act_lbl) → act_node_id
        self._proc_mat_idx: dict[tuple[str, str], str] = {}  # (pl, mat_lbl) → mat_node_id

        self._process_act_nodes: dict[str, list[str]] = defaultdict(list)
        self._doi_processes: dict[str, set[str]] = defaultdict(set)  # doi → {process_labels}

    # ── id generators ─────────────────────────────────────────────────────────

    def _new_act_id(self) -> str:
        nid = f"a{self._act_counter}"
        self._act_counter += 1
        return nid

    def _new_mat_id(self) -> str:
        nid = f"m{self._mat_counter}"
        self._mat_counter += 1
        return nid

    # ── activity node CRUD ────────────────────────────────────────────────────

    def _get_or_create_act(
        self,
        pl: str,
        act_lbl: str,
        doi: str,
        task: str,
        conditions: dict[str, str] | None = None,
    ) -> str:
        key = (pl, act_lbl)
        if key in self._proc_act_idx:
            nid = self._proc_act_idx[key]
            if conditions:
                self.act_nodes[nid].conditions.update(
                    {k: v for k, v in conditions.items() if v}
                )
            return nid
        nid = self._new_act_id()
        self.act_nodes[nid] = ActivityNode(nid, act_lbl, dict(conditions or {}), pl, doi, task)
        self._proc_act_idx[key] = nid
        self._process_act_nodes[pl].append(nid)
        self._doi_processes[doi].add(pl)
        return nid

    # ── material node CRUD ────────────────────────────────────────────────────

    def _get_or_create_mat(
        self,
        pl: str,
        mat_lbl: str,
        entity_type: str = "material",
        attrs: dict[str, str] | None = None,
    ) -> str:
        key = (pl, mat_lbl)
        if key in self._proc_mat_idx:
            nid = self._proc_mat_idx[key]
            if attrs:
                self.mat_nodes[nid].update_from_dict(attrs)
            return nid
        nid = self._new_mat_id()
        node = MaterialNode(nid, mat_lbl, entity_type)
        if attrs:
            node.update_from_dict(attrs)
        self.mat_nodes[nid] = node
        self._proc_mat_idx[key] = nid
        return nid

    # ── sequence linking ──────────────────────────────────────────────────────

    def _link_sequence(self, act_ids: list[str]) -> None:
        for i in range(len(act_ids) - 1):
            a, b = act_ids[i], act_ids[i + 1]
            if b not in self.act_nodes[a].succ_activities:
                self.act_nodes[a].succ_activities.append(b)
            if a not in self.act_nodes[b].pred_activities:
                self.act_nodes[b].pred_activities.append(a)

    # ═════════════════════════════════════════════════════════════════════════
    # SOURCE 1 – raw MatPROV.jsonl
    # ═════════════════════════════════════════════════════════════════════════

    @classmethod
    def build_from_raw(
        cls,
        raw_records: list[dict],
        train_dois: set[str] | None = None,
    ) -> "ProcessKnowledgeGraph":
        """
        Parse prov_jsonld graphs from MatPROV.jsonl.
        If train_dois is given, only processes from those DOIs are ingested.
        """
        g = cls()
        for rec in raw_records:
            doi = rec.get("doi", "")
            if train_dois and doi not in train_dois:
                continue
            pl = rec.get("label", "")
            if not pl:
                continue
            graph = rec.get("prov_jsonld", {}).get("@graph", [])
            g._ingest_raw_graph(graph, doi, pl)
        return g

    def _ingest_raw_graph(self, graph: list, doi: str, pl: str) -> None:
        """Parse one prov_jsonld @graph list into nodes and edges."""
        # ── Pass 1: create nodes ──────────────────────────────────────────────
        raw_eid_to_mat: dict[str, str] = {}   # raw @id → mat_node_id
        raw_aid_to_act: dict[str, str] = {}   # raw @id → act_node_id

        for item in graph:
            if not isinstance(item, dict):
                continue
            t = item.get("@type", "")
            raw_id = item.get("@id", "")

            if t == "Entity":
                lbl = _val(item.get("label", ""))
                if not lbl:
                    continue
                entity_type_raw = _val(item.get("type", "material"))
                etype = "tool" if entity_type_raw == "tool" else "material"
                # Extract all physical attributes
                attrs: dict[str, str] = {}
                for mk, sk in _ENT_ATTR_KEYS.items():
                    v = _val(item.get(mk, ""))
                    if v:
                        attrs[sk] = v
                nid = self._get_or_create_mat(pl, lbl, etype, attrs)
                raw_eid_to_mat[raw_id] = nid

            elif t == "Activity":
                lbl = _val(item.get("label", ""))
                if not lbl:
                    continue
                # Extract all condition attributes
                conds: dict[str, str] = {}
                for mk, sk in _ACT_COND_KEYS.items():
                    v = _val(item.get(mk, ""))
                    if v:
                        conds[sk] = v
                nid = self._get_or_create_act(pl, lbl, doi, "raw", conds)
                raw_aid_to_act[raw_id] = nid

        # ── Pass 2: wire edges ────────────────────────────────────────────────
        for item in graph:
            if not isinstance(item, dict):
                continue
            t = item.get("@type", "")

            if t == "Usage":
                # activity consumes entity
                raw_aid = item.get("activity", "")
                raw_eid = item.get("entity", "")
                act_nid = raw_aid_to_act.get(raw_aid)
                mat_nid = raw_eid_to_mat.get(raw_eid)
                if act_nid and mat_nid:
                    mn = self.mat_nodes[mat_nid]
                    if mn.entity_type == "tool":
                        if mat_nid not in self.act_nodes[act_nid].tool_ids:
                            self.act_nodes[act_nid].tool_ids.append(mat_nid)
                    else:
                        if mat_nid not in self.act_nodes[act_nid].in_mat_ids:
                            self.act_nodes[act_nid].in_mat_ids.append(mat_nid)

            elif t == "Generation":
                # activity produced entity
                raw_aid = item.get("activity", "")
                raw_eid = item.get("entity", "")
                act_nid = raw_aid_to_act.get(raw_aid)
                mat_nid = raw_eid_to_mat.get(raw_eid)
                if act_nid and mat_nid:
                    if mat_nid not in self.act_nodes[act_nid].out_mat_ids:
                        self.act_nodes[act_nid].out_mat_ids.append(mat_nid)

        # ── Pass 3: derive activity ordering from material flow ───────────────
        # Operate only on the activities just ingested (O(M) not O(total_nodes)).
        cur_act_nids = list(raw_aid_to_act.values())

        gen_map: dict[str, str] = {}   # mat_nid → act_nid that produced it
        for act_nid in cur_act_nids:
            for mat_nid in self.act_nodes[act_nid].out_mat_ids:
                gen_map[mat_nid] = act_nid

        for act_nid in cur_act_nids:
            an = self.act_nodes[act_nid]
            for mat_nid in an.in_mat_ids:
                pred_nid = gen_map.get(mat_nid)
                if pred_nid and pred_nid != act_nid:
                    if act_nid not in self.act_nodes[pred_nid].succ_activities:
                        self.act_nodes[pred_nid].succ_activities.append(act_nid)
                    if pred_nid not in an.pred_activities:
                        an.pred_activities.append(pred_nid)

    # ═════════════════════════════════════════════════════════════════════════
    # SOURCE 2 – processed QA records (evidence fields)
    # ═════════════════════════════════════════════════════════════════════════

    def ingest_qa_records(self, train_records: list[dict]) -> None:
        """
        Supplement the graph using evidence fields in processed QA records.
        Used either as the sole source (no raw file) or to fill gaps.
        """
        for rec in train_records:
            ev = rec.get("evidence", {})
            task = rec.get("task", "")
            doi = ev.get("doi", "")
            pl = ev.get("process_label", "")
            if not pl:
                continue

            if task == "A1_route_retrieval":
                self._qa_A1(ev, doi, pl, task)
            elif task == "A3_next_activity":
                self._qa_A3(ev, doi, pl, task)
            elif task in ("B1_condition_prediction", "B2_full_condition_set"):
                self._qa_B(ev, doi, pl, task)
            elif task == "A2_missing_step":
                self._qa_A2(ev, doi, pl, task)
            elif task == "C1_tool_selection":
                self._qa_C1(ev, doi, pl, task)
            elif task == "D1_process_ordering":
                self._qa_D1(ev, doi, pl, task)

    # ── QA evidence ingestors ─────────────────────────────────────────────────

    def _qa_A1(self, ev: dict, doi: str, pl: str, task: str) -> None:
        seq = ev.get("correct_sequence", [])
        ids = [self._get_or_create_act(pl, lbl, doi, task) for lbl in seq]
        self._link_sequence(ids)
        # Attach precursors to first activity, products to last
        precs = ev.get("precursors", [])
        prods = ev.get("products", [])
        if ids and precs:
            for p in precs:
                mid = self._get_or_create_mat(pl, p, "material")
                if mid not in self.act_nodes[ids[0]].in_mat_ids:
                    self.act_nodes[ids[0]].in_mat_ids.append(mid)
        if ids and prods:
            for p in prods:
                mid = self._get_or_create_mat(pl, p, "material")
                if mid not in self.act_nodes[ids[-1]].out_mat_ids:
                    self.act_nodes[ids[-1]].out_mat_ids.append(mid)

    def _qa_A3(self, ev: dict, doi: str, pl: str, task: str) -> None:
        seq = ev.get("prefix_activities", []) + (
            [ev["next_activity"]] if ev.get("next_activity") else []
        )
        self._link_sequence(
            [self._get_or_create_act(pl, lbl, doi, task) for lbl in seq]
        )

    def _qa_B(self, ev: dict, doi: str, pl: str, task: str) -> None:
        lbl = ev.get("activity_label", "")
        if not lbl:
            return
        conds = {k: v for k, v in
                 (ev.get("all_conditions") or ev.get("conditions") or {}).items()
                 if v}
        self._get_or_create_act(pl, lbl, doi, task, conds)

    def _qa_A2(self, ev: dict, doi: str, pl: str, task: str) -> None:
        lbl = ev.get("missing_activity", "")
        if not lbl:
            return
        # Store form transition as condition metadata
        conds: dict[str, str] = {}
        if ev.get("input_form"):
            conds["input_form"] = ev["input_form"]
        if ev.get("output_form"):
            conds["output_form"] = ev["output_form"]
        self._get_or_create_act(pl, lbl, doi, task, conds)

    def _qa_C1(self, ev: dict, doi: str, pl: str, task: str) -> None:
        lbl = ev.get("activity_label", "")
        tools = ev.get("tools", [])
        if not lbl:
            return
        act_nid = self._get_or_create_act(pl, lbl, doi, task)
        for tool_lbl in tools:
            tid = self._get_or_create_mat(pl, tool_lbl, "tool")
            if tid not in self.act_nodes[act_nid].tool_ids:
                self.act_nodes[act_nid].tool_ids.append(tid)

    def _qa_D1(self, ev: dict, doi: str, pl: str, task: str) -> None:
        seq = ev.get("correct_sequence", [])
        ids = [self._get_or_create_act(pl, lbl, doi, task) for lbl in seq]
        self._link_sequence(ids)
        for edge in ev.get("provenance_edges", []):
            if len(edge) == 3:
                pred_lbl, mat_lbl, cons_lbl = edge
                pred_nid = self._proc_act_idx.get((pl, pred_lbl))
                cons_nid = self._proc_act_idx.get((pl, cons_lbl))
                mid = self._get_or_create_mat(pl, mat_lbl, "material")
                if pred_nid and mid not in self.act_nodes[pred_nid].out_mat_ids:
                    self.act_nodes[pred_nid].out_mat_ids.append(mid)
                if cons_nid and mid not in self.act_nodes[cons_nid].in_mat_ids:
                    self.act_nodes[cons_nid].in_mat_ids.append(mid)

    # ── neighbourhood extraction ──────────────────────────────────────────────

    def get_neighbourhood(self, act_nid: str, hops: int = 1) -> dict:
        """
        Return a rich neighbourhood dict for an activity node:
          - sequence context (predecessor / successor activity labels)
          - input / output material details (label, form, purity, size)
          - tools used
          - all condition attributes
          - process and DOI metadata
        """
        node = self.act_nodes[act_nid]

        def _act_lbl(nid: str) -> str:
            return self.act_nodes[nid].label if nid in self.act_nodes else nid

        def _mat_dict(mid: str) -> dict:
            mn = self.mat_nodes.get(mid)
            if not mn:
                return {"label": mid}
            d: dict[str, str] = {"label": mn.label}
            for attr in ("form", "purity", "mass", "concentration",
                         "length_diameter", "length_thickness", "length"):
                v = getattr(mn, attr, "")
                if v:
                    d[attr] = v
            return d

        preds = [_act_lbl(p) for p in node.pred_activities]
        succs = [_act_lbl(s) for s in node.succ_activities]

        if hops >= 2:
            extended_preds: list[str] = []
            for p in node.pred_activities:
                extended_preds += [_act_lbl(pp) for pp in self.act_nodes[p].pred_activities]
            preds = extended_preds + preds

            extended_succs: list[str] = []
            for s in node.succ_activities:
                extended_succs += [_act_lbl(ss) for ss in self.act_nodes[s].succ_activities]
            succs = succs + extended_succs

        return {
            "process_label":          node.process_label,
            "doi":                    node.doi,
            "activity":               node.label,
            "conditions":             node.conditions,
            "in_materials":           [_mat_dict(mid) for mid in node.in_mat_ids],
            "out_materials":          [_mat_dict(mid) for mid in node.out_mat_ids],
            "tools":                  [self.mat_nodes[tid].label
                                       for tid in node.tool_ids if tid in self.mat_nodes],
            "predecessor_activities": preds,
            "successor_activities":   succs,
        }

    # ── statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        n_with_conds  = sum(1 for n in self.act_nodes.values() if n.conditions)
        n_with_tools  = sum(1 for n in self.act_nodes.values() if n.tool_ids)
        n_with_preds  = sum(1 for n in self.act_nodes.values() if n.pred_activities)
        n_with_succs  = sum(1 for n in self.act_nodes.values() if n.succ_activities)
        n_mat_w_form  = sum(1 for m in self.mat_nodes.values() if m.form)
        n_mat_w_purity= sum(1 for m in self.mat_nodes.values() if m.purity)
        return {
            "activity_nodes": len(self.act_nodes),
            "material_nodes": len(self.mat_nodes),
            "processes":      len(self._process_act_nodes),
            "with_conditions":  n_with_conds,
            "with_tools":       n_with_tools,
            "with_predecessors":n_with_preds,
            "with_successors":  n_with_succs,
            "materials_with_form":  n_mat_w_form,
            "materials_with_purity":n_mat_w_purity,
        }


# ── FAISS retriever ───────────────────────────────────────────────────────────

class GraphRetriever:
    """
    FAISS index over ActivityNode embeddings.

    Each node is embedded using a rich text that includes:
      activity label, all conditions, input material summaries (with form/purity),
      output material summaries, tool names, predecessor/successor activity labels.
    """

    def __init__(
        self,
        graph: ProcessKnowledgeGraph,
        embedder_name: str = "all-mpnet-base-v2",
        cache_path: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.graph = graph
        self.embedder = SentenceTransformer(embedder_name)
        self.node_ids: list[str] = list(graph.act_nodes.keys())
        self._build_index(cache_path, batch_size)

    def _node_text(self, act_nid: str) -> str:
        """Rich text representation of an ActivityNode for embedding."""
        n = self.graph.act_nodes[act_nid]
        # Include process_label so embeddings are aware of the material system context
        parts = [f"Activity: {n.label}", f"Process: {n.process_label}"]

        # Input materials with physical properties
        in_mats = [self.graph.mat_nodes[mid].summary()
                   for mid in n.in_mat_ids if mid in self.graph.mat_nodes
                   and self.graph.mat_nodes[mid].entity_type == "material"]
        if in_mats:
            parts.append(f"Input materials: {'; '.join(in_mats)}")

        # Output materials
        out_mats = [self.graph.mat_nodes[mid].summary()
                    for mid in n.out_mat_ids if mid in self.graph.mat_nodes]
        if out_mats:
            parts.append(f"Output materials: {'; '.join(out_mats)}")

        # Tools
        tools = [self.graph.mat_nodes[tid].label
                 for tid in n.tool_ids if tid in self.graph.mat_nodes]
        if tools:
            parts.append(f"Tools: {', '.join(tools)}")

        # All conditions
        if n.conditions:
            cond_str = "; ".join(f"{k}: {v}" for k, v in n.conditions.items() if v)
            parts.append(f"Conditions: {cond_str}")

        # Sequence context
        preds = [self.graph.act_nodes[p].label
                 for p in n.pred_activities if p in self.graph.act_nodes]
        succs = [self.graph.act_nodes[s].label
                 for s in n.succ_activities if s in self.graph.act_nodes]
        if preds:
            parts.append(f"After: {', '.join(preds)}")
        if succs:
            parts.append(f"Before: {', '.join(succs)}")

        return ". ".join(parts)

    def _build_index(self, cache_path: str | None, batch_size: int) -> None:
        texts = [self._node_text(nid) for nid in self.node_ids]

        if cache_path and Path(cache_path).exists():
            print(f"[GraphRetriever] Loading embeddings from cache: {cache_path}")
            with open(cache_path, "rb") as fh:
                embs = pickle.load(fh)
        else:
            print(f"[GraphRetriever] Embedding {len(texts)} activity nodes...")
            embs = self.embedder.encode(
                texts,
                show_progress_bar=True,
                batch_size=batch_size,
                convert_to_numpy=True,
            ).astype(np.float32)
            faiss.normalize_L2(embs)
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as fh:
                    pickle.dump(embs, fh)
                print(f"[GraphRetriever] Cached to: {cache_path}")

        self._embs = embs
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def search(
        self,
        query: str,
        k: int,
        exclude_doi: str | None = None,
        hops: int = 1,
    ) -> list[dict]:
        """Return top-k neighbourhood dicts; exclude nodes from the test DOI.

        Parameters
        ----------
        hops : neighbourhood depth passed to get_neighbourhood (1 or 2).
               Use 2 for richer sequence context at the cost of larger prompts.
        """
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        # Wider buffer: absorb DOI exclusions + one-per-process dedup across a
        # potentially large graph.
        n_fetch = min(max(k * 8, k + 40), len(self.node_ids))
        _, idxs = self.index.search(q_emb, n_fetch)

        results: list[dict] = []
        seen_processes: set[str] = set()

        for idx in idxs[0]:
            if idx < 0:
                continue
            nid = self.node_ids[idx]
            node = self.graph.act_nodes[nid]
            if exclude_doi and node.doi == exclude_doi:
                continue
            if node.process_label in seen_processes:
                continue
            seen_processes.add(node.process_label)
            results.append(self.graph.get_neighbourhood(nid, hops=hops))
            if len(results) >= k:
                break

        return results


# ── Context formatter ─────────────────────────────────────────────────────────

def _fmt_material(m: dict) -> str:
    """Compact material representation: label (form, purity, size)."""
    lbl = m.get("label", "")
    extras = []
    for k in ("form", "purity", "concentration", "mass", "length_diameter", "length_thickness"):
        v = m.get(k, "")
        if v:
            extras.append(f"{k}={v}")
    return f"{lbl} ({', '.join(extras)})" if extras else lbl


def format_graph_context(neighbourhoods: list[dict], max_chars: int = 4000) -> str:
    """
    Format retrieved graph neighbourhoods as structured synthesis context.

    Each block shows:
      Sequence  : pred1 → pred2 → [ACTIVITY] → succ1 → succ2
      Inputs    : material label (form, purity, size, …)
      Outputs   : material label (form, …)
      Tools     : tool names
      Conditions: temperature; duration; atmosphere; pressure; …
    """
    parts: list[str] = []
    total = 0

    for i, nb in enumerate(neighbourhoods, 1):
        act = nb["activity"]
        pl = nb["process_label"]
        lines = [f"[Process {i}]  {pl}"]

        # Sequence
        preds = nb.get("predecessor_activities", [])
        succs = nb.get("successor_activities", [])
        seq_parts = (preds[-2:] if preds else []) + [f"[{act}]"] + (succs[:2] if succs else [])
        lines.append(f"  Sequence  : {' → '.join(seq_parts)}")

        # Input materials (with physical attributes)
        in_mats = nb.get("in_materials", [])
        if in_mats:
            lines.append(f"  Inputs    : {'; '.join(_fmt_material(m) for m in in_mats[:4])}")

        # Output materials
        out_mats = nb.get("out_materials", [])
        if out_mats:
            lines.append(f"  Outputs   : {'; '.join(_fmt_material(m) for m in out_mats[:2])}")

        # Tools
        tools = nb.get("tools", [])
        if tools:
            lines.append(f"  Tools     : {', '.join(tools[:4])}")

        # Conditions (full set)
        conds = nb.get("conditions", {})
        if conds:
            cond_str = "; ".join(f"{k}: {v}" for k, v in conds.items() if v)
            lines.append(f"  Conditions: {cond_str}")

        block = "\n".join(lines)
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


# ── Message builder ───────────────────────────────────────────────────────────

def build_messages_graph_rag(rec: dict, neighbourhoods: list[dict]) -> list[dict]:
    context = format_graph_context(neighbourhoods)
    user_content = (
        f"Synthesis process knowledge retrieved from graph:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"{format_question_block(rec)}"
        f"{_ANSWER_PROMPT}"
    )
    return [
        {"role": "system", "content": _SYS_GRAPH_RAG},
        {"role": "user",   "content": user_content},
    ]


# ── Output record ─────────────────────────────────────────────────────────────

def make_output_record(item: dict, model_answer: str) -> dict:
    choices_dict = item.get("choices", {})
    gold_letter = item.get("answer", "")
    return {
        "qid":          item.get("qid", ""),
        "task":         item.get("task", ""),
        "method":       "graph_rag",
        "question":     item["question"],
        "model_answer": model_answer,
        "gt_answer":    gold_letter,
        "choices":      choices_to_list(choices_dict),
        "answer_index": letter_to_index(gold_letter),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MatProcBench GraphRAG baseline")
    parser.add_argument("--model",      type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_file",  type=str, required=True,
                        help="Test split JSONL.")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Train split JSONL (used for DOI whitelist + QA supplement).")
    parser.add_argument("--raw_file",   type=str, default=None,
                        help="MatPROV.jsonl raw provenance file (recommended for richer graph).")
    parser.add_argument("--save_file",  type=str, required=True)
    parser.add_argument("--graph_k",    type=int, default=3)
    parser.add_argument("--graph_hops", type=int, default=1,
                        help="Neighbourhood depth for graph retrieval (1 or 2). "
                             "2 gives richer sequence context.")
    parser.add_argument("--embedder",   type=str, default="all-mpnet-base-v2")
    parser.add_argument("--emb_cache",  type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit quantization (requires bitsandbytes). Recommended for 32B+ models.")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[graph_rag] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    # ── Load QA data ──────────────────────────────────────────────────────────
    print(f"[graph_rag] Loading test  : {args.data_file}")
    test_records  = load_jsonl(args.data_file)
    print(f"[graph_rag] Loading train : {args.train_file}")
    train_records = load_jsonl(args.train_file)
    print(f"[graph_rag] Test: {len(test_records)}, Train: {len(train_records)}")

    # Collect train DOIs for filtering raw data
    train_dois: set[str] = {
        r.get("evidence", {}).get("doi", "")
        for r in train_records if r.get("evidence", {}).get("doi")
    }

    # ── Build knowledge graph ─────────────────────────────────────────────────
    if args.raw_file and Path(args.raw_file).exists():
        print(f"[graph_rag] Building graph from raw MatPROV: {args.raw_file}")
        raw_records = load_jsonl(args.raw_file)
        graph = ProcessKnowledgeGraph.build_from_raw(raw_records, train_dois)
        print(f"[graph_rag] Supplementing from QA evidence ...")
        graph.ingest_qa_records(train_records)
    else:
        print("[graph_rag] No --raw_file; building graph from QA evidence only.")
        graph = ProcessKnowledgeGraph()
        graph.ingest_qa_records(train_records)

    st = graph.stats()
    print(f"[graph_rag] Graph stats: {st}")

    # ── Build FAISS index ─────────────────────────────────────────────────────
    retriever = GraphRetriever(
        graph=graph,
        embedder_name=args.embedder,
        cache_path=args.emb_cache,
    )

    # ── Inference loop ────────────────────────────────────────────────────────
    results: list[dict] = []
    for item in tqdm(test_records, desc="[graph_rag]"):
        doi = item.get("evidence", {}).get("doi", None)
        neighbourhoods = retriever.search(
            query=item["question"],
            k=args.graph_k,
            exclude_doi=doi,
            hops=args.graph_hops,
        )
        messages = build_messages_graph_rag(item, neighbourhoods)
        raw = run_inference(pipe, messages, args.max_new_tokens)
        results.append(make_output_record(item, raw))

    # ── Save ──────────────────────────────────────────────────────────────────
    write_jsonl(args.save_file, results)
    print(f"[graph_rag] Saved {len(results)} records → {args.save_file}")
    if results:
        print(f"[graph_rag] Sample: {results[0]}")


if __name__ == "__main__":
    main()
