"""
retriever.py – FAISS-based retrievers for MatProcBench RAG baseline.

Two retriever classes:

  MatProcRetriever   – legacy: embeds training *question* texts and retrieves
                       the most similar QA records (evidence fields are used as
                       context).  Task-filtered search (search_by_task) builds a
                       temporary sub-index restricted to the target task type.

  MatProcDocRetriever – standard document-level RAG: builds one text document
                        per synthesis process from MatPROV.jsonl raw provenance
                        records, embeds those documents, and retrieves the most
                        relevant process descriptions at query time.  This is the
                        conventional RAG setup (index knowledge, not questions).
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ── JSON-LD value extractor ───────────────────────────────────────────────────

def _val(field: Any) -> str:
    """Extract a plain string from a JSON-LD value field."""
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


# ── MatPROV attribute keys (subset needed for document text) ─────────────────

_DOC_COND_KEYS: dict[str, str] = {
    "matprov:temperature":      "temperature",
    "matprov:temperature_start":"temperature_start",
    "matprov:temperature_end":  "temperature_end",
    "matprov:temperature_rate": "temperature_rate",
    "matprov:duration":         "duration",
    "matprov:pressure":         "pressure",
    "matprov:atmosphere":       "atmosphere",
    "matprov:atmosphere_start": "atmosphere_start",
    "matprov:rotation":         "rotation",
    "matprov:rotation_rate":    "rotation_rate",
    "matprov:concentration":    "concentration",
    "matprov:mass":             "mass",
    "matprov:heating_rate":     "heating_rate",
    "matprov:form":             "form",
}

_DOC_ENT_KEYS: dict[str, str] = {
    "matprov:form":           "form",
    "matprov:purity":         "purity",
    "matprov:mass":           "mass",
    "matprov:concentration":  "concentration",
    "matprov:length_diameter":"length_diameter",
    "matprov:length_thickness":"length_thickness",
}


# ── Process-to-text converter ─────────────────────────────────────────────────

def _raw_process_to_text(rec: dict) -> str:
    """
    Convert one MatPROV provenance record into a single searchable text document.

    Format:
      Process: <label>. Materials: <mat (form, purity)>; ... Equipment: <tool>; ...
      <activity> (inputs: <mat>, ...) -> <output> [temperature=..., duration=...]
      ...
    """
    label = rec.get("label", "")
    graph = rec.get("prov_jsonld", {}).get("@graph", [])

    entities: dict[str, dict] = {}     # raw_id → attrs
    activities: dict[str, dict] = {}   # raw_id → {label, conditions}
    act_inputs: dict[str, list[str]] = defaultdict(list)
    act_outputs: dict[str, list[str]] = defaultdict(list)

    # Pass 1: collect nodes
    for item in graph:
        if not isinstance(item, dict):
            continue
        t = item.get("@type", "")
        raw_id = item.get("@id", "")

        if t == "Entity":
            lbl = _val(item.get("label", ""))
            if not lbl:
                continue
            attrs: dict[str, str] = {
                "label": lbl,
                "etype": _val(item.get("type", "material")),
            }
            for mk, sk in _DOC_ENT_KEYS.items():
                v = _val(item.get(mk, ""))
                if v:
                    attrs[sk] = v
            entities[raw_id] = attrs

        elif t == "Activity":
            lbl = _val(item.get("label", ""))
            if not lbl:
                continue
            conds: dict[str, str] = {}
            for mk, sk in _DOC_COND_KEYS.items():
                v = _val(item.get(mk, ""))
                if v:
                    conds[sk] = v
            activities[raw_id] = {"label": lbl, "conditions": conds}

    # Pass 2: wire edges (relations may appear before or after nodes)
    for item in graph:
        if not isinstance(item, dict):
            continue
        t = item.get("@type", "")
        aid = item.get("activity", "")
        eid = item.get("entity", "")
        if t == "Usage" and aid in activities and eid in entities:
            if eid not in act_inputs[aid]:
                act_inputs[aid].append(eid)
        elif t == "Generation" and aid in activities and eid in entities:
            if eid not in act_outputs[aid]:
                act_outputs[aid].append(eid)

    # Build text
    parts = [f"Process: {label}"]

    mat_strs, tool_strs = [], []
    for e in entities.values():
        extras = [f"{k}={e[k]}" for k in ("form", "purity", "mass", "concentration")
                  if e.get(k)]
        desc = e["label"] + (f" ({', '.join(extras)})" if extras else "")
        (tool_strs if e.get("etype") == "tool" else mat_strs).append(desc)
    if mat_strs:
        parts.append(f"Materials: {'; '.join(mat_strs)}")
    if tool_strs:
        parts.append(f"Equipment: {'; '.join(tool_strs)}")

    for aid, act in activities.items():
        ins  = [entities[e]["label"] for e in act_inputs.get(aid, [])  if e in entities]
        outs = [entities[e]["label"] for e in act_outputs.get(aid, []) if e in entities]
        step = act["label"]
        if ins:
            step += f" (inputs: {', '.join(ins)})"
        if outs:
            step += f" -> {', '.join(outs)}"
        if act["conditions"]:
            step += " [" + ", ".join(f"{k}={v}" for k, v in act["conditions"].items()) + "]"
        parts.append(step)

    return ". ".join(parts)


class MatProcRetriever:
    """
    Retriever over a list of MatProcBench records.

    Parameters
    ----------
    records : list[dict]
        Training split records (each must have a "question" field and
        optionally "task" and "qid").
    embedder_name : str
        Sentence-transformer model identifier (HuggingFace hub or local path).
    cache_path : str | None
        If given, embeddings are loaded from / saved to this pickle file.
    batch_size : int
        Batch size for sentence-transformer encoding.
    """

    def __init__(
        self,
        records: list[dict],
        embedder_name: str = "all-mpnet-base-v2",
        cache_path: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        self.records = records
        self.embedder = SentenceTransformer(embedder_name)
        self.batch_size = batch_size

        # Build task → record-index mapping for task-filtered search
        self.task_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            self.task_to_indices[r.get("task", "unknown")].append(i)

        self._build_index(cache_path)

    # ── Index construction ────────────────────────────────────────────────────

    def _build_index(self, cache_path: Optional[str]) -> None:
        texts = [r["question"] for r in self.records]

        if cache_path and Path(cache_path).exists():
            print(f"[Retriever] Loading embeddings from cache: {cache_path}")
            with open(cache_path, "rb") as fh:
                embs = pickle.load(fh)
        else:
            print(f"[Retriever] Encoding {len(texts)} questions with {self.embedder}...")
            embs = self.embedder.encode(
                texts,
                show_progress_bar=True,
                batch_size=self.batch_size,
                convert_to_numpy=True,
            ).astype(np.float32)
            faiss.normalize_L2(embs)  # unit-normalise for cosine similarity
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as fh:
                    pickle.dump(embs, fh)
                print(f"[Retriever] Embeddings cached to: {cache_path}")

        self._embs = embs
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int,
        exclude_qid: Optional[str] = None,
        dedup_doi: bool = True,
    ) -> list[dict]:
        """Return the top-k most similar records to *query*.

        Parameters
        ----------
        query       : question string to embed and search.
        k           : desired number of results.
        exclude_qid : if given, skip the record whose qid matches this value
                      (avoids returning the test question itself when the test
                      set overlaps with the index).
        dedup_doi   : if True (default), return at most one record per source DOI
                      so that retrieved context covers diverse synthesis processes.
        """
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        # Over-fetch to absorb exclusions and DOI dedup
        n_fetch = min(k * 4 + 10, len(self.records))
        _, idxs = self.index.search(q_emb, n_fetch)

        out: list[dict] = []
        seen_dois: set[str] = set()
        for idx in idxs[0]:
            if idx < 0:
                continue
            r = self.records[idx]
            if exclude_qid and r.get("qid") == exclude_qid:
                continue
            if dedup_doi:
                doi = r.get("evidence", {}).get("doi", "")
                if doi and doi in seen_dois:
                    continue
                if doi:
                    seen_dois.add(doi)
            out.append(r)
            if len(out) >= k:
                break
        return out

    def search_by_task(
        self,
        query: str,
        task: str,
        k: int,
        exclude_qid: Optional[str] = None,
        dedup_doi: bool = True,
    ) -> list[dict]:
        """Return top-k similar records restricted to the same *task* type.

        Falls back to corpus-wide search when the task pool is empty or too
        small to satisfy k.  Applies the same DOI-deduplication as search().
        """
        pool = self.task_to_indices.get(task, [])
        if not pool:
            return self.search(query, k, exclude_qid, dedup_doi)

        if exclude_qid:
            pool = [i for i in pool if self.records[i].get("qid") != exclude_qid]

        if len(pool) <= k:
            return [self.records[i] for i in pool][:k]

        # Build a temporary sub-index for this task type
        sub_embs = self._embs[pool].copy()
        sub_index = faiss.IndexFlatIP(sub_embs.shape[1])
        sub_index.add(sub_embs)

        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        n_fetch = min(k * 4 + 10, len(pool))
        _, sub_results = sub_index.search(q_emb, n_fetch)

        out: list[dict] = []
        seen_dois: set[str] = set()
        for sub_i in sub_results[0]:
            if sub_i < 0:
                continue
            r = self.records[pool[sub_i]]
            if dedup_doi:
                doi = r.get("evidence", {}).get("doi", "")
                if doi and doi in seen_dois:
                    continue
                if doi:
                    seen_dois.add(doi)
            out.append(r)
            if len(out) >= k:
                break
        return out


# ── Document-level RAG retriever (standard RAG) ───────────────────────────────

class MatProcDocRetriever:
    """
    Standard document-level RAG retriever for MatProcBench.

    Indexes one text document per synthesis process, built from MatPROV.jsonl
    raw provenance records.  At query time the test question is embedded and
    matched against process documents; the top-k process descriptions are
    returned as context.

    This follows the conventional RAG paradigm — index *knowledge*, not
    *questions* — and is preferable to MatProcRetriever when the raw MatPROV
    file is available.
    """

    def __init__(
        self,
        docs: list[dict],
        embedder_name: str = "all-mpnet-base-v2",
        cache_path: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        self.docs = docs   # list of {text, doi, process_label}
        self.embedder = SentenceTransformer(embedder_name)
        self._build_index(cache_path, batch_size)

    @classmethod
    def from_raw_matprov(
        cls,
        raw_records: list[dict],
        train_dois: Optional[set[str]] = None,
        embedder_name: str = "all-mpnet-base-v2",
        cache_path: Optional[str] = None,
        batch_size: int = 64,
    ) -> "MatProcDocRetriever":
        """Build retriever from MatPROV raw records (one document per process).

        Parameters
        ----------
        raw_records : list of records from MatPROV.jsonl.
        train_dois  : if given, only processes from these DOIs are indexed
                      (avoids leaking test-set knowledge).
        """
        docs: list[dict] = []
        for rec in raw_records:
            doi = rec.get("doi", "")
            if train_dois and doi not in train_dois:
                continue
            pl = rec.get("label", "")
            if not pl:
                continue
            text = _raw_process_to_text(rec)
            docs.append({"text": text, "doi": doi, "process_label": pl})
        print(f"[DocRetriever] Built {len(docs)} process documents from MatPROV.")
        return cls(docs, embedder_name, cache_path, batch_size)

    def _build_index(self, cache_path: Optional[str], batch_size: int) -> None:
        texts = [d["text"] for d in self.docs]

        if cache_path and Path(cache_path).exists():
            print(f"[DocRetriever] Loading embeddings from cache: {cache_path}")
            with open(cache_path, "rb") as fh:
                embs = pickle.load(fh)
        else:
            print(f"[DocRetriever] Encoding {len(texts)} process documents...")
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
                print(f"[DocRetriever] Cached to: {cache_path}")

        self._embs = embs
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def search(
        self,
        query: str,
        k: int,
        exclude_doi: Optional[str] = None,
    ) -> list[dict]:
        """Return top-k process documents; exclude processes from *exclude_doi*."""
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        n_fetch = min(k + (20 if exclude_doi else 0), len(self.docs))
        _, idxs = self.index.search(q_emb, n_fetch)

        out: list[dict] = []
        for idx in idxs[0]:
            if idx < 0:
                continue
            doc = self.docs[idx]
            if exclude_doi and doc["doi"] == exclude_doi:
                continue
            out.append(doc)
            if len(out) >= k:
                break
        return out


# ── Context formatter for MatProcDocRetriever results ─────────────────────────

def format_doc_context(docs: list[dict], max_chars: int = 3000) -> str:
    """Format retrieved process documents as structured synthesis context."""
    parts: list[str] = []
    total = 0
    for i, doc in enumerate(docs, 1):
        pl   = doc.get("process_label", "")
        text = doc.get("text", "")
        block = f"[Retrieved Process {i}]  {pl}\n{text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
