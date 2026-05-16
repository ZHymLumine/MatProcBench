"""
build_process_tasks.py  (v2 – MCQ edition)
==========================================
Builds MatProcBench multiple-choice QA dataset from MatPROV.jsonl.

Every task is a 4-choice MCQ (choices A-D, exactly one correct).

Tasks
-----
A1  Route Retrieval          – given precursors + product, choose correct activity sequence
A2  Missing Step             – given partial route + entity-form context, identify missing step
A3  Next Activity            – given k≥2 prefix, predict the next step

B1  Condition Prediction     – given activity + partial conditions, predict one masked condition
B2  Full Condition Set       – choose the correct (temperature, duration, atmosphere) triple

C1  Tool Selection           – given activity + conditions, predict the required tool

D1  Process Ordering         – choose the physically valid ordering given an explicit material-flow edge

Usage
-----
python data_utils/build_process_tasks.py \\
    --input  /path/to/MatPROV.jsonl \\
    --output /path/to/output_dir \\
    --seed   42
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ── helpers ────────────────────────────────────────────────────────────────────

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


def _make_choices(correct: str, distractors: list[str], rng: random.Random) -> tuple[dict[str, str], str]:
    """Shuffle correct + distractors into 4 labeled choices; return (choices, correct_letter)."""
    opts = [correct] + [d for d in distractors if d != correct][:3]
    while len(opts) < 4:
        opts.append(f"{opts[-1]}*")
    rng.shuffle(opts)
    letters = ["A", "B", "C", "D"]
    choices = {letters[i]: opts[i] for i in range(4)}
    answer = next(l for l, v in choices.items() if v == correct)
    return choices, answer


def _sample_unique(pool: list, n: int, rng: random.Random) -> list:
    """Return up to n unique non-empty items sampled from pool."""
    unique = list(dict.fromkeys(str(x) for x in pool if x))
    rng.shuffle(unique)
    return unique[:n]


def _parse_temp_K(s: str) -> float | None:
    """Parse a temperature string to Kelvin float, or None if unparseable."""
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    val = float(m.group(1))
    s_lower = s.lower()
    if "k" in s_lower:
        return val
    return val + 273.15 if val > 100 else val + 273.15


# Atmosphere canonical forms – used to block synonym distractors in B1
_ATMOSPHERE_CANONICAL: dict[str, str] = {
    "ar": "Ar", "argon": "Ar", "argon gas": "Ar", "argon atmosphere": "Ar",
    "flowing argon": "Ar", "argon flow": "Ar", "under argon": "Ar",
    "n2": "N2", "nitrogen": "N2", "nitrogen gas": "N2", "n₂": "N2",
    "flowing nitrogen": "N2", "nitrogen flow": "N2", "under nitrogen": "N2",
    "air": "air", "air atmosphere": "air", "ambient air": "air",
    "static air": "air", "in air": "air", "flowing air": "air",
    "vacuum": "vacuum", "under vacuum": "vacuum", "high vacuum": "vacuum",
    "h2": "H2", "hydrogen": "H2", "hydrogen gas": "H2", "h₂": "H2",
    "o2": "O2", "oxygen": "O2", "oxygen gas": "O2", "o₂": "O2",
    "co2": "CO2", "carbon dioxide": "CO2",
    "nh3": "NH3", "ammonia": "NH3",
    "he": "He", "helium": "He",
}


def _norm_text(s: str) -> str:
    """Normalize a free-text value for duplicate-option checks."""
    return re.sub(r"\s+", " ", s.lower().replace("℃", "°c").strip())


def _canonical_atm(s: str) -> str:
    """Return a coarse atmosphere identifier for synonym-aware distractors."""
    raw = _norm_text(s)
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    if raw in _ATMOSPHERE_CANONICAL:
        return _ATMOSPHERE_CANONICAL[raw]
    if "vacuum" in raw or "torr" in raw:
        return "vacuum"
    if re.search(r"\bpa\b", raw) and not re.search(r"\b(ar|argon|n2|nitrogen|h2|hydrogen|air)\b", raw):
        return "vacuum"
    if ("argon" in raw or re.search(r"\bar\b", raw)) and not re.search(
        r"\b(h2|hydrogen|n2|nitrogen|o2|oxygen)\b", raw
    ):
        return "Ar"
    if "nitrogen" in raw or re.search(r"\bn2\b", raw):
        return "N2"
    if "hydrogen" in raw or re.search(r"\bh2\b", raw):
        return "H2"
    if "oxygen" in raw or re.search(r"\bo2\b", raw):
        return "O2"
    if "air" in raw or "ambient" in raw:
        return "air"
    return _ATMOSPHERE_CANONICAL.get(compact, raw)


def _condition_value_key(cond_key: str, value: str) -> tuple[str, Any]:
    """Canonical key used to block duplicate or near-synonymous choices."""
    if cond_key == "atmosphere":
        return (cond_key, _canonical_atm(value))
    if cond_key == "temperature":
        numbers = re.findall(r"\d+(?:\.\d+)?", value)
        parsed = _parse_temp_K(value)
        if parsed is not None and len(numbers) == 1:
            return (cond_key, round(parsed, 1))
    return (cond_key, _norm_text(value))


def _filter_distinct_distractors(
    correct: str,
    candidates: list[str],
    cond_key: str,
    n: int,
) -> list[str]:
    """Keep distractors that are distinct from the answer under coarse normalization."""
    seen = {_condition_value_key(cond_key, correct), ("text", _norm_text(correct))}
    out: list[str] = []
    for cand in candidates:
        keys = {_condition_value_key(cond_key, cand), ("text", _norm_text(cand))}
        if keys & seen:
            continue
        seen.update(keys)
        out.append(cand)
        if len(out) >= n:
            break
    return out


_ACTIVITY_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("thermal", ("anneal", "heat", "calc", "sinter", "firing", "pyro", "melt", "quench", "aging", "ageing")),
    ("mechanical", ("mill", "grind", "crush", "press", "compact", "roll", "siev", "polish", "pulver")),
    ("mixing", ("mix", "stir", "blend", "combine", "add", "dispers", "dissolv")),
    ("separation", ("wash", "filter", "centrifug", "decant", "separat")),
    ("drying", ("dry", "evaporat", "lyophiliz", "dehydrat")),
    ("deposition", ("deposit", "sputter", "evaporat", "coat", "spin", "grow", "pld", "cvd")),
)


def _activity_families(label: str) -> set[str]:
    """Coarse operation families for plausible A2 distractors."""
    norm = _norm_text(label)
    return {
        family
        for family, needles in _ACTIVITY_FAMILY_PATTERNS
        if any(needle in norm for needle in needles)
    }


def _parse_duration_h(s: str) -> float | None:
    """Parse a duration string to hours float, or None if unparseable."""
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    val = float(m.group(1))
    s_lower = s.lower()
    if "min" in s_lower:
        return val / 60.0
    if re.search(r"\bsec\b|\bs\b", s_lower) and "h" not in s_lower:
        return val / 3600.0
    return val  # assume hours


def _fmt_conds(c: dict) -> str:
    """Format a condition dict into a readable semicolon-separated string."""
    return "; ".join(f"{k}: {v}" for k, v in sorted(c.items()) if v) or "N/A"


# ── ProcessGraph ──────────────────────────────────────────────────────────────

class ProcessGraph:
    """
    Parses a single MatPROV prov_jsonld graph into structured dicts.
    """

    COND_KEYS = {
        "matprov:temperature": "temperature",
        "matprov:duration": "duration",
        "matprov:atmosphere": "atmosphere",
        "matprov:pressure": "pressure",
        "matprov:temperature_start": "temperature_start",
        "matprov:temperature_end": "temperature_end",
        "matprov:temperature_rate": "temperature_rate",
        "matprov:atmosphere_start": "atmosphere_start",
        "matprov:rotation": "rotation",
        "matprov:concentration": "concentration",
    }

    def __init__(self, record: dict) -> None:
        self.doi = record.get("doi", "")
        self.label = record.get("label", "")
        self.entities: dict[str, dict] = {}
        self.activities: dict[str, dict] = {}
        self.usage: list[tuple[str, str]] = []      # (entity_id, activity_id)
        self.generation: list[tuple[str, str]] = [] # (activity_id, entity_id)
        self._parse(record.get("prov_jsonld", {}).get("@graph", []))
        self._assign_roles()
        self._order_activities()

    def _parse(self, graph: list) -> None:
        for item in graph:
            if not isinstance(item, dict):
                continue
            obj_type = item.get("@type", "")
            local_id = item.get("@id", "")

            if obj_type == "Entity":
                entity_type_raw = _val(item.get("type", "material"))
                entity_type = "tool" if entity_type_raw == "tool" else "material"
                self.entities[local_id] = {
                    "label": _val(item.get("label", "")),
                    "entity_type": entity_type,
                    "purity": _val(item.get("matprov:purity")),
                    "form": _val(item.get("matprov:form")),
                    "mass": _val(item.get("matprov:mass")),
                    "concentration": _val(item.get("matprov:concentration")),
                }

            elif obj_type == "Activity":
                conds: dict[str, str] = {}
                for mk, sk in self.COND_KEYS.items():
                    v = _val(item.get(mk))
                    if v:
                        conds[sk] = v
                self.activities[local_id] = {
                    "label": _val(item.get("label", "")),
                    "conditions": conds,
                    "role": None,
                    "order": None,
                }

            elif obj_type == "Usage":
                eid = item.get("entity", "")
                aid = item.get("activity", "")
                if eid and aid:
                    self.usage.append((eid, aid))

            elif obj_type == "Generation":
                aid = item.get("activity", "")
                eid = item.get("entity", "")
                if aid and eid:
                    self.generation.append((aid, eid))

    def _assign_roles(self) -> None:
        generated_by: dict[str, list[str]] = defaultdict(list)
        used_in: dict[str, list[str]] = defaultdict(list)
        for aid, eid in self.generation:
            generated_by[eid].append(aid)
        for eid, aid in self.usage:
            used_in[eid].append(aid)

        for eid, ent in self.entities.items():
            in_deg = len(generated_by[eid])
            out_deg = len(used_in[eid])
            if in_deg == 0 and out_deg > 0:
                ent["role"] = "precursor"
            elif in_deg > 0 and out_deg > 0:
                ent["role"] = "intermediate"
            elif in_deg > 0 and out_deg == 0:
                ent["role"] = "product"
            else:
                ent["role"] = "isolated"

        self._generated_by = generated_by
        self._used_in = used_in

    def _order_activities(self) -> None:
        gen_map: dict[str, str] = {}
        for aid, eid in self.generation:
            gen_map[eid] = aid

        act_preds: dict[str, list[str]] = defaultdict(list)
        for eid, aid_consumer in self.usage:
            if eid in gen_map:
                aid_producer = gen_map[eid]
                if aid_producer != aid_consumer:
                    act_preds[aid_consumer].append(aid_producer)

        in_degree: dict[str, int] = {a: len(set(p)) for a, p in act_preds.items()}
        for aid in self.activities:
            if aid not in in_degree:
                in_degree[aid] = 0

        from collections import deque
        queue = deque(a for a, d in in_degree.items() if d == 0)
        order = 0
        while queue:
            aid = queue.popleft()
            self.activities[aid]["order"] = order
            order += 1
            for _, eid in [g for g in self.generation if g[0] == aid]:
                for eid2, aid_next in self.usage:
                    if eid2 == eid and aid_next in in_degree:
                        in_degree[aid_next] -= 1
                        if in_degree[aid_next] == 0:
                            queue.append(aid_next)

        for aid in self.activities:
            if self.activities[aid]["order"] is None:
                self.activities[aid]["order"] = order
                order += 1

    def ordered_activities(self) -> list[tuple[str, dict]]:
        return sorted(self.activities.items(), key=lambda x: x[1]["order"] or 0)

    def precursors(self) -> list[tuple[str, dict]]:
        return [(eid, e) for eid, e in self.entities.items()
                if e["role"] == "precursor" and e["entity_type"] == "material"]

    def products(self) -> list[tuple[str, dict]]:
        return [(eid, e) for eid, e in self.entities.items() if e["role"] == "product"]

    def activity_inputs(self, aid: str) -> list[dict]:
        """Return material entity dicts consumed by activity aid."""
        return [self.entities[eid] for eid, a in self.usage
                if a == aid and eid in self.entities
                and self.entities[eid]["entity_type"] == "material"]

    def activity_outputs(self, aid: str) -> list[dict]:
        """Return entity dicts produced by activity aid."""
        return [self.entities[eid] for a, eid in self.generation
                if a == aid and eid in self.entities]

    def activity_tools(self, aid: str) -> list[dict]:
        """Return tool entity dicts consumed by activity aid."""
        return [self.entities[eid] for eid, a in self.usage
                if a == aid and eid in self.entities
                and self.entities[eid]["entity_type"] == "tool"]

    def is_valid(self) -> bool:
        return len(self.activities) >= 1 and len(self.precursors()) >= 1


# ── DistractorPool ─────────────────────────────────────────────────────────────

class DistractorPool:
    """
    Pre-computes distractor value pools from the full corpus.
    Must be built once before task generation begins.
    """

    COND_POOL_KEYS = ["temperature", "duration", "atmosphere", "pressure",
                      "rotation", "temperature_rate"]

    def __init__(self, all_pgs: list[ProcessGraph]) -> None:
        # activity_label (lower) -> condition_key -> list of observed values
        self.cond_pools: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.global_cond: dict[str, list[str]] = defaultdict(list)
        self.tool_pools: dict[str, list[str]] = defaultdict(list)
        self.global_tools: list[str] = []
        self.form_values: list[str] = []
        self.precursor_sets: list[list[str]] = []
        self.all_activity_labels: list[str] = []
        self.cond_records: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.condition_set_records: dict[str, list[dict]] = defaultdict(list)
        # (in_form, out_form) -> [activity_labels] – hard negatives for A2
        self.form_transition_pools: dict[tuple[str, str], list[str]] = defaultdict(list)
        # full ordered sequences from corpus – cross-process distractors for A1
        self.activity_seqs: list[list[str]] = []
        # observed next-step alternatives after an identical activity prefix – hard negatives for A3
        self.prefix_next_pools: dict[tuple[str, ...], list[str]] = defaultdict(list)
        self.prefix_next_records: dict[tuple[str, ...], list[dict]] = defaultdict(list)

        form_set: set[str] = set()
        label_set: set[str] = set()

        for pg in all_pgs:
            for aid, act in pg.activities.items():
                lbl = act["label"].lower()
                if act["label"]:
                    label_set.add(act["label"])
                conds = act["conditions"]
                in_forms = [e["form"].lower() for e in pg.activity_inputs(aid) if e.get("form")]
                out_forms = [e["form"].lower() for e in pg.activity_outputs(aid) if e.get("form")]
                for k in self.COND_POOL_KEYS:
                    v = conds.get(k, "")
                    if v:
                        self.cond_pools[lbl][k].append(v)
                        self.global_cond[k].append(v)
                        self.cond_records[lbl][k].append({
                            "value": v,
                            "input_forms": in_forms,
                            "output_forms": out_forms,
                        })
                if all(conds.get(k, "") for k in ("temperature", "duration", "atmosphere")):
                    self.condition_set_records[lbl].append({
                        "temperature": conds["temperature"],
                        "duration": conds["duration"],
                        "atmosphere": conds["atmosphere"],
                        "input_forms": in_forms,
                        "output_forms": out_forms,
                    })
                for tool_ent in pg.activity_tools(aid):
                    if tool_ent["label"]:
                        self.tool_pools[lbl].append(tool_ent["label"])
                        self.global_tools.append(tool_ent["label"])
                # form-transition pool: what activity performs (in_form → out_form)?
                if in_forms and out_forms and act["label"]:
                    self.form_transition_pools[(in_forms[0], out_forms[0])].append(act["label"])

            for ent in pg.entities.values():
                if ent.get("form"):
                    form_set.add(ent["form"].lower())

            prec = sorted({e["label"] for _, e in pg.precursors() if e["label"]})
            if prec:
                self.precursor_sets.append(prec)

            seq = [a["label"] for _, a in pg.ordered_activities() if a["label"]]
            if len(seq) >= 2:
                self.activity_seqs.append(seq)
                ordered = [(aid, a) for aid, a in pg.ordered_activities() if a["label"]]
                for k in range(1, len(seq)):
                    self.prefix_next_pools[tuple(seq[:k])].append(seq[k])
                    next_aid, next_act = ordered[k]
                    self.prefix_next_records[tuple(seq[:k])].append({
                        "label": next_act["label"],
                        "inputs": _entity_list_brief(pg.activity_inputs(next_aid)),
                        "outputs": _entity_list_brief(pg.activity_outputs(next_aid)),
                    })

        self.form_values = sorted(form_set)
        self.all_activity_labels = sorted(label_set)

    def get_cond_distractors(
        self,
        act_lbl: str,
        cond_key: str,
        correct: str,
        n: int,
        rng: random.Random,
        input_forms: list[str] | None = None,
        output_forms: list[str] | None = None,
    ) -> list[str]:
        def _overlap(rec: dict) -> int:
            ins = set(input_forms or [])
            outs = set(output_forms or [])
            return len(ins & set(rec.get("input_forms", []))) + len(outs & set(rec.get("output_forms", [])))

        records = [
            r for r in self.cond_records.get(act_lbl.lower(), {}).get(cond_key, [])
            if r.get("value") != correct
        ]
        rng.shuffle(records)
        records.sort(key=_overlap, reverse=True)
        pool = [r["value"] for r in records]
        if len(pool) < n:
            pool += [v for v in self.global_cond.get(cond_key, []) if v != correct]
        return _sample_unique(pool, n, rng)

    def get_condition_set_distractors(
        self,
        act_lbl: str,
        correct: dict[str, str],
        n: int,
        rng: random.Random,
        input_forms: list[str] | None = None,
        output_forms: list[str] | None = None,
    ) -> list[dict[str, str]]:
        correct_key = (
            correct.get("temperature", ""),
            correct.get("duration", ""),
            correct.get("atmosphere", ""),
        )

        def _overlap(rec: dict) -> int:
            ins = set(input_forms or [])
            outs = set(output_forms or [])
            return len(ins & set(rec.get("input_forms", []))) + len(outs & set(rec.get("output_forms", [])))

        records = [
            r for r in self.condition_set_records.get(act_lbl.lower(), [])
            if (r["temperature"], r["duration"], r["atmosphere"]) != correct_key
        ]
        rng.shuffle(records)
        records.sort(key=_overlap, reverse=True)
        seen: set[tuple[str, str, str]] = set()
        out: list[dict[str, str]] = []
        for rec in records:
            key = (rec["temperature"], rec["duration"], rec["atmosphere"])
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "temperature": rec["temperature"],
                "duration": rec["duration"],
                "atmosphere": rec["atmosphere"],
            })
            if len(out) >= n:
                break
        return out

    def get_tool_distractors(self, act_lbl: str, correct: str, n: int, rng: random.Random) -> list[str]:
        pool = [v for v in self.tool_pools.get(act_lbl.lower(), []) if v != correct]
        if len(pool) < n:
            pool += [v for v in self.global_tools if v != correct]
        return _sample_unique(pool, n, rng)

    def get_form_distractors(self, correct: str, n: int, rng: random.Random) -> list[str]:
        pool = [f for f in self.form_values if f != correct.lower()]
        rng.shuffle(pool)
        return pool[:n]

    def get_precursor_distractors(self, correct: list[str], n: int, rng: random.Random) -> list[list[str]]:
        correct_key = tuple(correct)
        pool = [s for s in self.precursor_sets if tuple(s) != correct_key]
        rng.shuffle(pool)
        same_size = [s for s in pool if len(s) == len(correct)]
        other = [s for s in pool if len(s) != len(correct)]
        return (same_size + other)[:n]

    def get_seq_distractors(self, correct_seq: list[str], n: int, rng: random.Random) -> list[list[str]]:
        """Generate n distractor sequences by permuting the correct one."""
        distractors: list[list[str]] = []
        seen = {tuple(correct_seq)}

        def _add(s: list[str]) -> None:
            k = tuple(s)
            if k not in seen and s:
                seen.add(k)
                distractors.append(s)

        if len(correct_seq) >= 2:
            _add(list(reversed(correct_seq)))
            _add([correct_seq[-1]] + correct_seq[:-1])
            swapped = list(correct_seq)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            _add(swapped)

        for _ in range(20):
            s = list(correct_seq)
            rng.shuffle(s)
            _add(s)

        return distractors[:n]

    def get_cross_seq_distractors(
        self, correct_seq: list[str], n: int, rng: random.Random
    ) -> list[list[str]]:
        """
        Hard negatives for A1: prefer real sequences from other processes that
        share ≥ half the activity names with the correct sequence (same step vocab,
        different order / context).  Falls back to permutations if needed.
        """
        correct_tup = tuple(correct_seq)
        correct_set = frozenset(correct_seq)
        overlap_threshold = max(1, len(correct_seq) // 2)

        same_length: list[list[str]] = []
        nearby_length: list[list[str]] = []
        seen: set[tuple] = {correct_tup}
        # Prefer high-overlap cross-process routes
        for seq in self.activity_seqs:
            t = tuple(seq)
            if t not in seen and len(frozenset(seq) & correct_set) >= overlap_threshold:
                seen.add(t)
                if len(seq) == len(correct_seq):
                    same_length.append(seq)
                elif abs(len(seq) - len(correct_seq)) <= 1:
                    nearby_length.append(seq)
        rng.shuffle(same_length)
        rng.shuffle(nearby_length)
        distractors = (same_length + nearby_length)[:n]
        # Fill remaining slots with permutations
        if len(distractors) < n:
            distractors += self.get_seq_distractors(correct_seq, n - len(distractors), rng)
        return distractors[:n]

    def get_form_transition_distractors(
        self, in_form: str, out_form: str, correct: str, n: int, rng: random.Random
    ) -> list[str]:
        """
        Hard negatives for A2: activities that perform the same (in_form → out_form)
        transformation as the correct activity, but are different operations.
        E.g., correct='pressing' → distractor='cold isostatic pressing'.
        """
        pool = [lbl for lbl in self.form_transition_pools.get((in_form, out_form), [])
                if lbl.lower() != correct.lower()]
        return _sample_unique(pool, n, rng)

    def get_next_activity_distractors(
        self, prefix: list[str], correct: str, n: int, rng: random.Random
    ) -> list[str]:
        """Return observed alternate next activities for the same route prefix."""
        pool = [
            lbl for lbl in self.prefix_next_pools.get(tuple(prefix), [])
            if lbl.lower() != correct.lower()
        ]
        return _sample_unique(pool, n, rng)

    def get_next_activity_records(
        self, prefix: list[str], correct: str, n: int, rng: random.Random
    ) -> list[dict]:
        pool = [
            rec for rec in self.prefix_next_records.get(tuple(prefix), [])
            if rec.get("label", "").lower() != correct.lower()
        ]
        rng.shuffle(pool)
        seen = {_norm_text(correct)}
        out: list[dict] = []
        for rec in pool:
            key = _norm_text(rec.get("label", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(rec)
            if len(out) >= n:
                break
        return out

    def get_atmosphere_distractors(
        self,
        act_lbl: str,
        correct: str,
        n: int,
        rng: random.Random,
        input_forms: list[str] | None = None,
        output_forms: list[str] | None = None,
    ) -> list[str]:
        """
        Hard negatives for B1 atmosphere: only return values whose canonical form
        differs from the correct value's canonical form, preventing Ar/argon/argon gas
        from appearing as distractors when the answer is one of the same family.
        """
        correct_canon = _canonical_atm(correct)
        def _overlap(rec: dict) -> int:
            ins = set(input_forms or [])
            outs = set(output_forms or [])
            return len(ins & set(rec.get("input_forms", []))) + len(outs & set(rec.get("output_forms", [])))

        records = [
            r for r in self.cond_records.get(act_lbl.lower(), {}).get("atmosphere", [])
            if _canonical_atm(r.get("value", "")) != correct_canon
        ]
        rng.shuffle(records)
        records.sort(key=_overlap, reverse=True)
        pool = [r["value"] for r in records]
        if len(pool) < n:
            pool += [v for v in self.global_cond.get("atmosphere", [])
                     if _canonical_atm(v) != correct_canon]
        return _sample_unique(pool, n, rng)


# ── task builders ──────────────────────────────────────────────────────────────

def _entity_brief(ent: dict) -> str:
    """Compact entity label with form when available."""
    label = ent.get("label", "")
    form = ent.get("form", "")
    return f"{label} ({form})" if label and form else label


def _entity_list_brief(ents: list[dict]) -> list[str]:
    """Stable compact display strings for a list of material entities."""
    return [_entity_brief(e) for e in ents if e.get("label")]


def _route_labels(ordered: list[tuple[str, dict]]) -> list[str]:
    return [a["label"] for _, a in ordered if a["label"]]


def _numbered_route(labels: list[str]) -> str:
    return " → ".join(f"{i + 1}. {lbl}" for i, lbl in enumerate(labels))


def _unique_labels(labels: list[str], correct: str, n: int, rng: random.Random) -> list[str]:
    seen = {_norm_text(correct)}
    out: list[str] = []
    candidates = list(labels)
    rng.shuffle(candidates)
    for lbl in candidates:
        key = _norm_text(lbl)
        if not lbl or key in seen:
            continue
        seen.add(key)
        out.append(lbl)
        if len(out) >= n:
            break
    return out


def _operation_instance_labels(labeled: list[tuple[str, dict]]) -> dict[str, str]:
    """Display each activity occurrence distinctly when labels repeat."""
    totals = Counter(act["label"] for _, act in labeled if act.get("label"))
    seen: dict[str, int] = defaultdict(int)
    out: dict[str, str] = {}
    for aid, act in labeled:
        label = act["label"]
        if totals[label] > 1:
            seen[label] += 1
            out[aid] = f"{label} #{seen[label]}"
        else:
            out[aid] = label
    return out


def _sequence_satisfies_edges(seq: list[str], edges: list[tuple[str, str, str]]) -> bool:
    pos = {step: i for i, step in enumerate(seq)}
    return all(p in pos and c in pos and pos[p] < pos[c] for p, _, c in edges)


def _edge_violations(seq: list[str], edges: list[tuple[str, str, str]]) -> set[int]:
    pos = {step: i for i, step in enumerate(seq)}
    return {
        i for i, (p, _, c) in enumerate(edges)
        if p not in pos or c not in pos or pos[p] >= pos[c]
    }


def _sample_order_distractors(
    correct_seq: list[str],
    edges: list[tuple[str, str, str]],
    rng: random.Random,
    n: int = 3,
) -> tuple[list[list[str]], list[tuple[str, str, str]]]:
    """Return near-miss invalid orderings plus material-flow facts that invalidate them."""
    seen = {tuple(correct_seq)}
    candidates: list[list[str]] = []

    def _add(seq: list[str]) -> None:
        key = tuple(seq)
        if key not in seen and _edge_violations(seq, edges):
            seen.add(key)
            candidates.append(seq)

    if len(correct_seq) >= 2:
        _add(list(reversed(correct_seq)))
        _add([correct_seq[-1]] + correct_seq[:-1])
        _add(correct_seq[1:] + [correct_seq[0]])

    for i in range(len(correct_seq) - 1):
        seq = list(correct_seq)
        seq[i], seq[i + 1] = seq[i + 1], seq[i]
        _add(seq)

    for _ in range(80):
        seq = list(correct_seq)
        rng.shuffle(seq)
        _add(seq)

    rng.shuffle(candidates)
    candidates.sort(key=lambda seq: (len(_edge_violations(seq, edges)), _order_distance(seq, correct_seq)))
    chosen = candidates[:n]
    if len(chosen) < n:
        return [], []

    remaining = set(range(len(chosen)))
    selected_edge_ids: list[int] = []
    violations = [_edge_violations(seq, edges) for seq in chosen]
    while remaining:
        best_edge = max(
            range(len(edges)),
            key=lambda edge_id: sum(edge_id in violations[i] for i in remaining),
        )
        covered = {i for i in remaining if best_edge in violations[i]}
        if not covered:
            return [], []
        selected_edge_ids.append(best_edge)
        remaining -= covered

    selected_edges = [edges[i] for i in selected_edge_ids]
    extra_edges = [edge for i, edge in enumerate(edges) if i not in set(selected_edge_ids)]
    rng.shuffle(extra_edges)
    target_edge_count = min(len(edges), max(2, len(selected_edges) + 2))
    selected_edges.extend(extra_edges[:max(0, target_edge_count - len(selected_edges))])
    if any(_sequence_satisfies_edges(seq, selected_edges) for seq in chosen):
        return [], []
    return chosen, selected_edges


def _order_distance(seq: list[str], correct_seq: list[str]) -> int:
    pos = {step: i for i, step in enumerate(correct_seq)}
    return sum(abs(i - pos.get(step, i)) for i, step in enumerate(seq))


def _activity_io_context(pg: ProcessGraph, aid_to_step: dict[str, str]) -> dict[str, dict]:
    """Map display step names to their consumed and generated material labels."""
    out: dict[str, dict] = {}
    for aid, step_name in aid_to_step.items():
        inputs = _entity_list_brief(pg.activity_inputs(aid))
        outputs = _entity_list_brief(pg.activity_outputs(aid))
        out[step_name] = {"inputs": inputs, "outputs": outputs}
    return out


def _masked_activity_io_context(pg: ProcessGraph, aid_to_step: dict[str, str]) -> dict[str, dict]:
    """Map activity steps to material refs while masking generated sample labels."""
    out: dict[str, dict] = {}
    aliases: dict[str, str] = {}
    generated_eids = {eid for _, eid in pg.generation}

    def _ref(eid: str) -> str:
        ent = pg.entities.get(eid, {})
        form = _masked_form(ent)
        if (
            eid not in generated_eids
            and ent.get("role") == "precursor"
            and ent.get("label")
            and not _is_sample_like_entity(ent)
        ):
            return _entity_brief(ent)
        if eid not in aliases:
            alias = f"M{len(aliases) + 1}"
            aliases[eid] = f"{alias} ({form})" if form else alias
        return aliases[eid]

    for aid, step_name in aid_to_step.items():
        input_refs = [
            _ref(eid) for eid, used_aid in pg.usage
            if used_aid == aid and eid in pg.entities
            and pg.entities[eid]["entity_type"] == "material"
        ]
        output_refs = [
            _ref(eid) for gen_aid, eid in pg.generation
            if gen_aid == aid and eid in pg.entities
            and pg.entities[eid]["entity_type"] == "material"
        ]
        out[step_name] = {"inputs": input_refs, "outputs": output_refs}
    return out


def _render_step_flow(step: str, ctx: dict[str, dict]) -> str:
    data = ctx.get(step, {})
    inputs = data.get("inputs") or ["unknown input"]
    outputs = data.get("outputs") or ["unknown output"]
    return f"{step}: {' + '.join(inputs)} -> {' + '.join(outputs)}"


def _render_flow_option(label: str, inputs: list[str], outputs: list[str]) -> str:
    return f"{label}: {' + '.join(inputs or ['unknown input'])} -> {' + '.join(outputs or ['unknown output'])}"


def _render_flow_sequence(seq: list[str], ctx: dict[str, dict]) -> str:
    return " ; ".join(_render_step_flow(step, ctx) for step in seq)


def _anonymous_material_refs(ents: list[dict], prefix: str) -> list[str]:
    """Render material nodes without operation-derived sample labels."""
    refs: list[str] = []
    for i, ent in enumerate(ents, start=1):
        form = _masked_form(ent)
        refs.append(f"{prefix}{i} ({form})" if form else f"{prefix}{i} (unknown form)")
    return refs


def _masked_form(ent: dict) -> str:
    form = ent.get("form", "")
    return "" if "sample" in _norm_text(form) else form


def _is_sample_like_entity(ent: dict) -> bool:
    return "sample" in _norm_text(ent.get("label", "")) or "sample" in _norm_text(ent.get("form", ""))


def _starting_material_refs(ents: list[dict]) -> list[str]:
    """Keep named chemical precursors but mask sample-like starting nodes."""
    refs: list[str] = []
    sample_idx = 1
    for ent in ents:
        label = ent.get("label", "")
        if label and not _is_sample_like_entity(ent):
            refs.append(_entity_brief(ent))
            continue
        form = _masked_form(ent)
        refs.append(f"S{sample_idx} ({form})" if form else f"S{sample_idx} (unknown form)")
        sample_idx += 1
    return refs


def _render_order_sequence(seq: list[str]) -> str:
    return " -> ".join(seq)


def build_A1(pg: ProcessGraph, pool: DistractorPool, rng: random.Random) -> list[dict]:
    """
    A1 Route Retrieval
    Given precursors + target product, choose the correct ordered activity sequence.
    """
    ordered = [(aid, act) for aid, act in pg.ordered_activities() if act["label"]]
    if len(ordered) < 2:
        return []
    precursor_entities = [e for _, e in pg.precursors()]
    precursors = _starting_material_refs(precursor_entities)
    product_entities = [e for _, e in pg.products()]
    product_refs = _anonymous_material_refs(product_entities, "P")
    correct_seq = [act["label"] for _, act in ordered]
    if not precursors or not product_entities or not correct_seq:
        return []

    correct_str = _render_order_sequence(correct_seq)
    candidate_seqs = [
        seq for seq in pool.get_cross_seq_distractors(correct_seq, 12, rng)
        if len(seq) == len(correct_seq)
    ]
    candidate_seqs.extend(pool.get_seq_distractors(correct_seq, 12, rng))
    distractors: list[str] = []
    seen = {_norm_text(correct_str)}
    for seq in candidate_seqs:
        opt = _render_order_sequence(seq)
        key = _norm_text(opt)
        if key in seen:
            continue
        seen.add(key)
        distractors.append(opt)
        if len(distractors) >= 3:
            break
    if len(distractors) < 3:
        return []

    choices, answer = _make_choices(correct_str, distractors, rng)
    return [{
        "task": "A1_route_retrieval",
        "question": (
            f"Study DOI: {pg.doi}\n"
            f"Process: {pg.label}\n"
            f"Recorded route length: {len(correct_seq)} operations.\n"
            f"Precursor materials: {', '.join(precursors)}.\n"
            f"Target product node(s): {', '.join(product_refs) if product_refs else 'unknown'}.\n"
            f"Which operation route matches the recorded synthesis process?"
        ),
        "choices": choices,
        "answer": answer,
        "evidence": {
            "doi": pg.doi,
            "process_label": pg.label,
            "precursors": _entity_list_brief(precursor_entities),
            "anonymized_precursors": precursors,
            "products": _entity_list_brief(product_entities),
            "anonymized_products": product_refs,
            "correct_sequence": correct_seq,
            "raw_activity_sequence": [act["label"] for _, act in ordered],
            "total_steps": len(correct_seq),
        },
    }]


def build_A2(pg: ProcessGraph, pool: DistractorPool, rng: random.Random) -> list[dict]:
    """
    A2 Missing Step Identification
    Given a route with one operation removed, predict the recorded operation label.
    Material form and masked-step conditions are not revealed; the model must infer
    the missing step from route context and anonymous I/O node references alone.
    Accepts steps with a form transition OR steps with recorded conditions.
    """
    samples = []
    ordered = [(aid, act) for aid, act in pg.ordered_activities() if act["label"]]
    if len(ordered) < 3:
        return []

    for i, (aid, act) in enumerate(ordered):
        lbl = act["label"]
        if not lbl:
            continue
        inputs = pg.activity_inputs(aid)
        outputs = pg.activity_outputs(aid)
        input_materials = _entity_list_brief(inputs)
        output_materials = _entity_list_brief(outputs)
        in_forms = [e["form"].lower() for e in inputs if e.get("form")]
        out_forms = [e["form"].lower() for e in outputs if e.get("form")]
        has_form_transition = bool(in_forms and out_forms and in_forms[0] != out_forms[0])
        has_conditions = bool(act["conditions"])
        if not has_form_transition and not has_conditions:
            continue

        # Plain anonymous refs — no form hint exposed
        input_refs  = [f"I{j + 1}" for j in range(len(inputs))]
        output_refs = [f"O{j + 1}" for j in range(len(outputs))]

        before = [a["label"] for _, a in ordered[:i] if a["label"]]
        after = [a["label"] for _, a in ordered[i + 1:] if a["label"]]
        route_parts = (
            (([" → ".join(before)] if before else []))
            + ["[MISSING STEP]"]
            + (([" → ".join(after)] if after else []))
        )
        partial_route = " → ".join(route_parts)

        if in_forms and out_forms:
            candidate_labels = pool.get_form_transition_distractors(
                in_forms[0], out_forms[0], lbl, 12, rng
            )
        else:
            candidate_labels = []
        lbl_families = _activity_families(lbl)
        if lbl_families:
            family_labels = [
                cand for cand in pool.all_activity_labels
                if cand.lower() != lbl.lower() and (_activity_families(cand) & lbl_families)
            ]
            rng.shuffle(family_labels)
            candidate_labels.extend(family_labels)
        route_labels = [
            a["label"] for _, a in ordered
            if a["label"] and a["label"].lower() != lbl.lower()
            and (not lbl_families or (_activity_families(a["label"]) & lbl_families))
        ]
        candidate_labels.extend(route_labels)
        if len(_unique_labels(candidate_labels, lbl, 3, rng)) < 3:
            candidate_labels.extend(pool.all_activity_labels)
        distractors = _unique_labels(candidate_labels, lbl, 3, rng)
        if len(distractors) < 3:
            continue

        conds = act["conditions"]
        choices, answer = _make_choices(lbl, distractors, rng)
        samples.append({
            "task": "A2_missing_step",
            "question": (
                f"Study DOI: {pg.doi}\n"
                f"Process: {pg.label}\n"
                f"Synthesis route with one operation masked:\n{partial_route}\n"
                f"Masked operation position: {i + 1} of {len(ordered)}.\n"
                f"Masked operation input node(s): {', '.join(input_refs) if input_refs else 'unknown'}.\n"
                f"Masked operation output node(s): {', '.join(output_refs) if output_refs else 'unknown'}.\n"
                f"Which operation label is recorded for the masked step?"
            ),
            "choices": choices,
            "answer": answer,
            "evidence": {
                "doi": pg.doi,
                "process_label": pg.label,
                "activity_id": aid,
                "missing_activity": lbl,
                "input_form": in_forms[0] if in_forms else "",
                "output_form": out_forms[0] if out_forms else "",
                "anonymized_inputs": input_refs,
                "anonymized_outputs": output_refs,
                "input_materials": input_materials,
                "output_materials": output_materials,
                "conditions": conds,
                "step_index": i + 1,
                "total_steps": len(ordered),
            },
        })
    return samples


def build_A3(pg: ProcessGraph, pool: DistractorPool, rng: random.Random) -> list[dict]:
    """
    A3 Next Activity Prediction
    Given ≥2 completed steps + precursors and current material state, predict the next step.
    Next-step graph signature and conditions are not revealed; the model must predict
    from the route prefix and current material node alone.
    """
    samples = []
    ordered = [(aid, act) for aid, act in pg.ordered_activities() if act["label"]]
    if len(ordered) < 3:
        return []
    precursor_entities = [e for _, e in pg.precursors()]
    precursors = _starting_material_refs(precursor_entities)
    product_entities = [e for _, e in pg.products()]
    product_refs = _anonymous_material_refs(product_entities, "P")
    aid_to_step = _operation_instance_labels(ordered)
    masked_context = _masked_activity_io_context(pg, aid_to_step)

    for k in range(2, len(ordered)):
        prefix = ordered[:k]
        last_prefix_aid, _ = ordered[k - 1]
        next_aid, next_act = ordered[k]
        next_lbl = next_act["label"]
        prefix_labels = _route_labels(prefix)
        prefix_display = [aid_to_step[aid] for aid, _ in prefix]
        if not next_lbl or not prefix_labels:
            continue

        current_materials = masked_context.get(aid_to_step[last_prefix_aid], {}).get("outputs", [])
        next_step_refs = masked_context.get(aid_to_step[next_aid], {})
        correct_inputs = next_step_refs.get("inputs", [])
        correct_outputs = next_step_refs.get("outputs", [])
        raw_current_materials = _entity_list_brief(pg.activity_outputs(last_prefix_aid))
        raw_correct_inputs = _entity_list_brief(pg.activity_inputs(next_aid))
        raw_correct_outputs = _entity_list_brief(pg.activity_outputs(next_aid))

        candidate_labels = pool.get_next_activity_distractors(prefix_labels, next_lbl, 12, rng)

        next_inputs = pg.activity_inputs(next_aid)
        next_outputs = pg.activity_outputs(next_aid)
        in_forms = [e["form"].lower() for e in next_inputs if e.get("form")]
        out_forms = [e["form"].lower() for e in next_outputs if e.get("form")]
        if in_forms and out_forms:
            candidate_labels.extend(
                pool.get_form_transition_distractors(in_forms[0], out_forms[0], next_lbl, 12, rng)
            )

        lbl_families = _activity_families(next_lbl)
        if lbl_families:
            family_labels = [
                cand for cand in pool.all_activity_labels
                if cand.lower() != next_lbl.lower() and (_activity_families(cand) & lbl_families)
            ]
            rng.shuffle(family_labels)
            candidate_labels.extend(family_labels)

        route_labels = [
            cand_act["label"] for _, cand_act in (list(ordered[k + 1:]) + list(ordered[:k]))
            if cand_act["label"].lower() != next_lbl.lower()
            and (not lbl_families or (_activity_families(cand_act["label"]) & lbl_families))
        ]
        candidate_labels.extend(route_labels)

        if len(_unique_labels(candidate_labels, next_lbl, 3, rng)) < 3:
            candidate_labels.extend(pool.all_activity_labels)

        distractors = _unique_labels(candidate_labels, next_lbl, 3, rng)
        if len(distractors) < 3:
            continue

        conds = next_act["conditions"]
        choices, answer = _make_choices(next_lbl, distractors, rng)
        samples.append({
            "task": "A3_next_activity",
            "question": (
                f"Study DOI: {pg.doi}\n"
                f"Process: {pg.label}\n"
                f"Precursor materials: {', '.join(precursors) if precursors else 'unknown'}.\n"
                f"Target product node(s): {', '.join(product_refs) if product_refs else 'unknown'}.\n"
                f"Recorded route length: {len(ordered)} operations.\n"
                f"Completed route prefix: {_numbered_route(prefix_display)}.\n"
                f"Current material node(s) after the completed prefix: "
                f"{', '.join(current_materials) if current_materials else 'unknown'}.\n"
                f"Which operation label is recorded next for this process?"
            ),
            "choices": choices,
            "answer": answer,
            "evidence": {
                "doi": pg.doi,
                "process_label": pg.label,
                "precursors": _entity_list_brief(precursor_entities),
                "anonymized_precursors": precursors,
                "products": _entity_list_brief(product_entities),
                "anonymized_products": product_refs,
                "prefix_activities": prefix_labels,
                "prefix_display": prefix_display,
                "next_activity": next_lbl,
                "anonymized_current_materials": current_materials,
                "anonymized_next_inputs": correct_inputs,
                "anonymized_next_outputs": correct_outputs,
                "raw_current_materials": raw_current_materials,
                "raw_next_inputs": raw_correct_inputs,
                "raw_next_outputs": raw_correct_outputs,
                "next_conditions": conds,
                "step_index": k,
                "total_steps": len(ordered),
            },
        })
    return samples


def build_B1(pg: ProcessGraph, pool: DistractorPool, rng: random.Random) -> list[dict]:
    """
    B1 Condition Prediction
    Given activity + other conditions + inputs, predict one masked condition.
    """
    samples = []
    COND_DESCS = {
        "temperature": "reaction temperature",
        "duration": "reaction duration",
        "atmosphere": "required atmosphere / gas environment",
        "pressure": "applied pressure",
        "rotation": "rotation / stirring speed",
        "temperature_rate": "heating or cooling rate",
    }
    ordered = [(oid, a) for oid, a in pg.ordered_activities() if a["label"]]
    step_index = {oid: i + 1 for i, (oid, _) in enumerate(ordered)}
    total_steps = len(ordered)
    route_str = " → ".join(f"{i + 1}. {a['label']}" for i, (_, a) in enumerate(ordered))

    for aid, act in ordered:
        lbl = act["label"]
        conds = act["conditions"]
        if not lbl or not conds:
            continue
        step_no = step_index[aid]
        prev_lbl = ordered[step_no - 2][1]["label"] if step_no > 1 else "START"
        next_lbl = ordered[step_no][1]["label"] if step_no < total_steps else "END"
        mat_inputs = [_entity_brief(e) for e in pg.activity_inputs(aid) if e["label"]]
        mat_outputs = [_entity_brief(e) for e in pg.activity_outputs(aid) if e["label"]]
        input_forms = [e["form"].lower() for e in pg.activity_inputs(aid) if e.get("form")]
        output_forms = [e["form"].lower() for e in pg.activity_outputs(aid) if e.get("form")]

        for cond_key, cond_desc in COND_DESCS.items():
            correct = conds.get(cond_key, "")
            if not correct:
                continue
            other_conds = {k: v for k, v in conds.items() if k != cond_key}
            context_parts = [
                f"Study DOI: {pg.doi}",
                f"Process: {pg.label}",
                f"Route: {route_str}",
                f"Target step: {step_no} of {total_steps}, {lbl}",
                f"Previous step: {prev_lbl}",
                f"Next step: {next_lbl}",
            ]
            if mat_inputs:
                context_parts.append(f"Target inputs: {', '.join(mat_inputs)}")
            if mat_outputs:
                context_parts.append(f"Target outputs: {', '.join(mat_outputs)}")
            if other_conds:
                known = "; ".join(
                    f"{k.replace('_', ' ')}: {v}" for k, v in sorted(other_conds.items())
                )
                context_parts.append(f"Known target-step conditions: {known}")

            if cond_key == "atmosphere":
                # Use canonical-aware distractors to prevent Ar/argon/argon gas
                # synonyms from appearing as wrong options for the same gas.
                raw_distractors = pool.get_atmosphere_distractors(
                    lbl, correct, 20, rng, input_forms, output_forms
                )
            else:
                raw_distractors = pool.get_cond_distractors(
                    lbl, cond_key, correct, 20, rng, input_forms, output_forms
                )
            distractors = _filter_distinct_distractors(correct, raw_distractors, cond_key, 3)
            if len(distractors) < 3:
                continue

            choices, answer = _make_choices(correct, distractors, rng)
            samples.append({
                "task": "B1_condition_prediction",
                "question": (
                    f"Given the recorded synthesis context:\n" + "\n".join(context_parts) + "\n"
                    f"What is the {cond_desc} for this operation?"
                ),
                "choices": choices,
                "answer": answer,
                "evidence": {
                    "doi": pg.doi,
                    "process_label": pg.label,
                    "activity_id": aid,
                    "activity_label": lbl,
                    "step_index": step_no,
                    "total_steps": total_steps,
                    "route": [a["label"] for _, a in ordered],
                    "input_materials": mat_inputs,
                    "output_materials": mat_outputs,
                    "predicted_condition": cond_key,
                    "all_conditions": conds,
                },
            })
    return samples


def build_B2(pg: ProcessGraph, pool: DistractorPool, rng: random.Random) -> list[dict]:
    """
    B2 Full Condition Set Prediction
    Given activity + input materials, choose the correct (temperature, duration, atmosphere) triple.
    """
    samples = []
    ordered = [(oid, a) for oid, a in pg.ordered_activities() if a["label"]]
    step_index = {oid: i + 1 for i, (oid, _) in enumerate(ordered)}
    total_steps = len(ordered)
    route_str = " → ".join(f"{i + 1}. {a['label']}" for i, (_, a) in enumerate(ordered))

    for aid, act in ordered:
        lbl = act["label"]
        conds = act["conditions"]
        temp = conds.get("temperature", "")
        dur = conds.get("duration", "")
        atm = conds.get("atmosphere", "")
        if not lbl or not temp or not dur or not atm:
            continue

        step_no = step_index[aid]
        prev_lbl = ordered[step_no - 2][1]["label"] if step_no > 1 else "START"
        next_lbl = ordered[step_no][1]["label"] if step_no < total_steps else "END"
        mat_inputs = [_entity_brief(e) for e in pg.activity_inputs(aid) if e["label"]]
        mat_outputs = [_entity_brief(e) for e in pg.activity_outputs(aid) if e["label"]]
        input_forms = [e["form"].lower() for e in pg.activity_inputs(aid) if e.get("form")]
        output_forms = [e["form"].lower() for e in pg.activity_outputs(aid) if e.get("form")]
        correct_str = f"temperature: {temp}; duration: {dur}; atmosphere: {atm}"
        correct_conds = {"temperature": temp, "duration": dur, "atmosphere": atm}

        temp_d = _filter_distinct_distractors(
            temp,
            pool.get_cond_distractors(lbl, "temperature", temp, 20, rng, input_forms, output_forms),
            "temperature",
            6,
        )
        dur_d = _filter_distinct_distractors(
            dur,
            pool.get_cond_distractors(lbl, "duration", dur, 20, rng, input_forms, output_forms),
            "duration",
            6,
        )
        atm_d = _filter_distinct_distractors(
            atm,
            pool.get_atmosphere_distractors(lbl, atm, 20, rng, input_forms, output_forms),
            "atmosphere",
            6,
        )
        if len(temp_d) < 2 or len(dur_d) < 2:
            continue

        def _triple(t: str, d: str, a: str) -> str:
            return f"temperature: {t}; duration: {d}; atmosphere: {a}"

        def _triple_key(t: str, d: str, a: str) -> tuple:
            return (
                _condition_value_key("temperature", t),
                _condition_value_key("duration", d),
                _condition_value_key("atmosphere", a),
            )

        graph_condition_sets = pool.get_condition_set_distractors(
            lbl, correct_conds, 6, rng, input_forms, output_forms
        )
        dist_candidates = [
            (rec["temperature"], rec["duration"], rec["atmosphere"])
            for rec in graph_condition_sets
        ] + [
            (temp_d[0], dur_d[0], atm),
            (temp_d[1], dur, atm_d[0] if atm_d else atm),
            (temp, dur_d[1], atm_d[1] if len(atm_d) > 1 else (atm_d[0] if atm_d else atm)),
            (temp_d[0], dur, atm_d[2] if len(atm_d) > 2 else (atm_d[0] if atm_d else atm)),
            (temp, dur_d[0], atm),
            (temp_d[1], dur_d[1], atm_d[0] if atm_d else atm),
        ]
        seen_keys = {_triple_key(temp, dur, atm)}
        distractors: list[str] = []
        for t_d, d_d, a_d in dist_candidates:
            key = _triple_key(t_d, d_d, a_d)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            distractors.append(_triple(t_d, d_d, a_d))
            if len(distractors) >= 3:
                break
        if len(distractors) < 3:
            continue

        choices, answer = _make_choices(correct_str, distractors, rng)
        samples.append({
            "task": "B2_full_condition_set",
            "question": (
                "Given the recorded synthesis context:\n"
                f"Study DOI: {pg.doi}\n"
                f"Process: {pg.label}\n"
                f"Route: {route_str}\n"
                f"Target step: {step_no} of {total_steps}, {lbl}\n"
                f"Previous step: {prev_lbl}\n"
                f"Next step: {next_lbl}\n"
                + (f"Target inputs: {', '.join(mat_inputs)}\n" if mat_inputs else "")
                + (f"Target outputs: {', '.join(mat_outputs)}\n" if mat_outputs else "")
                + "Which complete set of process conditions is recorded for this target step?"
            ),
            "choices": choices,
            "answer": answer,
            "evidence": {
                "doi": pg.doi,
                "process_label": pg.label,
                "activity_id": aid,
                "activity_label": lbl,
                "step_index": step_no,
                "total_steps": total_steps,
                "route": [a["label"] for _, a in ordered],
                "input_materials": mat_inputs,
                "output_materials": mat_outputs,
                "conditions": {"temperature": temp, "duration": dur, "atmosphere": atm},
            },
        })
    return samples


def build_C1(pg: ProcessGraph, pool: DistractorPool, rng: random.Random) -> list[dict]:
    """
    C1 Tool Selection
    Given activity + conditions + input materials, predict the required tool.
    """
    samples = []
    ordered = [(oid, a) for oid, a in pg.ordered_activities() if a["label"]]
    step_index = {oid: i + 1 for i, (oid, _) in enumerate(ordered)}
    total_steps = len(ordered)
    route_str = _numbered_route(_route_labels(ordered))

    for aid, act in ordered:
        lbl = act["label"]
        if not lbl:
            continue
        tools = pg.activity_tools(aid)
        if not tools:
            continue
        tool_labels = sorted({t["label"] for t in tools if t["label"]})

        step_no = step_index[aid]
        prev_lbl = ordered[step_no - 2][1]["label"] if step_no > 1 else "START"
        next_lbl = ordered[step_no][1]["label"] if step_no < total_steps else "END"
        mat_inputs = _entity_list_brief(pg.activity_inputs(aid))
        mat_outputs = _entity_list_brief(pg.activity_outputs(aid))
        conds = act["conditions"]
        context_parts = [
            f"Study DOI: {pg.doi}",
            f"Process: {pg.label}",
            f"Route: {route_str}",
            f"Target step: {step_no} of {total_steps}, {lbl}",
            f"Previous step: {prev_lbl}",
            f"Next step: {next_lbl}",
        ]
        if mat_inputs:
            context_parts.append(f"Target inputs: {', '.join(mat_inputs)}")
        if mat_outputs:
            context_parts.append(f"Target outputs: {', '.join(mat_outputs)}")
        if conds:
            context_parts.append(
                "Known target-step conditions: "
                + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in sorted(conds.items()))
            )

        correct_str = ", ".join(tool_labels)
        distractors = pool.get_tool_distractors(lbl, correct_str, 3, rng)
        if not distractors:
            continue

        choices, answer = _make_choices(correct_str, distractors, rng)
        samples.append({
            "task": "C1_tool_selection",
            "question": (
                "Given the recorded synthesis context:\n"
                + "\n".join(context_parts)
                + "\nThe non-material Usage entity for the target step is masked. "
                  "Which tool, equipment, vessel, accessory, or processing medium is recorded?"
            ),
            "choices": choices,
            "answer": answer,
            "evidence": {
                "doi": pg.doi,
                "process_label": pg.label,
                "activity_id": aid,
                "activity_label": lbl,
                "step_index": step_no,
                "total_steps": total_steps,
                "route": [a["label"] for _, a in ordered],
                "input_materials": mat_inputs,
                "output_materials": mat_outputs,
                "conditions": conds,
                "tools": tool_labels,
            },
        })
    return samples


def build_D1(pg: ProcessGraph, pool: DistractorPool, rng: random.Random) -> list[dict]:
    """
    D1 Process Ordering
    Given starting materials and operation instances, choose the physically correct ordering.
    Material-flow hints are not shown; the model must determine the sequence using
    domain knowledge and any retrieved analogous processes.
    Distractor generation uses the internal edge graph (not shown to model).
    """
    labeled = [(aid, act) for aid, act in pg.ordered_activities() if act["label"]]
    if len(labeled) < 3:
        return []

    aid_to_step = _operation_instance_labels(labeled)
    correct_seq = [aid_to_step[aid] for aid, _ in labeled]
    correct_pos = {step: i for i, step in enumerate(correct_seq)}
    step_context = _activity_io_context(pg, aid_to_step)
    correct_str = _render_order_sequence(correct_seq)

    gen_map = {eid: aid for aid, eid in pg.generation}
    entity_aliases: dict[str, str] = {}

    def _entity_alias(eid: str) -> str:
        if eid not in entity_aliases:
            ent = pg.entities.get(eid, {})
            form = _masked_form(ent)
            alias = f"M{len(entity_aliases) + 1}"
            entity_aliases[eid] = f"{alias} ({form})" if form else alias
        return entity_aliases[eid]

    edges: list[tuple[str, str, str]] = []
    labeled_edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for bid, b_act in labeled:
        consumed_eids = [eid for eid, aid in pg.usage
                         if aid == bid and eid in pg.entities
                         and pg.entities[eid]["entity_type"] == "material"]
        for eid in consumed_eids:
            pred_aid = gen_map.get(eid)
            if pred_aid and pred_aid != bid and pred_aid in aid_to_step and bid in aid_to_step:
                producer_step = aid_to_step[pred_aid]
                consumer_step = aid_to_step[bid]
                if correct_pos[producer_step] >= correct_pos[consumer_step]:
                    continue
                ent_alias = _entity_alias(eid)
                edge = (producer_step, ent_alias, consumer_step)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edges.append(edge)
                    labeled_edges.append({
                        "producer": producer_step,
                        "entity_alias": ent_alias,
                        "entity_label": _entity_brief(pg.entities[eid]),
                        "consumer": consumer_step,
                    })

    if len(edges) < 2:
        return []

    dist_seqs, shown_edges = _sample_order_distractors(correct_seq, edges, rng, 3)
    if not dist_seqs or not shown_edges:
        return []
    distractors = [_render_order_sequence(s) for s in dist_seqs]

    precursors = _entity_list_brief([e for _, e in pg.precursors()])
    products = _entity_list_brief([e for _, e in pg.products()])
    shuffled_instances = list(correct_seq)
    for _ in range(8):
        rng.shuffle(shuffled_instances)
        if shuffled_instances != correct_seq:
            break
    if shuffled_instances == correct_seq:
        shuffled_instances = list(reversed(correct_seq))
    operation_instances = ", ".join(shuffled_instances)

    choices, answer = _make_choices(correct_str, distractors, rng)
    return [{
        "task": "D1_process_ordering",
        "question": (
            f"Study DOI: {pg.doi}\n"
            f"Process: {pg.label}\n"
            f"Starting material(s): {', '.join(precursors) if precursors else 'unknown'}\n"
            f"Target product(s): {', '.join(products) if products else 'unknown'}\n"
            f"Operation instances to arrange: {operation_instances}\n"
            f"Which operation order is the physically correct synthesis sequence for this process?"
        ),
        "choices": choices,
        "answer": answer,
        "evidence": {
            "doi": pg.doi,
            "process_label": pg.label,
            "correct_sequence": correct_seq,
            "raw_activity_sequence": [act["label"] for _, act in labeled],
            "operation_instances": shuffled_instances,
            "operation_io": step_context,
            "provenance_edges": shown_edges,
            "all_provenance_edges": edges,
            "labeled_provenance_edges": labeled_edges,
        },
    }]


# ── main pipeline ──────────────────────────────────────────────────────────────

def build_all_tasks(records: list[dict], rng: random.Random) -> list[dict]:
    samples: list[dict] = []

    print("Parsing process graphs...")
    all_pgs: list[ProcessGraph] = []
    for rec in records:
        pg = ProcessGraph(rec)
        if not pg.is_valid():
            continue
        all_pgs.append(pg)
    print(f"Valid process graphs: {len(all_pgs)}")

    print("Building distractor pool...")
    pool = DistractorPool(all_pgs)

    print("Building per-process tasks (A1–D1)...")
    for pg in all_pgs:
        samples.extend(build_A1(pg, pool, rng))
        samples.extend(build_A2(pg, pool, rng))
        samples.extend(build_A3(pg, pool, rng))
        samples.extend(build_B1(pg, pool, rng))
        samples.extend(build_B2(pg, pool, rng))
        samples.extend(build_C1(pg, pool, rng))
        samples.extend(build_D1(pg, pool, rng))

    for idx, s in enumerate(samples):
        s["qid"] = str(idx)

    return samples




# ── DOI metadata (starrydata) ─────────────────────────────────────────────────

# Priority order follows MatPROV paper Figure 4; GeneralDB is excluded.
_PRIORITY_PROJECTS: list[str] = [
    "ThermoelectricMaterials",
    "MagneticMaterials",
    "BatteryMaterials",
    "SolidStateBatteryMaterials",
    "CondensedMatter",
    "LowThermalConductivityMaterials",
    "HighThermalConductivityMaterials",
    "Hypermaterial",
    "Resistor",
]
_PROJECT_TO_TYPE: dict[str, str] = {
    "ThermoelectricMaterials":         "Thermoelectric",
    "MagneticMaterials":               "Magnetic",
    "BatteryMaterials":                "Battery",
    "SolidStateBatteryMaterials":      "Battery",
    "CondensedMatter":                 "Condensed",
    "LowThermalConductivityMaterials": "LowTC",
    "HighThermalConductivityMaterials":"HighTC",
    "Hypermaterial":                   "Hypermaterial",
    "Resistor":                        "Resistor",
}


def load_doi_meta(starrydata_path: str) -> dict[str, dict]:
    """
    Parse starrydata_papers.csv and return a mapping:
      doi (lower-cased) -> {"year": int | None, "mat_type": str}

    mat_type is assigned by the first match in _PRIORITY_PROJECTS; falls back
    to "Other" when no recognised project is found (e.g. GeneralDB-only DOIs).
    """
    import csv as _csv

    doi_map: dict[str, dict] = {}
    with open(starrydata_path, encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            doi = row["DOI"].strip().lower()
            if not doi:
                continue

            year: int | None = None
            try:
                parts = json.loads(row.get("issued", "{}")).get("date_parts", [[]])[0]
                year = int(parts[0]) if parts else None
            except Exception:
                pass

            mat_type = "Other"
            try:
                projects: list[str] = json.loads(row.get("project_names", "[]"))
                for p in _PRIORITY_PROJECTS:
                    if p in projects:
                        mat_type = _PROJECT_TO_TYPE[p]
                        break
            except Exception:
                pass

            doi_map[doi] = {"year": year, "mat_type": mat_type}
    return doi_map


def _sample_meta(sample: dict, doi_meta: dict[str, dict]) -> dict:
    """Return {"year": ..., "mat_type": ...} for a QA sample."""
    doi = sample.get("evidence", {}).get("doi", "").lower()
    return doi_meta.get(doi, {"year": None, "mat_type": "Other"})


# ── Split strategies ──────────────────────────────────────────────────────────

def split_random(samples: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """Baseline random 80 / 10 / 10 split (original behaviour)."""
    xs = list(samples)
    rng.shuffle(xs)
    n = len(xs)
    n_tr = int(n * 0.8)
    n_dv = int(n * 0.1)
    return {
        "train": xs[:n_tr],
        "dev":   xs[n_tr: n_tr + n_dv],
        "test":  xs[n_tr + n_dv:],
    }


def split_by_year(
    samples: list[dict],
    doi_meta: dict[str, dict],
) -> dict[str, list[dict]]:
    """
    Temporal split — guaranteed zero DOI overlap across partitions:
      train : year ≤ 2019  (records with no year info also land here)
      dev   : year == 2020
      test  : year ≥ 2021
    """
    train, dev, test = [], [], []
    for s in samples:
        yr = _sample_meta(s, doi_meta)["year"]
        if yr is None or yr <= 2019:
            train.append(s)
        elif yr == 2020:
            dev.append(s)
        else:
            test.append(s)
    return {"train": train, "dev": dev, "test": test}


def split_by_type(
    samples: list[dict],
    doi_meta: dict[str, dict],
    rng: random.Random,
) -> dict[str, list[dict]]:
    """
    Material-type split — Combination A:
      test  : all Battery records
      dev   : random 10 % of non-Battery records
      train : remaining 90 % of non-Battery records

    Battery (Li/Na-ion + solid-state battery) forms a chemically distinct
    OOD test set relative to the Thermoelectric / Magnetic training domain.
    """
    test, non_battery = [], []
    for s in samples:
        if _sample_meta(s, doi_meta)["mat_type"] == "Battery":
            test.append(s)
        else:
            non_battery.append(s)
    rng.shuffle(non_battery)
    n_dv = int(len(non_battery) * 0.1)
    return {
        "train": non_battery[n_dv:],
        "dev":   non_battery[:n_dv],
        "test":  test,
    }


def split_additional(
    samples: list[dict],
    doi_meta: dict[str, dict],
) -> dict[str, list[dict]]:
    """
    Double-OOD split (year × material type):
      train    : non-Battery  AND  year ≤ 2019
      dev      : non-Battery  AND  year == 2020
      test     : Battery      AND  year ≥ 2021
      excluded : everything else
                 (Battery pre-2021; non-Battery post-2019; no-doi records)

    This is the strictest evaluation setting: the model is trained on old
    thermoelectric/magnetic literature and tested on recent battery literature.
    """
    train, dev, test, excluded = [], [], [], []
    for s in samples:
        meta = _sample_meta(s, doi_meta)
        yr, mt = meta["year"], meta["mat_type"]
        is_battery = mt == "Battery"
        if not is_battery and (yr is None or yr <= 2019):
            train.append(s)
        elif not is_battery and yr == 2020:
            dev.append(s)
        elif is_battery and yr is not None and yr >= 2021:
            test.append(s)
        else:
            excluded.append(s)
    return {"train": train, "dev": dev, "test": test, "excluded": excluded}


# ── Split I/O helpers ─────────────────────────────────────────────────────────

def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_stats(split: dict[str, list[dict]]) -> dict:
    from collections import Counter
    stats: dict[str, Any] = {}
    for part, rows in split.items():
        task_cnt = Counter(r["task"] for r in rows)
        stats[part] = {
            "n": len(rows),
            "tasks": dict(sorted(task_cnt.items())),
        }
    return stats


def save_split(
    split: dict[str, list[dict]],
    output_dir: Path,
    split_name: str,
) -> None:
    """Write each partition to <output_dir>/<split_name>/<part>.jsonl."""
    folder = output_dir / split_name
    folder.mkdir(parents=True, exist_ok=True)
    for part, rows in split.items():
        write_jsonl(folder / f"{part}.jsonl", rows)
    stats = _split_stats(split)
    (folder / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[{split_name}]  →  {folder}")
    for part, info in stats.items():
        print(f"  {part:10s}: {info['n']:6d} records")


def print_stats(samples: list[dict]) -> None:
    from collections import Counter
    task_counts = Counter(s["task"] for s in samples)
    print(f"\nTotal samples: {len(samples)}")
    print("By task:")
    for task, cnt in sorted(task_counts.items()):
        print(f"  {task}: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MatProcBench MCQ dataset from MatPROV.jsonl"
    )
    parser.add_argument(
        "--input",
        default="data/raw_data/MatPROV.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
    )
    parser.add_argument(
        "--starrydata",
        default="data/starrydata_dataset_20260406/starrydata_papers.csv",
        help="Path to starrydata_papers.csv (provides year + material type per DOI).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=0,
                        help="Limit input records for quick testing (0 = all)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Load raw process graphs ───────────────────────────────────────────────
    print(f"Loading {args.input} ...")
    records = []
    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_records > 0 and i >= args.max_records:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records.")

    # ── Load DOI metadata (year + material type) ──────────────────────────────
    doi_meta: dict[str, dict] = {}
    if args.starrydata and Path(args.starrydata).exists():
        print(f"Loading DOI metadata from {args.starrydata} ...")
        doi_meta = load_doi_meta(args.starrydata)
        print(f"  {len(doi_meta)} DOIs indexed.")
    else:
        print("Warning: --starrydata not found; year/type splits will be skipped.")

    # ── Build QA samples ──────────────────────────────────────────────────────
    samples = build_all_tasks(records, rng)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write full dataset
    write_jsonl(output_dir / "data_full.jsonl", samples)
    print_stats(samples)

    # ── Save full-dataset stats ───────────────────────────────────────────────
    from collections import Counter
    task_counts = Counter(s["task"] for s in samples)
    (output_dir / "stats.json").write_text(
        json.dumps({
            "total": len(samples),
            "task_counts": dict(task_counts),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── Split 1: random (baseline) ────────────────────────────────────────────
    save_split(split_random(samples, rng), output_dir, "random_split")

    if not doi_meta:
        print("\nSkipping year/type splits (no starrydata).")
        print(f"\nAll outputs saved to: {output_dir}")
        return

    # ── Split 2: year ─────────────────────────────────────────────────────────
    save_split(split_by_year(samples, doi_meta), output_dir, "year_split")

    # ── Split 3: material type (Combination A — Battery as test) ─────────────
    save_split(split_by_type(samples, doi_meta, rng), output_dir, "type_split")

    # ── Split 4: double OOD (year × material type) ────────────────────────────
    save_split(split_additional(samples, doi_meta), output_dir, "additional_split")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
