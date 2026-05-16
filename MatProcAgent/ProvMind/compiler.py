from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


def _val(field: Any) -> str:
    """Extract a readable string from JSON-LD value fields."""
    if isinstance(field, list):
        for item in field:
            if isinstance(item, dict):
                value = item.get("@value", "")
                if value:
                    return str(value).strip()
            elif isinstance(item, str) and item.strip():
                return item.strip()
    if isinstance(field, dict):
        return str(field.get("@value", "")).strip()
    if isinstance(field, str):
        return field.strip()
    return ""


@dataclass
class EntityState:
    entity_id: str
    label: str
    entity_type: str
    form: str = ""
    purity: str = ""
    role: str = ""


@dataclass
class ActivityState:
    activity_id: str
    label: str
    conditions: dict[str, str] = field(default_factory=dict)
    order: int = -1


class ProcessState:
    """Compact executable view of a single MatPROV process graph."""

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

    def __init__(self, record: dict[str, Any]) -> None:
        self.doi = record.get("doi", "")
        self.label = record.get("label", "")
        self.entities: dict[str, EntityState] = {}
        self.activities: dict[str, ActivityState] = {}
        self.usage: list[tuple[str, str]] = []
        self.generation: list[tuple[str, str]] = []
        self._parse(record.get("prov_jsonld", {}).get("@graph", []))
        self._assign_roles()
        self._order_activities()
        self._build_instance_labels()
        self._build_material_aliases()

    def _parse(self, graph: list[dict[str, Any]]) -> None:
        for item in graph:
            if not isinstance(item, dict):
                continue
            obj_type = item.get("@type", "")
            local_id = item.get("@id", "")
            if obj_type == "Entity":
                entity_type_raw = _val(item.get("type", "material"))
                entity_type = "tool" if entity_type_raw == "tool" else "material"
                self.entities[local_id] = EntityState(
                    entity_id=local_id,
                    label=_val(item.get("label", "")),
                    entity_type=entity_type,
                    purity=_val(item.get("matprov:purity")),
                    form=_val(item.get("matprov:form")),
                )
            elif obj_type == "Activity":
                conds: dict[str, str] = {}
                for raw_key, clean_key in self.COND_KEYS.items():
                    value = _val(item.get(raw_key))
                    if value:
                        conds[clean_key] = value
                self.activities[local_id] = ActivityState(
                    activity_id=local_id,
                    label=_val(item.get("label", "")),
                    conditions=conds,
                )
            elif obj_type == "Usage":
                entity_id = item.get("entity", "")
                activity_id = item.get("activity", "")
                if entity_id and activity_id:
                    self.usage.append((entity_id, activity_id))
            elif obj_type == "Generation":
                activity_id = item.get("activity", "")
                entity_id = item.get("entity", "")
                if entity_id and activity_id:
                    self.generation.append((activity_id, entity_id))

    def _assign_roles(self) -> None:
        generated_by: dict[str, list[str]] = defaultdict(list)
        used_in: dict[str, list[str]] = defaultdict(list)
        for activity_id, entity_id in self.generation:
            generated_by[entity_id].append(activity_id)
        for entity_id, activity_id in self.usage:
            used_in[entity_id].append(activity_id)

        for entity_id, entity in self.entities.items():
            in_degree = len(generated_by[entity_id])
            out_degree = len(used_in[entity_id])
            if in_degree == 0 and out_degree > 0:
                entity.role = "precursor"
            elif in_degree > 0 and out_degree > 0:
                entity.role = "intermediate"
            elif in_degree > 0 and out_degree == 0:
                entity.role = "product"
            else:
                entity.role = "isolated"

        self._generated_by = generated_by
        self._used_in = used_in

    def _order_activities(self) -> None:
        producer_of: dict[str, str] = {}
        for activity_id, entity_id in self.generation:
            producer_of[entity_id] = activity_id

        predecessors: dict[str, set[str]] = defaultdict(set)
        for entity_id, consumer_id in self.usage:
            producer_id = producer_of.get(entity_id)
            if producer_id and producer_id != consumer_id:
                predecessors[consumer_id].add(producer_id)

        in_degree = {activity_id: len(predecessors.get(activity_id, set())) for activity_id in self.activities}
        queue = deque(activity_id for activity_id, degree in in_degree.items() if degree == 0)

        order = 0
        while queue:
            activity_id = queue.popleft()
            self.activities[activity_id].order = order
            order += 1
            for generated_activity_id, entity_id in self.generation:
                if generated_activity_id != activity_id:
                    continue
                for used_entity_id, next_activity_id in self.usage:
                    if used_entity_id != entity_id or next_activity_id not in in_degree:
                        continue
                    in_degree[next_activity_id] -= 1
                    if in_degree[next_activity_id] == 0:
                        queue.append(next_activity_id)

        for activity_id, activity in self.activities.items():
            if activity.order < 0:
                activity.order = order
                order += 1

    def _build_instance_labels(self) -> None:
        ordered = self.ordered_activities()
        totals = Counter(activity.label for _, activity in ordered if activity.label)
        seen: dict[str, int] = defaultdict(int)
        self.instance_labels: dict[str, str] = {}
        for activity_id, activity in ordered:
            if totals[activity.label] > 1:
                seen[activity.label] += 1
                self.instance_labels[activity_id] = f"{activity.label} #{seen[activity.label]}"
            else:
                self.instance_labels[activity_id] = activity.label

    def _build_material_aliases(self) -> None:
        self.material_aliases: dict[str, str] = {}
        counter = 1
        for activity_id, _activity in self.ordered_activities():
            for generated_activity_id, entity_id in self.generation:
                if generated_activity_id != activity_id:
                    continue
                entity = self.entities.get(entity_id)
                if not entity or entity.entity_type != "material" or entity_id in self.material_aliases:
                    continue
                self.material_aliases[entity_id] = f"M{counter}"
                counter += 1

        self.product_aliases: dict[str, str] = {}
        for index, (entity_id, _entity) in enumerate(self.products(), start=1):
            self.product_aliases[entity_id] = f"P{index}"

    def ordered_activities(self) -> list[tuple[str, ActivityState]]:
        return sorted(self.activities.items(), key=lambda item: item[1].order)

    def ordered_activity_ids(self) -> list[str]:
        return [activity_id for activity_id, _activity in self.ordered_activities()]

    def step_count(self) -> int:
        return len(self.activities)

    def step_by_position(self, position: int) -> tuple[str, ActivityState]:
        ordered = self.ordered_activities()
        if position < 1 or position > len(ordered):
            raise IndexError(f"step position out of range: {position}")
        return ordered[position - 1]

    def activity_inputs(self, activity_id: str) -> list[EntityState]:
        return [
            self.entities[entity_id]
            for entity_id, linked_activity_id in self.usage
            if linked_activity_id == activity_id
            and entity_id in self.entities
            and self.entities[entity_id].entity_type == "material"
        ]

    def activity_outputs(self, activity_id: str) -> list[EntityState]:
        return [
            self.entities[entity_id]
            for linked_activity_id, entity_id in self.generation
            if linked_activity_id == activity_id
            and entity_id in self.entities
        ]

    def activity_tools(self, activity_id: str) -> list[EntityState]:
        return [
            self.entities[entity_id]
            for entity_id, linked_activity_id in self.usage
            if linked_activity_id == activity_id
            and entity_id in self.entities
            and self.entities[entity_id].entity_type == "tool"
        ]

    def precursors(self) -> list[tuple[str, EntityState]]:
        return [
            (entity_id, entity)
            for entity_id, entity in self.entities.items()
            if entity.role == "precursor" and entity.entity_type == "material"
        ]

    def products(self) -> list[tuple[str, EntityState]]:
        return [
            (entity_id, entity)
            for entity_id, entity in self.entities.items()
            if entity.role == "product" and entity.entity_type == "material"
        ]

    def route_labels(self, instance_aware: bool = False) -> list[str]:
        labels: list[str] = []
        for activity_id, activity in self.ordered_activities():
            labels.append(self.instance_labels[activity_id] if instance_aware else activity.label)
        return labels

    def joined_tools(self, activity_id: str) -> str:
        labels = sorted({tool.label for tool in self.activity_tools(activity_id) if tool.label})
        return ", ".join(labels)

    def provenance_edges(self) -> list[tuple[str, str, str]]:
        producer_for_entity = {entity_id: activity_id for activity_id, entity_id in self.generation}
        edges: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        positions = {label: idx for idx, label in enumerate(self.route_labels(instance_aware=True))}
        for entity_id, consumer_activity_id in self.usage:
            entity = self.entities.get(entity_id)
            if not entity or entity.entity_type != "material":
                continue
            producer_activity_id = producer_for_entity.get(entity_id)
            if not producer_activity_id or producer_activity_id == consumer_activity_id:
                continue
            producer_step = self.instance_labels[producer_activity_id]
            consumer_step = self.instance_labels[consumer_activity_id]
            if positions.get(producer_step, -1) >= positions.get(consumer_step, -1):
                continue
            alias = self.material_aliases.get(entity_id, entity.label)
            edge = (producer_step, alias, consumer_step)
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
        return edges
