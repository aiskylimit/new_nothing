"""Canonical schema model and adapters for the three benchmark formats.

The canonical representation intentionally preserves graph-schema semantics while
removing benchmark-specific field names:

* node unit: label and its properties;
* relation unit: source label, relationship type, target label, and properties.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PROPERTY_RE = re.compile(
    r"^\s*-\s+`(?P<name>[^`]+)`\s*:\s*(?P<type>[A-Za-z_][A-Za-z0-9_\[\], ]*)"
)
_GROUP_HEADER_RE = re.compile(
    r"^\s*-\s+(?:\*\*(?P<bold>[^*]+)\*\*|`(?P<tick>[^`]+)`|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))\s*$"
)
_RELATION_PATTERN_RE = re.compile(
    r"\((?P<left>[^()]*)\)\s*(?:(?P<left_arrow><-)|-)\s*"
    r"\[(?P<relation>[^\]]*)\]\s*(?:(?P<right_arrow>->)|-)\s*\((?P<right>[^()]*)\)"
)
_OVERLAPPING_RELATION_PATTERN_RE = re.compile(r"(?=" + _RELATION_PATTERN_RE.pattern + r")")
_INLINE_GROUP_RE = re.compile(
    r"(?P<label>:?`[^`]+`|:?[A-Za-z_][A-Za-z0-9_]*)\s*\{(?P<properties>[^{}]*)\}"
)
_RELEVANT_RELATION_RE = re.compile(
    r"\{\s*['\"]?start['\"]?\s*:\s*(?P<source>`?[^,` }]+`?)\s*,\s*"
    r"['\"]?type['\"]?\s*:\s*(?P<relation_type>`?[^,` }]+`?)\s*,\s*"
    r"['\"]?end['\"]?\s*:\s*(?P<target>`?[^,` }]+`?)\s*\}"
)
_NEO4J_REPR_NODE_RE = re.compile(
    r"labels=frozenset\(\{['\"](?P<label>[^'\"]+)['\"]\}\)"
)
_NEO4J_REPR_RELATION_RE = re.compile(
    r"nodes=\(\s*<Node[^>]*?labels=frozenset\(\{['\"](?P<source>[^'\"]+)['\"]\}\)"
    r"[^>]*>,\s*<Node[^>]*?labels=frozenset\(\{['\"](?P<target>[^'\"]+)['\"]\}\)"
    r"[^>]*>\)\s*type=['\"](?P<relation_type>[^'\"]+)['\"]",
    re.DOTALL,
)


def clean_identifier(value: Any) -> str:
    """Return a Cypher identifier without surrounding whitespace/backticks."""

    text = str(value).strip()
    if text.startswith("`") and text.endswith("`") and len(text) >= 2:
        text = text[1:-1]
    return text.lstrip(":").strip()


def normalize_property_type(value: Any) -> str:
    """Map frequent source type spellings to a small, stable vocabulary."""

    raw = str(value or "UNKNOWN").strip()
    normalized = raw.upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "STR": "STRING",
        "STRING": "STRING",
        "TEXT": "STRING",
        "INT": "INTEGER",
        "INTEGER": "INTEGER",
        "LONG": "INTEGER",
        "FLOAT": "FLOAT",
        "DOUBLE": "FLOAT",
        "DECIMAL": "FLOAT",
        "NUMBER": "FLOAT",
        "BOOL": "BOOLEAN",
        "BOOLEAN": "BOOLEAN",
        "DATETIME": "DATE_TIME",
        "DATE_TIME": "DATE_TIME",
        "DATE": "DATE",
        "TIME": "TIME",
        "DURATION": "DURATION",
        "POINT": "POINT",
        "LIST": "LIST",
        "MAP": "MAP",
        "UNKNOWN": "UNKNOWN",
    }
    if normalized.startswith("LIST"):
        return "LIST"
    return aliases.get(normalized, normalized)


def _coerce_properties(properties: Any) -> tuple[tuple[str, str], ...]:
    """Coerce dict/list property representations into sorted canonical pairs."""

    values: dict[str, str] = {}
    if isinstance(properties, Mapping):
        iterator = properties.items()
        for name, value in iterator:
            if isinstance(value, Mapping):
                value = value.get("datatype", value.get("type", "UNKNOWN"))
            values[clean_identifier(name)] = normalize_property_type(value)
    elif isinstance(properties, Sequence) and not isinstance(properties, str | bytes):
        for item in properties:
            if not isinstance(item, Mapping):
                continue
            name = item.get("property", item.get("name"))
            if name is None:
                continue
            value = item.get("datatype", item.get("type", "UNKNOWN"))
            values[clean_identifier(name)] = normalize_property_type(value)
    return tuple(sorted(values.items()))


@dataclass(frozen=True)
class NodeUnit:
    label: str
    properties: tuple[tuple[str, str], ...] = ()

    @property
    def id(self) -> str:
        return f"node:{self.label}"

    def to_dict(self) -> dict[str, Any]:
        """Return the node's canonical schema representation."""

        return {
            "label": self.label,
            "properties": dict(self.properties),
        }

    def to_unit_dict(self) -> dict[str, Any]:
        """Return selection-task metadata for this atomic schema unit."""

        return {
            "id": self.id,
            "kind": "node",
            "schema": self.to_dict(),
            "text": self.text,
        }

    @property
    def text(self) -> str:
        properties = ", ".join(f"{name}: {kind}" for name, kind in self.properties)
        return f"(:{self.label}" + (f" {{ {properties} }}" if properties else "") + ")"


@dataclass(frozen=True)
class RelationUnit:
    source: str
    relation_type: str
    target: str
    properties: tuple[tuple[str, str], ...] = ()

    @property
    def id(self) -> str:
        return f"relation:{self.source}|{self.relation_type}|{self.target}"

    def to_dict(self) -> dict[str, Any]:
        """Return the relationship's canonical schema representation."""

        return {
            "source": self.source,
            "type": self.relation_type,
            "target": self.target,
            "properties": dict(self.properties),
        }

    def to_unit_dict(self) -> dict[str, Any]:
        """Return selection-task metadata for this atomic schema unit."""

        return {
            "id": self.id,
            "kind": "relation",
            "schema": self.to_dict(),
            "text": self.text,
        }

    @property
    def text(self) -> str:
        properties = ", ".join(f"{name}: {kind}" for name, kind in self.properties)
        pattern = f"(:{self.source})-[:{self.relation_type}]->(:{self.target})"
        return pattern + (f" {{ {properties} }}" if properties else "")


@dataclass(frozen=True)
class CanonicalSchema:
    """Benchmark-independent graph schema used by downstream data generation."""

    source: str
    graph: str
    nodes: tuple[NodeUnit, ...]
    relations: tuple[RelationUnit, ...]

    @property
    def schema_id(self) -> str:
        payload = {
            "source": self.source,
            "graph": self.graph,
            "nodes": [node.to_dict() for node in self.nodes],
            "relationships": [relation.to_dict() for relation in self.relations],
        }
        digest = sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return f"{self.source}:{self.graph}:{digest}"

    @property
    def node_by_label(self) -> dict[str, NodeUnit]:
        return {node.label: node for node in self.nodes}

    @property
    def relation_by_id(self) -> dict[str, RelationUnit]:
        return {relation.id: relation for relation in self.relations}

    @property
    def units(self) -> tuple[NodeUnit | RelationUnit, ...]:
        return (*self.nodes, *self.relations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "benchmark": self.source,
            "graph": self.graph,
            "nodes": [node.to_dict() for node in self.nodes],
            "relationships": [relation.to_dict() for relation in self.relations],
        }

    def render(
        self,
        node_ids: Iterable[str] | None = None,
        relation_ids: Iterable[str] | None = None,
    ) -> str:
        """Render a full schema or a selected sub-schema deterministically."""

        wanted_nodes = set(node_ids) if node_ids is not None else None
        wanted_relations = set(relation_ids) if relation_ids is not None else None
        nodes = [node for node in self.nodes if wanted_nodes is None or node.id in wanted_nodes]
        relations = [
            relation
            for relation in self.relations
            if wanted_relations is None or relation.id in wanted_relations
        ]
        lines = ["Nodes:"]
        lines.extend(f"- {node.text}" for node in nodes)
        lines.append("Relationships:")
        lines.extend(f"- {relation.text}" for relation in relations)
        return "\n".join(lines)

    def subset_dict(self, node_ids: Iterable[str], relation_ids: Iterable[str]) -> dict[str, Any]:
        wanted_nodes = set(node_ids)
        wanted_relations = set(relation_ids)
        nodes = [node for node in self.nodes if node.id in wanted_nodes]
        relations = [relation for relation in self.relations if relation.id in wanted_relations]
        return {
            "nodes": [node.to_dict() for node in nodes],
            "relationships": [relation.to_dict() for relation in relations],
        }


def canonical_schema(
    source: str,
    graph: str,
    nodes: Iterable[tuple[str, Any]],
    relations: Iterable[tuple[str, str, str, Any]],
) -> CanonicalSchema:
    """Build a deterministic schema, merging duplicated input definitions."""

    node_properties: dict[str, dict[str, str]] = {}
    relation_properties: dict[tuple[str, str, str], dict[str, str]] = {}

    for raw_label, raw_properties in nodes:
        label = clean_identifier(raw_label)
        if not label:
            continue
        merged = node_properties.setdefault(label, {})
        merged.update(dict(_coerce_properties(raw_properties)))

    for raw_source, raw_type, raw_target, raw_properties in relations:
        source_label = clean_identifier(raw_source)
        relation_type = clean_identifier(raw_type)
        target_label = clean_identifier(raw_target)
        if not source_label or not relation_type or not target_label:
            continue
        node_properties.setdefault(source_label, {})
        node_properties.setdefault(target_label, {})
        merged = relation_properties.setdefault((source_label, relation_type, target_label), {})
        merged.update(dict(_coerce_properties(raw_properties)))

    normalized_nodes = tuple(
        NodeUnit(label, tuple(sorted(properties.items())))
        for label, properties in sorted(node_properties.items())
    )
    normalized_relations = tuple(
        RelationUnit(source_label, relation_type, target_label, tuple(sorted(properties.items())))
        for (source_label, relation_type, target_label), properties in sorted(
            relation_properties.items()
        )
    )
    return CanonicalSchema(
        source=source,
        graph=graph or "unknown",
        nodes=normalized_nodes,
        relations=normalized_relations,
    )


def from_cypherbench(payload: Mapping[str, Any], graph: str) -> CanonicalSchema:
    return canonical_schema(
        "cypherbench",
        graph,
        (
            (entity.get("label", ""), entity.get("properties", {}))
            for entity in payload.get("entities", [])
            if isinstance(entity, Mapping)
        ),
        (
            (
                relation.get("subj_label", ""),
                relation.get("label", ""),
                relation.get("obj_label", ""),
                relation.get("properties", {}),
            )
            for relation in payload.get("relations", [])
            if isinstance(relation, Mapping)
        ),
    )


def from_mind_the_query(payload: Mapping[str, Any], graph: str) -> CanonicalSchema:
    """Normalize Mind-the-Query's graph-keyed JSON schema."""

    body: Mapping[str, Any]
    if graph in payload and isinstance(payload[graph], Mapping):
        body = payload[graph]
    elif len(payload) == 1 and isinstance(next(iter(payload.values())), Mapping):
        body = next(iter(payload.values()))
    else:
        body = payload
    node_props = body.get("node_props", {})
    relation_props = body.get("rel_props", {})
    return canonical_schema(
        "mind_the_query",
        graph,
        (
            (label, properties)
            for label, properties in node_props.items()
            if isinstance(node_props, Mapping)
        ),
        (
            (
                relation.get("start", ""),
                relation.get("type", ""),
                relation.get("end", ""),
                relation_props.get(relation.get("type", ""), [])
                if isinstance(relation_props, Mapping)
                else [],
            )
            for relation in body.get("relationships", [])
            if isinstance(relation, Mapping)
        ),
    )


def _labels_in_node_pattern(body: str) -> list[str]:
    prefix = body.split("{", 1)[0]
    labels: list[str] = []
    for match in re.finditer(r":\s*(`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)", prefix):
        labels.append(clean_identifier(match.group(1)))
    return labels


def _relationship_types(body: str) -> list[str]:
    text = body.split("{", 1)[0]
    colon = text.find(":")
    if colon < 0:
        return []
    remainder = text[colon + 1 :].split("*", 1)[0].strip()
    if remainder.startswith("(") and remainder.endswith(")"):
        remainder = remainder[1:-1].strip()
    types: list[str] = []
    for part in remainder.split("|"):
        candidate = clean_identifier(part.strip())
        if candidate and _IDENTIFIER_RE.fullmatch(candidate):
            types.append(candidate)
    return types


def _inline_properties(text: str) -> list[dict[str, str]]:
    """Parse compact ``name: TYPE`` property declarations used by some sources."""

    properties: list[dict[str, str]] = []
    for item in text.split(","):
        if ":" not in item:
            continue
        raw_name, raw_type = item.split(":", 1)
        name = clean_identifier(raw_name.strip().strip("'\""))
        raw_type = raw_type.strip()
        if raw_type.startswith(("'", '\"')):
            raw_type = "STRING"
        property_type = raw_type.split()[0] if raw_type else "UNKNOWN"
        if name:
            properties.append({"property": name, "datatype": property_type})
    return properties


def _add_inline_groups(text: str, target: dict[str, list[dict[str, str]]]) -> None:
    for match in _INLINE_GROUP_RE.finditer(text):
        label = clean_identifier(match.group("label").strip().strip("'\""))
        if not label:
            continue
        target.setdefault(label, []).extend(_inline_properties(match.group("properties")))


def _add_bare_labels(text: str, target: dict[str, list[dict[str, str]]]) -> None:
    """Read one label per line from compact, property-free schema descriptions."""

    for line in text.splitlines():
        value = line.strip().strip(",").strip("'\"")
        if re.fullmatch(r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*", value):
            target.setdefault(clean_identifier(value), [])


def _schema_from_neo4j_json_object(
    payload: Mapping[str, Any], graph: str
) -> CanonicalSchema | None:
    """Normalize Neo4j schema-inspection JSON embedded as a string in the corpus."""

    node_entries = {
        clean_identifier(label): value
        for label, value in payload.items()
        if isinstance(value, Mapping) and value.get("type") == "node"
    }
    if not node_entries:
        return None

    nodes = [(label, value.get("properties", {})) for label, value in node_entries.items()]
    relations: list[tuple[str, str, str, Any]] = []
    for label, value in node_entries.items():
        relationship_map = value.get("relationships", {})
        if not isinstance(relationship_map, Mapping):
            continue
        for raw_type, details in relationship_map.items():
            if not isinstance(details, Mapping):
                continue
            targets = details.get("labels", [])
            if not isinstance(targets, Sequence) or isinstance(targets, str | bytes):
                continue
            global_relation = payload.get(raw_type, {})
            relation_properties = (
                global_relation.get("properties", {})
                if isinstance(global_relation, Mapping)
                else details.get("properties", {})
            )
            direction = str(details.get("direction", "out")).lower()
            for raw_target in targets:
                target = clean_identifier(raw_target)
                if direction == "in":
                    relations.append((target, clean_identifier(raw_type), label, relation_properties))
                else:
                    relations.append((label, clean_identifier(raw_type), target, relation_properties))
    return canonical_schema("neo4j_text2cypher", graph, nodes, relations)


def from_neo4j_schema_text(schema_text: str, graph: str) -> CanonicalSchema:
    """Parse the textual schema embedded in Neo4j Text2Cypher examples.

    The parser targets the benchmark's documented sections rather than its prose.
    It accepts absent property sections and still preserves node/relation topology.
    """

    stripped = schema_text.strip()
    if stripped.startswith("{"):
        try:
            json_payload = json.loads(stripped)
        except json.JSONDecodeError:
            json_payload = None
        if isinstance(json_payload, Mapping):
            parsed = _schema_from_neo4j_json_object(json_payload, graph)
            if parsed is not None:
                return parsed

    node_properties: dict[str, list[dict[str, str]]] = {}
    relation_properties: dict[str, list[dict[str, str]]] = {}
    parsed_edges: list[tuple[str, str, str, Any]] = []
    section: str | None = None
    current_group: str | None = None

    for line in schema_text.splitlines():
        lower = line.strip().lower()
        if lower.startswith("node properties"):
            section, current_group = "nodes", None
            continue
        if lower.startswith("relationship properties"):
            section, current_group = "relation_properties", None
            continue
        if lower.startswith("the relationships") or lower.startswith("relationships:"):
            section, current_group = "relationships", None
            continue

        if section not in {"nodes", "relation_properties"}:
            continue
        header = _GROUP_HEADER_RE.match(line)
        if header:
            current_group = clean_identifier(
                header.group("bold") or header.group("tick") or header.group("plain")
            )
            target = node_properties if section == "nodes" else relation_properties
            target.setdefault(current_group, [])
            continue
        property_match = _PROPERTY_RE.match(line)
        if property_match and current_group:
            target = node_properties if section == "nodes" else relation_properties
            target[current_group].append(
                {
                    "property": clean_identifier(property_match.group("name")),
                    "datatype": property_match.group("type").split()[0],
                }
            )

    lower_schema = schema_text.lower()
    # Functional-Cypher examples use compact prose such as
    # "Relevant node labels ... Article {abstract: STRING}" followed by
    # ``{'start': Article, 'type': HAS_KEY, 'end': Keyword}`` records.
    if "relevant node labels and their properties" in lower_schema:
        node_heading = re.search(
            r"relevant node labels and their properties(?:\s+\(with datatypes\))?\s+are\s*:",
            schema_text,
            re.I,
        )
        relation_heading = re.search(r"relevant relationships\s+are\s*:", schema_text, re.I)
        node_start = node_heading.end() if node_heading else 0
        node_end = relation_heading.start() if relation_heading else len(schema_text)
        node_text = schema_text[node_start:node_end]
        _add_inline_groups(node_text, node_properties)
        _add_bare_labels(node_text, node_properties)
        for match in _RELEVANT_RELATION_RE.finditer(schema_text):
            relation_type = clean_identifier(match.group("relation_type").strip("'\""))
            relationship_properties = relation_properties.get(relation_type, [])
            source = clean_identifier(match.group("source").strip("'\""))
            target = clean_identifier(match.group("target").strip("'\""))
            if source and relation_type and target:
                parsed_edges.append(
                    (source, relation_type, target, relationship_properties)
                )

    # A second compact schema format puts all node and relationship declarations
    # in one quoted sentence.  Its topology still uses regular Cypher patterns.
    inline_node_heading = re.search(r"node properties are the following\s*:", schema_text, re.I)
    if inline_node_heading:
        relationship_heading = re.search(
            r"relationship properties are the following\s*:", schema_text, re.I
        )
        node_end = relationship_heading.start() if relationship_heading else len(schema_text)
        _add_inline_groups(schema_text[inline_node_heading.end() : node_end], node_properties)
        if relationship_heading:
            endpoint_heading = re.search(r"relationship point", schema_text, re.I)
            relation_end = endpoint_heading.start() if endpoint_heading else len(schema_text)
            _add_inline_groups(
                schema_text[relationship_heading.end() : relation_end], relation_properties
            )

    # Some Neo4j inspection results were serialized through Python's repr rather
    # than JSON. They retain labels and relationship endpoints, but not reliable
    # property datatypes, so topology is normalized and properties are omitted.
    if stripped.startswith("[<Record"):
        for match in _NEO4J_REPR_NODE_RE.finditer(schema_text):
            node_properties.setdefault(clean_identifier(match.group("label")), [])
        for match in _NEO4J_REPR_RELATION_RE.finditer(schema_text):
            parsed_edges.append(
                (
                    clean_identifier(match.group("source")),
                    clean_identifier(match.group("relation_type")),
                    clean_identifier(match.group("target")),
                    [],
                )
            )

    relationships = list(parsed_edges)
    for match in _OVERLAPPING_RELATION_PATTERN_RE.finditer(schema_text):
        left_labels = _labels_in_node_pattern(match.group("left"))
        right_labels = _labels_in_node_pattern(match.group("right"))
        relation_types = _relationship_types(match.group("relation"))
        if not left_labels or not right_labels or not relation_types:
            continue
        # Inline Cypher schemas place ``label: type`` properties inside node
        # patterns, so retain them when they are available.
        for label in left_labels:
            body = match.group("left")
            if "{" in body:
                node_properties.setdefault(label, []).extend(
                    _inline_properties(body.split("{", 1)[1].rsplit("}", 1)[0])
                )
        for label in right_labels:
            body = match.group("right")
            if "{" in body:
                node_properties.setdefault(label, []).extend(
                    _inline_properties(body.split("{", 1)[1].rsplit("}", 1)[0])
                )
        if match.group("left_arrow"):
            source, target = right_labels[0], left_labels[0]
        else:
            source, target = left_labels[0], right_labels[0]
        for relation_type in relation_types:
            relationships.append(
                (source, relation_type, target, relation_properties.get(relation_type, []))
            )

    return canonical_schema(
        "neo4j_text2cypher",
        graph,
        node_properties.items(),
        relationships,
    )
