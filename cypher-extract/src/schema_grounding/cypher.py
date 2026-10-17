"""Schema-guided extraction of gold sub-schemas from Cypher queries.

This is deliberately not a full Cypher grammar. It recognizes node and
relationship patterns, resolves variables across the query, and then maps the
observed patterns back to canonical schema units. Broad patterns retain every
matching schema unit, while genuinely unmapped patterns are surfaced so strict
dataset builds cannot silently produce incomplete supervision.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .schema import CanonicalSchema, RelationUnit, clean_identifier

_NODE_PATTERN_RE = re.compile(r"\((?P<body>[^()]*)\)")
_RELATION_PATTERN_RE = re.compile(
    r"\((?P<left>[^()]*)\)\s*(?:(?P<left_arrow><-)|-)\s*"
    r"(?:\[(?P<relation>[^\]]*)\])?\s*(?:(?P<right_arrow>->)|-)\s*\((?P<right>[^()]*)\)"
)
# Relation patterns can share a node in a chain, for example
# ``(a)-[:FIRST]->(b)-[:SECOND]->(c)``.  A regular ``finditer`` over the
# pattern above consumes ``(b)`` while matching the first relation and would
# therefore skip the second.  Look ahead so every opening node can start a
# match while retaining the named capture groups used by the parser.
_OVERLAPPING_RELATION_PATTERN_RE = re.compile(r"(?=" + _RELATION_PATTERN_RE.pattern + r")")
_IDENTIFIER = r"`[^`]+`|[A-Za-z_][A-Za-z0-9_]*"
_VARIABLE_RE = re.compile(rf"^\s*(?P<variable>{_IDENTIFIER})\s*(?::|\{{|$)")
_LABEL_RE = re.compile(rf":\s*(?P<label>{_IDENTIFIER})")
_NODE_PREFIX_RE = re.compile(
    rf"^\s*(?:{_IDENTIFIER})?(?:\s*:\s*(?:{_IDENTIFIER}))*\s*$"
)
_NODE_CONTEXT_KEYWORDS = {
    "MATCH",
    "MERGE",
    "CREATE",
    "EXISTS",
}
_CLAUSE_RE = re.compile(
    r"\b(?:OPTIONAL\s+MATCH|MATCH|MERGE|CREATE|WITH|RETURN|WHERE|UNWIND|CALL|FOREACH|SET|DELETE|ORDER\s+BY|LIMIT|SKIP)\b",
    re.I,
)


@dataclass(frozen=True)
class NodePattern:
    start: int
    end: int
    variable: str | None
    labels: tuple[str, ...]


@dataclass(frozen=True)
class RelationPattern:
    start: int
    end: int
    left: NodePattern
    right: NodePattern
    relation_types: tuple[str, ...]
    direction: str  # out, in, or undirected relative to the written left node


@dataclass(frozen=True)
class SubSchemaExtraction:
    node_unit_ids: tuple[str, ...]
    relation_unit_ids: tuple[str, ...]
    unmapped_node_labels: tuple[str, ...]
    unmapped_relation_types: tuple[str, ...]
    unmatched_relationship_patterns: tuple[str, ...]
    unresolved_node_patterns: int
    ambiguous_relation_patterns: int

    @property
    def complete(self) -> bool:
        # Multiple matching relation units are not a coverage failure: a Cypher
        # pattern whose type or endpoints are broad can traverse every candidate.
        # All candidates are retained; the ambiguity count remains audit metadata.
        return not (
            self.unmapped_node_labels
            or self.unmapped_relation_types
            or self.unmatched_relationship_patterns
        )

    @property
    def has_units(self) -> bool:
        return bool(self.node_unit_ids or self.relation_unit_ids)

    def diagnostics(self) -> dict[str, object]:
        return {
            "unmapped_node_labels": list(self.unmapped_node_labels),
            "unmapped_relation_types": list(self.unmapped_relation_types),
            "unmatched_relationship_patterns": list(self.unmatched_relationship_patterns),
            "unresolved_node_patterns": self.unresolved_node_patterns,
            "ambiguous_relation_patterns": self.ambiguous_relation_patterns,
        }


def _previous_word(query: str, position: int) -> str | None:
    prefix = query[:position].rstrip()
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", prefix)
    return match.group(1).upper() if match else None


def _mask_quoted_contents(query: str) -> str:
    """Blank quoted literal contents while preserving character offsets.

    Cypher properties may contain parentheses, brackets, colons, or strings
    resembling labels. Pattern extraction operates on the masked form so those
    characters cannot be mistaken for graph syntax; offsets still point into the
    original query for diagnostics.
    """

    characters = list(query)
    quote: str | None = None
    in_backtick = False
    escaped = False
    index = 0
    while index < len(characters):
        character = characters[index]
        if quote is None:
            if in_backtick:
                if character == "`":
                    if index + 1 < len(characters) and characters[index + 1] == "`":
                        index += 2
                        continue
                    in_backtick = False
            elif character == "`":
                in_backtick = True
            elif character in {"'", '\"'}:
                quote = character
            index += 1
            continue
        if escaped:
            characters[index] = " "
            escaped = False
        elif character == "\\":
            characters[index] = " "
            escaped = True
        elif character == quote:
            quote = None
        else:
            characters[index] = " "
        index += 1
    return "".join(characters)


def _mask_comments(query: str) -> str:
    """Blank Cypher comments before quote masking while retaining offsets."""

    characters = list(query)
    quote: str | None = None
    in_backtick = False
    escaped = False
    index = 0

    def blank(start: int, end: int) -> None:
        for position in range(start, end):
            if characters[position] not in "\r\n":
                characters[position] = " "

    while index < len(query):
        character = query[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            index += 1
            continue
        if in_backtick:
            if character == "`":
                if index + 1 < len(query) and query[index + 1] == "`":
                    index += 2
                    continue
                in_backtick = False
            index += 1
            continue
        if query.startswith("//", index):
            line_end = query.find("\n", index + 2)
            line_end = len(query) if line_end < 0 else line_end
            blank(index, line_end)
            index = line_end
            continue
        if query.startswith("/*", index):
            close = query.find("*/", index + 2)
            comment_end = len(query) if close < 0 else close + 2
            blank(index, comment_end)
            index = comment_end
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "`":
            in_backtick = True
        index += 1
    return "".join(characters)


def _mask_string_literals(query: str) -> str:
    """Blank literals and comments without changing query offsets."""

    return _mask_quoted_contents(_mask_comments(query))


def _is_inside_square_brackets(query: str, position: int) -> bool:
    depth = 0
    for character in query[:position]:
        if character == "[":
            depth += 1
        elif character == "]" and depth:
            depth -= 1
    return depth > 0


def _is_nested_parenthesis(query: str, position: int) -> bool:
    depth = 0
    for character in query[:position]:
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
    return depth > 0


def _is_likely_node_pattern(query: str, start: int, body: str) -> bool:
    """Avoid treating function calls such as ``count(x)`` as node patterns."""

    prefix = body.split("{", 1)[0]
    if not _NODE_PREFIX_RE.fullmatch(prefix):
        return False
    if _is_inside_square_brackets(query, start) or _is_nested_parenthesis(query, start):
        return False
    if start == 0:
        return True
    previous = query[start - 1]
    if previous in "-<,{}[":
        return True
    if previous.isspace():
        word = _previous_word(query, start)
        if word is not None:
            return word in _NODE_CONTEXT_KEYWORDS
        prefix = query[:start]
        clauses = list(_CLAUSE_RE.finditer(prefix))
        last_clause = clauses[-1] if clauses else None
        # A graph pattern may appear inside WHERE EXISTS { ... }, including
        # after punctuation/whitespace. Other expression clauses (WITH,
        # RETURN, ...) should not turn parentheses into fake node patterns.
        last_exists = prefix.upper().rfind("EXISTS")
        if last_exists >= 0 and (last_clause is None or last_exists >= last_clause.start()):
            return True
        return bool(last_clause and last_clause.group(0).upper().endswith("MATCH")) or bool(
            last_clause and last_clause.group(0).upper() in {"MERGE", "CREATE"}
        )
    return not (previous.isalnum() or previous in "_.`")


def _parse_node_pattern(body: str, start: int, end: int) -> NodePattern:
    prefix = body.split("{", 1)[0]
    labels = tuple(clean_identifier(match.group("label")) for match in _LABEL_RE.finditer(prefix))
    variable_match = _VARIABLE_RE.match(prefix)
    variable = clean_identifier(variable_match.group("variable")) if variable_match else None
    return NodePattern(start=start, end=end, variable=variable, labels=labels)


def _parse_relation_types(body: str) -> tuple[str, ...]:
    prefix = body.split("{", 1)[0]
    colon = prefix.find(":")
    if colon < 0:
        return ()
    type_part = prefix[colon + 1 :].split("*", 1)[0].strip()
    if type_part.startswith("(") and type_part.endswith(")"):
        type_part = type_part[1:-1].strip()
    values: list[str] = []
    for raw_type in type_part.split("|"):
        candidate = clean_identifier(raw_type.strip())
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
            values.append(candidate)
    return tuple(values)


def _find_node_patterns(query: str) -> list[NodePattern]:
    patterns: list[NodePattern] = []
    masked_query = _mask_string_literals(query)
    for match in _NODE_PATTERN_RE.finditer(masked_query):
        if _is_likely_node_pattern(
            masked_query, match.start(), match.group("body")
        ):
            patterns.append(_parse_node_pattern(match.group("body"), match.start(), match.end()))
    return patterns


def _find_relation_patterns(query: str) -> list[RelationPattern]:
    patterns: list[RelationPattern] = []
    masked_query = _mask_string_literals(query)
    for match in _OVERLAPPING_RELATION_PATTERN_RE.finditer(masked_query):
        direction = "in" if match.group("left_arrow") else "out" if match.group("right_arrow") else "undirected"
        patterns.append(
            RelationPattern(
                start=match.start(),
                end=match.end(),
                left=_parse_node_pattern(
                    match.group("left"), match.start("left") - 1, match.end("left") + 1
                ),
                right=_parse_node_pattern(
                    match.group("right"), match.start("right") - 1, match.end("right") + 1
                ),
                relation_types=_parse_relation_types(match.group("relation") or ""),
                direction=direction,
            )
        )
    return patterns


def _variable_labels_from_predicates(
    query: str, schema_labels: Iterable[str]
) -> dict[str, set[str]]:
    """Recognize Cypher label predicates such as ``WHERE n:Person``."""

    result: dict[str, set[str]] = defaultdict(set)
    masked_query = _mask_string_literals(query)
    for label in schema_labels:
        escaped = re.escape(label)
        pattern = re.compile(rf"(?<![:A-Za-z0-9_`])(?P<var>{_IDENTIFIER})\s*:\s*`?{escaped}`?(?![A-Za-z0-9_`])")
        for match in pattern.finditer(masked_query):
            result[clean_identifier(match.group("var"))].add(label)
    return result


def _matching_relations(
    schema: CanonicalSchema,
    relation_type: str | None,
    left_labels: set[str],
    right_labels: set[str],
    direction: str,
) -> list[RelationUnit]:
    candidates = [
        relation
        for relation in schema.relations
        if relation_type is None or relation.relation_type == relation_type
    ]
    if direction == "in":
        source_labels, target_labels = right_labels, left_labels
        return [
            relation
            for relation in candidates
            if (not source_labels or relation.source in source_labels)
            and (not target_labels or relation.target in target_labels)
        ]
    if direction == "out":
        return [
            relation
            for relation in candidates
            if (not left_labels or relation.source in left_labels)
            and (not right_labels or relation.target in right_labels)
        ]
    return [
        relation
        for relation in candidates
        if (
            (not left_labels or relation.source in left_labels)
            and (not right_labels or relation.target in right_labels)
        )
        or (
            (not left_labels or relation.target in left_labels)
            and (not right_labels or relation.source in right_labels)
        )
    ]


def _describe_relationship_pattern(
    relation_type: str | None,
    left_labels: set[str],
    right_labels: set[str],
    direction: str,
) -> str:
    """Render an unresolved query pattern with its known endpoint labels."""

    def node(labels: set[str]) -> str:
        return ":" + ":".join(sorted(labels)) if labels else "?"

    left, right = node(left_labels), node(right_labels)
    relationship = relation_type or "?"
    if direction == "in":
        return f"({right})-[:{relationship}]->({left})"
    if direction == "out":
        return f"({left})-[:{relationship}]->({right})"
    return f"({left})-[:{relationship}]-({right})"


def _propagate_variable_labels(
    relation_patterns: Iterable[RelationPattern],
    schema: CanonicalSchema,
    variable_labels: dict[str, set[str]],
) -> None:
    """Infer an unlabelled variable's type from uniquely resolved relations.

    A query can introduce a variable without a label in one MATCH clause and
    constrain it in a later clause, e.g. ``(stream)-[:HAS_LANGUAGE]->(:Language)``.
    Repeating this propagation lets earlier patterns use that constraint too.
    """

    patterns = tuple(relation_patterns)
    for _ in range(len(patterns) + 1):
        changed = False
        for pattern in patterns:
            if len(pattern.relation_types) != 1 or pattern.direction == "undirected":
                continue
            left_labels = set(pattern.left.labels)
            right_labels = set(pattern.right.labels)
            if pattern.left.variable:
                left_labels.update(variable_labels[pattern.left.variable])
            if pattern.right.variable:
                right_labels.update(variable_labels[pattern.right.variable])
            candidates = _matching_relations(
                schema,
                pattern.relation_types[0],
                left_labels,
                right_labels,
                pattern.direction,
            )
            if len(candidates) != 1:
                continue
            relation = candidates[0]
            if pattern.direction == "out":
                endpoints = ((pattern.left.variable, relation.source), (pattern.right.variable, relation.target))
            else:
                endpoints = ((pattern.left.variable, relation.target), (pattern.right.variable, relation.source))
            for variable, label in endpoints:
                if variable and label not in variable_labels[variable]:
                    variable_labels[variable].add(label)
                    changed = True
        if not changed:
            return


def extract_subschema(query: str, schema: CanonicalSchema) -> SubSchemaExtraction:
    """Map graph units explicitly used by a gold Cypher query to schema unit IDs."""

    node_patterns = _find_node_patterns(query)
    relation_patterns = _find_relation_patterns(query)
    schema_nodes = schema.node_by_label
    variable_labels: dict[str, set[str]] = defaultdict(set)

    for pattern in node_patterns:
        if pattern.variable:
            variable_labels[pattern.variable].update(pattern.labels)
    for variable, labels in _variable_labels_from_predicates(query, schema_nodes).items():
        variable_labels[variable].update(labels)
    _propagate_variable_labels(relation_patterns, schema, variable_labels)

    node_ids: set[str] = set()
    relation_ids: set[str] = set()
    unmapped_nodes: set[str] = set()
    unmapped_relation_types: set[str] = set()
    unmatched_relationship_patterns: set[str] = set()

    for pattern in node_patterns:
        labels = set(pattern.labels)
        if pattern.variable:
            labels.update(variable_labels[pattern.variable])
        for label in labels:
            node = schema_nodes.get(label)
            if node:
                node_ids.add(node.id)
            else:
                unmapped_nodes.add(label)

    relationship_node_spans = {
        (pattern.left.start, pattern.left.end) for pattern in relation_patterns
    } | {(pattern.right.start, pattern.right.end) for pattern in relation_patterns}
    unresolved_node_patterns = 0
    for pattern in node_patterns:
        labels = set(pattern.labels)
        if pattern.variable:
            labels.update(variable_labels[pattern.variable])
        if not labels and (pattern.start, pattern.end) not in relationship_node_spans:
            # A standalone unlabeled node is a real wildcard graph pattern.
            # Preserve its broad semantics by selecting every canonical node,
            # while retaining an audit count of broad wildcard expansions.
            unresolved_node_patterns += 1
            node_ids.update(node.id for node in schema.nodes)

    ambiguous_relation_patterns = 0
    for pattern in relation_patterns:
        left_labels = set(pattern.left.labels)
        right_labels = set(pattern.right.labels)
        if pattern.left.variable:
            left_labels.update(variable_labels[pattern.left.variable])
        if pattern.right.variable:
            right_labels.update(variable_labels[pattern.right.variable])

        relation_types = pattern.relation_types or (None,)
        for relation_type in relation_types:
            candidates = _matching_relations(
                schema,
                relation_type,
                left_labels,
                right_labels,
                pattern.direction,
            )
            if relation_type is not None and not any(
                relation.relation_type == relation_type for relation in schema.relations
            ):
                unmapped_relation_types.add(relation_type)
            if not candidates:
                # A known type can still be invalid for the query's endpoints
                # or direction. Keep that distinct from a type absent from the
                # schema so corpus audits identify the real data issue.
                unmatched_relationship_patterns.add(
                    _describe_relationship_pattern(
                        relation_type, left_labels, right_labels, pattern.direction
                    )
                )
                continue
            if len(candidates) != 1:
                ambiguous_relation_patterns += 1
            for relation in candidates:
                relation_ids.add(relation.id)
                node_ids.add(f"node:{relation.source}")
                node_ids.add(f"node:{relation.target}")

    return SubSchemaExtraction(
        node_unit_ids=tuple(sorted(node_ids)),
        relation_unit_ids=tuple(sorted(relation_ids)),
        unmapped_node_labels=tuple(sorted(unmapped_nodes)),
        unmapped_relation_types=tuple(sorted(unmapped_relation_types)),
        unmatched_relationship_patterns=tuple(sorted(unmatched_relationship_patterns)),
        unresolved_node_patterns=unresolved_node_patterns,
        ambiguous_relation_patterns=ambiguous_relation_patterns,
    )
