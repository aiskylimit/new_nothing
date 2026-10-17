from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema_grounding.cypher import extract_subschema
from schema_grounding.datasets import _split_file
from schema_grounding.pipeline import build_dataset
from schema_grounding.schema import (
    canonical_schema,
    from_cypherbench,
    from_mind_the_query,
    from_neo4j_schema_text,
)


class SchemaNormalizationTest(unittest.TestCase):
    def test_cypherbench_adapter(self) -> None:
        schema = from_cypherbench(
            {
                "entities": [{"label": "Person", "properties": {"name": "str"}}],
                "relations": [
                    {
                        "label": "KNOWS",
                        "subj_label": "Person",
                        "obj_label": "Person",
                        "properties": {"since": "int"},
                    }
                ],
            },
            "social",
        )
        self.assertEqual(schema.nodes[0].properties, (("name", "STRING"),))
        self.assertEqual(schema.relations[0].id, "relation:Person|KNOWS|Person")

    def test_cypherbench_adapter_does_not_invent_properties(self) -> None:
        schema = from_cypherbench(
            {
                "entities": [{"label": "Person", "properties": {"id": "int"}}],
                "relations": [],
            },
            "social",
        )
        self.assertEqual(schema.nodes[0].properties, (("id", "INTEGER"),))

    def test_mind_the_query_adapter(self) -> None:
        schema = from_mind_the_query(
            {
                "demo": {
                    "node_props": {"Person": [{"property": "id", "datatype": "STRING"}]},
                    "rel_props": {"VISITS": [{"property": "duration", "datatype": "DURATION"}]},
                    "relationships": [{"start": "Person", "type": "VISITS", "end": "Place"}],
                }
            },
            "demo",
        )
        self.assertEqual({node.label for node in schema.nodes}, {"Person", "Place"})
        self.assertEqual(schema.relations[0].properties, (("duration", "DURATION"),))

    def test_neo4j_text_adapter(self) -> None:
        schema = from_neo4j_schema_text(
            """Node properties:
- **Person**
  - `name`: STRING
- **Movie**
  - `title`: STRING
Relationship properties:
- **ACTED_IN**
  - `role`: STRING
The relationships:
(:Person)-[:ACTED_IN]->(:Movie)
""",
            "movies",
        )
        self.assertEqual(len(schema.nodes), 2)
        self.assertEqual(schema.relations[0].id, "relation:Person|ACTED_IN|Movie")
        self.assertEqual(schema.relations[0].properties, (("role", "STRING"),))

    def test_neo4j_text_adapter_preserves_chained_relationships(self) -> None:
        schema = from_neo4j_schema_text(
            "(:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(:Genre)",
            "movies",
        )

        self.assertEqual(
            {node.label for node in schema.nodes},
            {"Person", "Movie", "Genre"},
        )
        self.assertEqual(
            {relation.id for relation in schema.relations},
            {
                "relation:Person|ACTED_IN|Movie",
                "relation:Movie|IN_GENRE|Genre",
            },
        )

    def test_neo4j_relevant_text_adapter(self) -> None:
        schema = from_neo4j_schema_text(
            """Graph schema: Relevant node labels and their properties (with datatypes) are:
Article {abstract: STRING}
Keyword {}

Relevant relationships are:
{'start': Article, 'type': HAS_KEY, 'end': Keyword }
""",
            "functional",
        )
        self.assertEqual({node.label for node in schema.nodes}, {"Article", "Keyword"})
        self.assertEqual(schema.relations[0].id, "relation:Article|HAS_KEY|Keyword")

    def test_neo4j_relevant_bare_labels_adapter(self) -> None:
        schema = from_neo4j_schema_text(
            """Graph schema: Relevant node labels and their properties are:
Author
DOI
""",
            "functional",
        )
        self.assertEqual({node.label for node in schema.nodes}, {"Author", "DOI"})

    def test_neo4j_repr_adapter(self) -> None:
        schema = from_neo4j_schema_text(
            """[<Record nodes=[<Node labels=frozenset({'author'}) properties={}>,
<Node labels=frozenset({'paper'}) properties={}>]
relationships=[<Relationship nodes=(<Node labels=frozenset({'author'}) properties={}>,
<Node labels=frozenset({'paper'}) properties={}>) type='WROTE' properties={}>]>]""",
            "inspection",
        )
        self.assertEqual({node.label for node in schema.nodes}, {"author", "paper"})
        self.assertEqual(schema.relations[0].id, "relation:author|WROTE|paper")

    def test_neo4j_json_schema_adapter(self) -> None:
        schema = from_neo4j_schema_text(
            '{"Person": {"type": "node", "properties": {"name": {"type": "STRING"}}, '
            '"relationships": {"KNOWS": {"direction": "out", "labels": ["Person"], "properties": {}}}}, '
            '"KNOWS": {"type": "relationship", "properties": {"since": {"type": "INTEGER"}}}}',
            "inspected",
        )
        self.assertEqual(schema.nodes[0].properties, (("name", "STRING"),))
        self.assertEqual(schema.relations[0].properties, (("since", "INTEGER"),))


class SubSchemaExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = canonical_schema(
            "test",
            "graph",
            [("Person", {"name": "STRING"}), ("Movie", {"title": "STRING"}), ("City", {})],
            [
                ("Person", "ACTED_IN", "Movie", {}),
                ("Person", "LIVES_IN", "City", {}),
            ],
        )

    def test_direction_and_variable_resolution(self) -> None:
        result = extract_subschema(
            "MATCH (m:Movie)<-[:ACTED_IN]-(p:Person) RETURN p.name", self.schema
        )
        self.assertTrue(result.complete)
        self.assertEqual(
            result.relation_unit_ids, ("relation:Person|ACTED_IN|Movie",)
        )
        self.assertEqual(set(result.node_unit_ids), {"node:Person", "node:Movie"})

    def test_adjacent_relation_patterns_are_all_extracted(self) -> None:
        result = extract_subschema(
            "MATCH (m:Movie)<-[:ACTED_IN]-(p:Person)-[:LIVES_IN]->(c:City) RETURN p",
            self.schema,
        )
        self.assertTrue(result.complete)
        self.assertEqual(
            set(result.relation_unit_ids),
            {"relation:Person|ACTED_IN|Movie", "relation:Person|LIVES_IN|City"},
        )
        self.assertEqual(
            set(result.node_unit_ids), {"node:Person", "node:Movie", "node:City"}
        )

    def test_anonymous_shorthand_relationships_are_extracted(self) -> None:
        cases = {
            "MATCH (p:Person)-->(m:Movie) RETURN m": "relation:Person|ACTED_IN|Movie",
            "MATCH (m:Movie)<--(p:Person) RETURN m": "relation:Person|ACTED_IN|Movie",
            "MATCH (p:Person)--(m:Movie) RETURN m": "relation:Person|ACTED_IN|Movie",
        }
        for query, expected_relation in cases.items():
            with self.subTest(query=query):
                result = extract_subschema(query, self.schema)
                self.assertTrue(result.complete)
                self.assertEqual(result.relation_unit_ids, (expected_relation,))
                self.assertEqual(set(result.node_unit_ids), {"node:Person", "node:Movie"})

    def test_adjacent_anonymous_shorthand_relationships_are_all_extracted(self) -> None:
        result = extract_subschema(
            "MATCH (m:Movie)<--(p:Person)-->(c:City) RETURN p",
            self.schema,
        )
        self.assertTrue(result.complete)
        self.assertEqual(
            set(result.relation_unit_ids),
            {"relation:Person|ACTED_IN|Movie", "relation:Person|LIVES_IN|City"},
        )

    def test_anonymous_shorthand_reports_wrong_direction(self) -> None:
        result = extract_subschema(
            "MATCH (p:Person)<--(m:Movie) RETURN m",
            self.schema,
        )
        self.assertFalse(result.complete)
        self.assertEqual(result.relation_unit_ids, ())
        self.assertEqual(
            result.unmatched_relationship_patterns,
            ("(:Movie)-[:?]->(:Person)",),
        )

    def test_anonymous_shorthand_keeps_all_matching_relations(self) -> None:
        ambiguous_schema = canonical_schema(
            "test",
            "graph",
            [("Person", {}), ("Movie", {})],
            [
                ("Person", "ACTED_IN", "Movie", {}),
                ("Person", "DIRECTED", "Movie", {}),
            ],
        )
        result = extract_subschema(
            "MATCH (p:Person)-->(m:Movie) RETURN m",
            ambiguous_schema,
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.ambiguous_relation_patterns, 1)
        self.assertEqual(
            set(result.relation_unit_ids),
            {"relation:Person|ACTED_IN|Movie", "relation:Person|DIRECTED|Movie"},
        )

    def test_typed_relationship_without_endpoints_keeps_all_matching_relations(self) -> None:
        schema = canonical_schema(
            "test",
            "graph",
            [("Person", {}), ("Movie", {}), ("Actor", {}), ("Film", {})],
            [
                ("Person", "ACTED_IN", "Movie", {}),
                ("Actor", "ACTED_IN", "Film", {}),
            ],
        )
        result = extract_subschema("MATCH ()-[:ACTED_IN]->() RETURN count(*)", schema)

        self.assertTrue(result.complete)
        self.assertEqual(result.ambiguous_relation_patterns, 1)
        self.assertEqual(
            set(result.relation_unit_ids),
            {
                "relation:Person|ACTED_IN|Movie",
                "relation:Actor|ACTED_IN|Film",
            },
        )
        self.assertEqual(
            set(result.node_unit_ids),
            {"node:Person", "node:Movie", "node:Actor", "node:Film"},
        )

    def test_label_predicate_resolves_unlabelled_node(self) -> None:
        result = extract_subschema(
            "MATCH (p) WHERE p:Person RETURN p.name", self.schema
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.node_unit_ids, ("node:Person",))


    def test_unknown_relation_is_reported(self) -> None:
        result = extract_subschema(
            "MATCH (p:Person)-[:DIRECTED]->(m:Movie) RETURN m", self.schema
        )
        self.assertIn("DIRECTED", result.unmapped_relation_types)
        self.assertFalse(result.complete)

    def test_relation_endpoint_mismatch_is_reported_separately(self) -> None:
        result = extract_subschema(
            "MATCH (m:Movie)-[:ACTED_IN]->(p:Person) RETURN p", self.schema
        )
        self.assertFalse(result.unmapped_relation_types)
        self.assertEqual(
            result.unmatched_relationship_patterns,
            ("(:Movie)-[:ACTED_IN]->(:Person)",),
        )

    def test_case_expression_is_not_a_node_pattern(self) -> None:
        result = extract_subschema(
            "MATCH (p:Person) WITH p, (CASE WHEN 1 > 0 THEN 1 ELSE 0 END) AS score RETURN score",
            self.schema,
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.node_unit_ids, ("node:Person",))

    def test_relation_propagates_variable_label(self) -> None:
        result = extract_subschema(
            "MATCH (person)-[:LIVES_IN]->(:City) RETURN person", self.schema
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.relation_unit_ids, ("relation:Person|LIVES_IN|City",))
        self.assertEqual(set(result.node_unit_ids), {"node:Person", "node:City"})

    def test_parentheses_inside_string_literal_are_not_patterns(self) -> None:
        result = extract_subschema(
            "MATCH (p:Person {name: 'Example (alias)'}) RETURN p", self.schema
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.node_unit_ids, ("node:Person",))

    def test_quote_inside_backticked_identifier_does_not_mask_later_patterns(self) -> None:
        result = extract_subschema(
            "MATCH (s:City) WHERE s.`owner's name` = 'x' "
            "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN m",
            self.schema,
        )

        self.assertTrue(result.complete)
        self.assertEqual(
            set(result.node_unit_ids),
            {"node:City", "node:Person", "node:Movie"},
        )
        self.assertEqual(result.relation_unit_ids, ("relation:Person|ACTED_IN|Movie",))

    def test_line_comment_quote_does_not_mask_following_patterns(self) -> None:
        result = extract_subschema(
            "MATCH (p:Person) // find the person's movies\n"
            "MATCH (p)-[:ACTED_IN]->(m:Movie) RETURN m",
            self.schema,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.relation_unit_ids, ("relation:Person|ACTED_IN|Movie",))
        self.assertEqual(set(result.node_unit_ids), {"node:Person", "node:Movie"})

    def test_comments_cannot_inject_fake_schema_patterns(self) -> None:
        result = extract_subschema(
            "MATCH (p:Person) /* ' (:City)<-[:LIVES_IN]-(:Person) */ RETURN p",
            self.schema,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.node_unit_ids, ("node:Person",))
        self.assertEqual(result.relation_unit_ids, ())

    def test_standalone_wildcard_node_selects_all_node_units(self) -> None:
        result = extract_subschema("MATCH (n) RETURN n", self.schema)

        self.assertTrue(result.complete)
        self.assertEqual(
            set(result.node_unit_ids),
            {"node:Person", "node:Movie", "node:City"},
        )
        self.assertEqual(result.relation_unit_ids, ())
        self.assertEqual(result.unresolved_node_patterns, 1)

    def test_wildcard_relationship_pattern_selects_all_matching_units(self) -> None:
        result = extract_subschema("MATCH (n)-[r]->(m) RETURN n, r, m", self.schema)

        self.assertTrue(result.complete)
        self.assertEqual(
            set(result.node_unit_ids),
            {"node:Person", "node:Movie", "node:City"},
        )
        self.assertEqual(
            set(result.relation_unit_ids),
            {"relation:Person|ACTED_IN|Movie", "relation:Person|LIVES_IN|City"},
        )

    def test_parenthesized_arithmetic_is_not_a_node_pattern(self) -> None:
        result = extract_subschema(
            "MATCH (p:Person) RETURN AVG((p.age + 2) / 2.0)",
            self.schema,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.node_unit_ids, ("node:Person",))

    def test_grouped_relationship_types_are_parsed_without_fake_nodes(self) -> None:
        result = extract_subschema(
            "MATCH (p:Person)-[r:(ACTED_IN|LIVES_IN)]->(target) RETURN target",
            self.schema,
        )

        self.assertTrue(result.complete)
        self.assertEqual(
            set(result.relation_unit_ids),
            {"relation:Person|ACTED_IN|Movie", "relation:Person|LIVES_IN|City"},
        )
        self.assertEqual(
            set(result.node_unit_ids),
            {"node:Person", "node:Movie", "node:City"},
        )


class PipelineAuditTest(unittest.TestCase):
    def test_shorthand_relationship_is_preserved_in_generation_and_selection_data(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "benchmarks"
            source_directory = root / "Cypherbench"
            (source_directory / "graphs" / "schemas").mkdir(parents=True)
            (source_directory / "graphs" / "schemas" / "demo_schema.json").write_text(
                json.dumps(
                    {
                        "entities": [
                            {"label": "Person", "properties": {}},
                            {"label": "Movie", "properties": {}},
                        ],
                        "relations": [
                            {
                                "label": "ACTED_IN",
                                "subj_label": "Person",
                                "obj_label": "Movie",
                                "properties": {},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (source_directory / "train.jsonl").write_text(
                json.dumps(
                    {
                        "qid": "shorthand-relation",
                        "graph": "demo",
                        "nl_question": "List movies with a person connection.",
                        "gold_cypher": "MATCH (p:Person)-->(m:Movie) RETURN m",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_directory = Path(temporary_directory) / "output"
            build_dataset(
                benchmarks_root=root,
                output_dir=output_directory,
                sources=("cypherbench",),
            )

            generation = json.loads(
                (output_directory / "generation_train.jsonl").read_text(encoding="utf-8")
            )
            selection = [
                json.loads(line)
                for line in (output_directory / "selection_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                generation["sub_schema"]["relationships"],
                [{"properties": {}, "source": "Person", "target": "Movie", "type": "ACTED_IN"}],
            )
            relationship_rows = [row for row in selection if row["unit_type"] == "relation"]
            self.assertEqual(len(relationship_rows), 1)
            self.assertEqual(relationship_rows[0]["label"], 1)

    def test_multi_match_shorthand_relationship_is_kept_in_strict_mode(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "benchmarks"
            source_directory = root / "Cypherbench"
            (source_directory / "graphs" / "schemas").mkdir(parents=True)
            (source_directory / "graphs" / "schemas" / "demo_schema.json").write_text(
                json.dumps(
                    {
                        "entities": [
                            {"label": "Person", "properties": {}},
                            {"label": "Movie", "properties": {}},
                        ],
                        "relations": [
                            {
                                "label": relation_type,
                                "subj_label": "Person",
                                "obj_label": "Movie",
                                "properties": {},
                            }
                            for relation_type in ("ACTED_IN", "DIRECTED")
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (source_directory / "train.jsonl").write_text(
                json.dumps(
                    {
                        "qid": "ambiguous-shorthand",
                        "graph": "demo",
                        "nl_question": "List connected movies.",
                        "gold_cypher": "MATCH (p:Person)-->(m:Movie) RETURN m",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_directory = Path(temporary_directory) / "output"
            manifest = build_dataset(
                benchmarks_root=root,
                output_dir=output_directory,
                sources=("cypherbench",),
            )

            generation = json.loads(
                (output_directory / "generation_train.jsonl").read_text(encoding="utf-8")
            )
            selection = [
                json.loads(line)
                for line in (output_directory / "selection_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                {relation["type"] for relation in generation["sub_schema"]["relationships"]},
                {"ACTED_IN", "DIRECTED"},
            )
            relation_rows = [row for row in selection if row["unit_type"] == "relation"]
            self.assertEqual([row["label"] for row in relation_rows], [1, 1])
            self.assertEqual(
                (output_directory / "rejected_train.jsonl").read_text(encoding="utf-8"),
                "",
            )
            counts = manifest["counts"]["cypherbench/train"]
            self.assertEqual(counts.get("records_rejected", 0), 0)
            self.assertEqual(counts["ambiguous_relation_patterns"], 1)

    def test_test_inference_keeps_rejected_examples_without_unreliable_gold_labels(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "benchmarks"
            source_directory = root / "Cypherbench"
            (source_directory / "graphs" / "schemas").mkdir(parents=True)
            (source_directory / "graphs" / "schemas" / "demo_schema.json").write_text(
                json.dumps(
                    {
                        "entities": [
                            {"label": "Person", "properties": {}},
                            {"label": "Movie", "properties": {}},
                        ],
                        "relations": [
                            {
                                "label": relation_type,
                                "subj_label": "Person",
                                "obj_label": "Movie",
                                "properties": {},
                            }
                            for relation_type in ("ACTED_IN", "DIRECTED")
                        ],
                    }
                ),
                encoding="utf-8",
            )
            examples = [
                {
                    "qid": "complete",
                    "graph": "demo",
                    "nl_question": "List acted-in movies.",
                    "gold_cypher": "MATCH (:Person)-[:ACTED_IN]->(m:Movie) RETURN m",
                },
                {
                    "qid": "unmatched",
                    "graph": "demo",
                    "nl_question": "List people reached from movies.",
                    "gold_cypher": "MATCH (:Movie)-[:ACTED_IN]->(p:Person) RETURN p",
                },
            ]
            (source_directory / "test.jsonl").write_text(
                "".join(json.dumps(example) + "\n" for example in examples),
                encoding="utf-8",
            )

            output_directory = Path(temporary_directory) / "output"
            manifest = build_dataset(
                benchmarks_root=root,
                output_dir=output_directory,
                sources=("cypherbench",),
                splits=("test",),
            )

            strict_generation = [
                json.loads(line)
                for line in (output_directory / "generation_test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            inference_generation = [
                json.loads(line)
                for line in (output_directory / "generation_inference_test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            inference_selection = [
                json.loads(line)
                for line in (output_directory / "selection_inference_test.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual([row["id"] for row in strict_generation], ["cypherbench:test:demo:complete"])
            self.assertEqual(len(inference_generation), 2)
            self.assertTrue(inference_generation[0]["gold_subschema_available"])
            self.assertIn("sub_schema", inference_generation[0])
            self.assertFalse(inference_generation[1]["gold_subschema_available"])
            self.assertNotIn("sub_schema", inference_generation[1])
            by_example: dict[str, list[dict[str, object]]] = {}
            for row in inference_selection:
                by_example.setdefault(str(row["example_id"]), []).append(row)
            self.assertTrue(all("label" in row for row in by_example["cypherbench:test:demo:complete"]))
            self.assertTrue(all("label" not in row for row in by_example["cypherbench:test:demo:unmatched"]))
            counts = manifest["counts"]["cypherbench/test"]
            self.assertEqual(counts["generation_examples"], 1)
            self.assertEqual(counts["inference_examples"], 2)
            self.assertEqual(counts["inference_labeled_examples"], 1)
            self.assertEqual(counts["inference_unlabeled_examples"], 1)

    def test_unrecognized_schema_is_written_to_normalization_audit(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "benchmarks"
            source_directory = root / "Neo4j_Text2Cypher"
            source_directory.mkdir(parents=True)
            (source_directory / "train.json").write_text(
                json.dumps(
                    [
                        {
                            "qid": "invalid-schema",
                            "graph": "demo",
                            "nl_question": "List every item.",
                            "gold_cypher": "MATCH (item) RETURN item",
                            "schema": "CREATE TABLE items (id INTEGER);",
                        },
                        {
                            "qid": "valid-schema",
                            "graph": "valid-demo",
                            "nl_question": "List every person.",
                            "gold_cypher": "MATCH (person:Person) RETURN person",
                            "schema": "Node properties:\n- **Person**\n  - `name`: STRING\n",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output_directory = Path(temporary_directory) / "output"
            manifest = build_dataset(
                benchmarks_root=root,
                output_dir=output_directory,
                sources=("neo4j_text2cypher",),
            )

            issue = json.loads(
                (output_directory / "normalization_issues_train.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            rejected = json.loads(
                (output_directory / "rejected_train.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(issue["issues"], ["empty_normalized_schema"])
            self.assertEqual(issue["raw_schema"], "CREATE TABLE items (id INTEGER);")
            self.assertEqual(rejected["reason"], "empty_normalized_schema")
            self.assertEqual(
                manifest["files"]["normalization_issues"]["train"],
                "normalization_issues_train.jsonl",
            )

    def test_structured_sources_do_not_fall_back_to_prompt_export_dev_jsonl(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for source_name in ("Mind_the_query", "Neo4j_Text2Cypher"):
                with self.subTest(source=source_name):
                    source_directory = root / source_name
                    source_directory.mkdir()
                    (source_directory / "dev.jsonl").write_text("{}\n", encoding="utf-8")

                    with self.assertRaises(FileNotFoundError):
                        _split_file(root, source_name, "dev")

    def test_all_rejected_source_split_fails_loudly(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "benchmarks"
            source_directory = root / "Cypherbench"
            source_directory.mkdir(parents=True)
            (source_directory / "train.jsonl").write_text(
                json.dumps({"qid": "empty", "graph": "demo"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "produced no usable generation examples"):
                build_dataset(
                    benchmarks_root=root,
                    output_dir=Path(temporary_directory) / "output",
                    sources=("cypherbench",),
                )

    def test_dev_split_build(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "benchmarks"
            source_directory = root / "Cypherbench"
            source_directory.mkdir(parents=True)
            (source_directory / "graphs" / "schemas").mkdir(parents=True)
            (source_directory / "graphs" / "schemas" / "demo_schema.json").write_text(
                json.dumps({"entities": [{"label": "Person", "properties": {}}], "relations": []}),
                encoding="utf-8",
            )
            (source_directory / "dev.jsonl").write_text(
                json.dumps(
                    {
                        "qid": "dev-sample",
                        "graph": "demo",
                        "nl_question": "List all people.",
                        "gold_cypher": "MATCH (p:Person) RETURN p",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_directory = Path(temporary_directory) / "output"
            manifest = build_dataset(
                benchmarks_root=root,
                output_dir=output_directory,
                sources=("cypherbench",),
                splits=("dev",),
            )
            self.assertEqual(manifest["counts"]["cypherbench/dev"]["generation_examples"], 1)
            self.assertTrue((output_directory / "generation_dev.jsonl").exists())
            self.assertTrue((output_directory / "selection_dev.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
