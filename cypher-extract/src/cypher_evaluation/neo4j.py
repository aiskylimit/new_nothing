from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any, Literal

logger = logging.getLogger(__name__)


class Neo4jConnector:
    """Small lifecycle-safe wrapper around the official Neo4j driver."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        *,
        database: str = "neo4j",
        name: str = "neo4j-db",
        max_connection_pool_size: int = 100,
        debug: bool = False,
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as error:
            raise RuntimeError("Install the project with the 'evaluation' extra to use Neo4j evaluation") from error
        self.database = database
        self.name = name
        self.debug = debug
        logger.info("Connecting to Neo4j at: %s", uri)
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_pool_size=max_connection_pool_size,
        )

    def __enter__(self) -> Neo4jConnector:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self, *, timeout: int = 10) -> None:
        """Verify both the server and the selected database before scoring."""
        self.driver.verify_connectivity()
        self.run_query("RETURN 1 AS ok", timeout=timeout)

    def run_query(
        self,
        cypher: str,
        *,
        timeout: int | None = None,
        convert: Literal["data", "graph"] = "data",
        **parameters: Any,
    ) -> Any:
        from neo4j import Query

        started = time.time()
        if self.debug:
            logger.info("Running Cypher:\n%s", cypher)

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(Query(cypher, timeout=timeout), **parameters)
                if convert == "data":
                    output = result.data()
                elif convert == "graph":
                    output = result.graph()
                else:
                    raise ValueError(f"Unsupported result conversion: {convert!r}")
        except Exception:
            logger.error("ERROR when executing Cypher: %s", cypher)
            raise
        if self.debug:
            logger.info("Query finished in %.2fs", time.time() - started)
        return output

    def iter_query(self, cypher: str, *, timeout: int | None = None, **parameters: Any) -> Iterator[dict[str, Any]]:
        from neo4j import Query

        with self.driver.session(database=self.database) as session:
            for record in session.run(Query(cypher, timeout=timeout), **parameters):
                yield dict(record)

    def get_num_entities(self) -> int:
        return int(self.run_query("MATCH (n) RETURN count(n) AS count")[0]["count"])

    def get_num_relations(self) -> int:
        return int(self.run_query("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"])
