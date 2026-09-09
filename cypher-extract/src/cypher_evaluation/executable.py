"""CypherKD executable metric, kept behavior-compatible with the reference."""

from .execution_accuracy import _neo4j_query_errors


def executable(
    pred_cypher: str,
    target_cypher: str,
    neo4j_connector,
    timeout: int = 120,
) -> float:
    """Whether the predicted Cypher query is executable."""

    del target_cypher
    try:
        neo4j_connector.run_query(pred_cypher, timeout=timeout)
    except _neo4j_query_errors():
        return 0.0
    except Exception as error:
        print(f"Warning: Exception {error} occurred while executing the predicted Cypher query {pred_cypher}")
        return 0.0
    return 1.0
