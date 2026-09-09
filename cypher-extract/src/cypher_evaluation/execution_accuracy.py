# ruff: noqa
"""CypherKD execution-accuracy implementation.

The algorithm is intentionally kept behavior-compatible with
``CypherKD_ref/src/metrics/execution_accuracy.py``. Do not simplify or make the
permutation search deterministic without re-baselining the published metrics.
"""

import logging
import random
import time
from collections import defaultdict
from itertools import product
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def _neo4j_query_errors():
    """Load the optional Neo4j exception classes only when evaluation runs."""

    try:
        import neo4j
    except ImportError:
        return ()
    return (
        neo4j.exceptions.CypherSyntaxError,
        neo4j.exceptions.DatabaseError,
        neo4j.exceptions.CypherTypeError,
        neo4j.exceptions.ClientError,
    )


def to_hashable(obj, unorder_list=True):
    """Recursively transform Cypher result values into CypherKD's hashable form."""

    if isinstance(obj, (tuple, int, float, str, bool, type(None))):
        return obj
    try:
        import neo4j
    except ImportError:
        neo4j = None
    if neo4j is not None and isinstance(obj, neo4j.time.Date):
        return obj.iso_format()
    elif isinstance(obj, (list, tuple)):
        if unorder_list:
            return tuple(sorted(to_hashable(item) for item in obj))
        else:
            return tuple(to_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return tuple(sorted(to_hashable(item) for item in obj))
    elif isinstance(obj, dict):
        return tuple(sorted((to_hashable(k), to_hashable(v)) for k, v in obj.items()))
    else:
        raise TypeError(f"Unhashable type: {type(obj)}")


def execution_accuracy(
    pred_cypher: str,
    target_cypher: str,
    neo4j_connector,
    timeout: int = 120,
) -> float:
    """Execution accuracy for two Cypher queries."""

    if pred_cypher == target_cypher:
        return 1.0
    t0 = time.time()
    target_executed = neo4j_connector.run_query(target_cypher, timeout=timeout)
    target_seconds = time.time() - t0
    if target_seconds > timeout:
        logger.warning(
            f"Execution of target cypher query took longer than {timeout} seconds. Query: {target_cypher}"
        )
    try:
        pred_executed = neo4j_connector.run_query(pred_cypher, timeout=timeout)
        pred_executed = [
            {k: to_hashable(v) for k, v in record.items()} for record in pred_executed
        ]
    except _neo4j_query_errors():
        return 0.0
    except TypeError:
        return 0.0
    except Exception as error:
        logger.warning(
            f"Exception {error} occurred while executing the predicted Cypher query: {pred_cypher}"
        )
        return 0.0

    target_executed = [
        {k: to_hashable(v) for k, v in record.items()} for record in target_executed
    ]
    return _compare_execution(
        pred_executed=pred_executed,
        target_executed=target_executed,
        order_matters="order by" in target_cypher.lower(),
    )


def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    for _ in range(20):
        random_tab2_row = random.choice(result2)
        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    if len(result1) == 0 and len(result2) == 0:
        return True
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])
    if len(result2[0]) != num_cols:
        return False
    if not quick_rej(result1, result2, order_matters):
        return False

    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


def to_tuples(result: List[Dict]) -> List[Tuple]:
    keys = list(result[0].keys())
    for row in result:
        assert set(row.keys()) == set(keys)
    return [tuple([row[key] for key in keys]) for row in result]


def _compare_execution(
    pred_executed: list[dict], target_executed: list[dict], order_matters: bool
) -> float:
    """Execution match considering same order of the output."""

    if not pred_executed and not target_executed:
        return 1.0
    elif not pred_executed or not target_executed:
        return 0.0

    gold_tuples = to_tuples(target_executed)
    pred_tuples = to_tuples(pred_executed)
    return float(result_eq(gold_tuples, pred_tuples, order_matters=order_matters))
