from servemind.api.routes.evaluations import cases
from servemind.evaluation.expanded_cases import (
    EXPANDED_COMMERCE_CASES, EXPANDED_INTENT_CASES, EXPANDED_TASK_GRAPH_CASES,
)


def test_evaluation_catalog_has_more_than_one_thousand_unique_cases():
    suites = cases()
    case_ids = [item.get("id", f"end-to-end-{index}")
                for suite in suites.values() for index, item in enumerate(suite)]
    assert len(case_ids) == 1051
    assert len(case_ids) == len(set(case_ids))
    assert len(EXPANDED_INTENT_CASES) == 300
    assert len(EXPANDED_TASK_GRAPH_CASES) == 200
    assert len(EXPANDED_COMMERCE_CASES) == 240


def test_expansion_covers_orders_merchants_categories_and_stock_states():
    # Two anonymous orders can share a SKU; a few SKU-only questions repeat.
    assert len({case["message"] for case in EXPANDED_INTENT_CASES}) >= 295
    assert len({case["message"] for case in EXPANDED_TASK_GRAPH_CASES}) == 200
    assert len({case["message"] for case in EXPANDED_COMMERCE_CASES}) == 240
    assert len({case["product"]["merchant_id"] for case in EXPANDED_COMMERCE_CASES}) == 4
    assert len({case["product"]["category"] for case in EXPANDED_COMMERCE_CASES}) == 10
    assert len({case["product"]["catalog_status"] for case in EXPANDED_COMMERCE_CASES}) == 5
    assert sum("must_intents" in case and len(case["must_intents"]) >= 2
               for case in EXPANDED_TASK_GRAPH_CASES) == 200
