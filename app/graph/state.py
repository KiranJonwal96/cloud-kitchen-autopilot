"""Shared LangGraph state for one planning workflow.

Values are plain data (dicts, lists, strings, numbers) so they are easy to save and inspect.
`trace` and `usage` use reducers so each node only returns what it added.
"""

import operator
from typing import Annotated, Optional, TypedDict


def add_usage(current: dict | None, update: dict | None) -> dict:
    """Reducer: sum LLM usage counters reported by each node."""
    result = dict(current or {})
    for key, value in (update or {}).items():
        result[key] = round(result.get(key, 0) + value, 1)
    return result


class KitchenState(TypedDict, total=False):
    # Input
    request: str
    explicit_horizon: Optional[int]
    trace_id: str
    started_at: float

    # Perceived context (guardrail node)
    check: dict
    current: dict
    horizon: int
    assumptions: list

    # Planning (supervisor node)
    mode: str                      # llm | llm_fallback | rule_based
    fallback_reason: Optional[str]
    steps: list                    # planner steps: thought, action, args, observation
    step_count: int
    next_node: str
    pending_horizon: int
    llm_summary: Optional[str]
    clarify_question: Optional[str]
    llm_escalation: Optional[str]

    # Specialist results
    forecast: Optional[dict]
    forecast_horizon: Optional[int]
    assessment: Optional[dict]
    demand_error: Optional[str]
    operations_error: Optional[str]

    # Accumulated across nodes
    trace: Annotated[list, operator.add]
    usage: Annotated[dict, add_usage]

    # Output
    decision_id: Optional[int]
