"""Builds and runs the LangGraph workflow.

    START -> guardrail -> supervisor <-> demand_agent / operations_agent
                                  \\-> finalize -> END

The supervisor (Orchestrator) decides at run time which agent works next; the agents always
report back to it. The finalize step checks the suggestions against the rules and saves the answer.
"""

import logging
import time
import uuid

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import DONE, make_nodes, route_next
from app.graph.state import KitchenState
from app.llm.client import LLMError, build_llm
from app.logging_setup import trace_id_var
from app.tools import memory_store

logger = logging.getLogger(__name__)


def build_graph(llm):
    nodes = make_nodes(llm)
    graph = StateGraph(KitchenState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", route_next, {"supervisor": "supervisor", "finalize": "finalize"})
    graph.add_conditional_edges("supervisor", route_next, {
        "supervisor": "supervisor",
        "demand_agent": "demand_agent",
        "operations_agent": "operations_agent",
        "finalize": "finalize",
    })
    graph.add_edge("demand_agent", "supervisor")      # agents always report back
    graph.add_edge("operations_agent", "supervisor")
    graph.add_conditional_edges("finalize", route_next, {DONE: END})
    return graph.compile()


def _resolve_llm(llm):
    if llm != "auto":
        return llm
    try:
        return build_llm()
    except (LLMError, ValueError) as error:
        logger.warning("LLM not available, using rule-based mode: %s", error)
        return None


def run_plan(request: str, horizon_hours: int | None = None, llm="auto") -> dict:
    """Run the multi-agent workflow for one question and return the saved decision."""
    llm = _resolve_llm(llm)
    trace_id = trace_id_var.get()
    if trace_id == "-":
        trace_id = uuid.uuid4().hex[:16]
    initial = {"request": request, "explicit_horizon": horizon_hours, "trace_id": trace_id,
               "started_at": time.time(), "trace": [], "usage": {}}
    config = {
        "recursion_limit": 60,
        "run_name": "kitchen_autopilot_plan",        # shown in LangSmith
        "tags": ["cloud-kitchen-autopilot"],
        "metadata": {"trace_id": trace_id},          # find a LangSmith run by our request id
    }
    try:
        final_state = build_graph(llm).invoke(initial, config)
        decision_id = final_state.get("decision_id")
        if decision_id is None:
            raise RuntimeError("Workflow ended without a decision")
    except Exception:  # noqa: BLE001 - any unexpected failure becomes an escalated decision
        logger.exception("Workflow failed")
        decision_id = memory_store.save_decision(
            trace_id=trace_id, request=request, horizon_hours=horizon_hours, status="escalated",
            summary="Unexpected system error. Please check the system.",
            escalation_reasons=["System error"], forecast=None, assessment=None, suggestions=[], trace=[],
            latency_ms=round((time.time() - initial["started_at"]) * 1000, 1),
            llm={"framework": "langgraph", "mode": "error"},
        )
    return memory_store.get_decision(decision_id)
