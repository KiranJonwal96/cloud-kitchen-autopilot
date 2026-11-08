"""Deterministic finalisation applied to every plan, whatever route the LLM took:
escalation rules, policy checks on each suggestion, the final answer and saving it.
"""

import time

from app.config import settings
from app.db import get_connection
from app.tools import memory_store, policy_engine


def check_suggestions(proposals: list[dict], trace: list[dict]) -> list[dict]:
    """Run every suggestion through the policy engine and attach the result."""
    started = time.perf_counter()
    checked = []
    planned_spend = 0.0
    with get_connection() as conn:
        for proposal in proposals:
            policy = policy_engine.validate_action(conn, proposal["action_type"], proposal["payload"],
                                                   planned_spend)
            if proposal["action_type"] == policy_engine.PURCHASE_ORDER and policy["allowed"]:
                planned_spend += proposal["payload"]["total_cost"]
            checked.append({**proposal, "policy": policy})
    trace.append({"component": "policy_engine", "kind": "tool", "status": "ok",
                  "detail": f"{len(checked)} suggestion(s) checked",
                  "duration_ms": round((time.perf_counter() - started) * 1000, 1)})
    return checked


def cannot_fulfil_sentence(dishes: list[dict]) -> str:
    """One sentence, grouped by the short ingredient, e.g.
    'With current stock we can't fulfil about 22 expected orders: 13 Chicken Biryani and 9 Butter Chicken
    (Chicken runs out around 20:40; reorder now, a delivery can arrive about 20:00).'"""
    short = [d for d in dishes if d["cannot_serve"] > 0]
    if not short:
        return ""
    groups: dict[str, list[dict]] = {}
    for dish in short:
        groups.setdefault(dish["short_ingredient"], []).append(dish)
    parts = []
    for ingredient, items in groups.items():
        names = " and ".join(f"{d['cannot_serve']} {d['dish']}" for d in items)
        first = items[0]
        when = f"{ingredient} runs out around {first['runs_out_at']}" if first["runs_out_at"] else f"{ingredient} is short"
        fix = (f"reorder now, a delivery can arrive about {first['restock_at']}" if first["restock_in_time"]
               else f"the next delivery only arrives about {first['restock_at']}")
        parts.append(f"{names} ({when}; {fix})")
    total = sum(d["cannot_serve"] for d in short)
    return f"With current stock we can't fulfil about {total} expected orders: " + "; ".join(parts) + "."


def summarise(result: dict, llm_summary: str | None) -> str:
    assessment = result.get("assessment")
    dishes = assessment["dishes"] if assessment else []
    if llm_summary:
        # The LLM writes the answer; code appends the facts it must not get wrong.
        parts = [llm_summary.strip()]
        if cannot_fulfil_sentence(dishes) and "can't fulfil" not in llm_summary:
            parts.append(cannot_fulfil_sentence(dishes))
        if result["escalation_reasons"]:
            parts.append("Needs your attention: " + "; ".join(result["escalation_reasons"]))
        return " ".join(parts)

    parts = []
    forecast = result.get("forecast")
    if forecast:
        spike_text = "a demand spike" if forecast["spike"] else "no unusual spike"
        parts.append(
            f"Expected about {forecast['expected_orders']:.0f} orders in the next "
            f"{forecast['horizon_hours']}h ({forecast['uplift_pct']:+.0f}% vs a normal "
            f"{forecast['normal_orders']:.0f}); {spike_text}."
        )
    if assessment:
        sentence = cannot_fulfil_sentence(dishes)
        bottlenecks = [s["name"] for s in assessment["kitchen"] if s["status"] == "bottleneck"]
        actionable = [x for x in result["suggestions"] if x["action_type"] != policy_engine.ADVISORY]
        if not sentence and not bottlenecks and not actionable:
            parts.append("Yes, stock and kitchen capacity can handle it.")
        else:
            if sentence:
                parts.append(sentence)
            if bottlenecks:
                parts.append("Overloaded: " + ", ".join(bottlenecks) + ".")
            parts.append("See the suggestions below.")
    if result["escalation_reasons"]:
        parts.append("Needs your attention: " + "; ".join(result["escalation_reasons"]))
    return " ".join(parts)


def build_result(state: dict, trace: list[dict]) -> dict:
    """Turn the graph state into a decision: status, escalations, checked suggestions and summary."""
    check = state.get("check") or {}
    base = {"forecast": state.get("forecast"), "assessment": state.get("assessment"),
            "suggestions": [], "escalation_reasons": [],
            "horizon_hours": state.get("forecast_horizon") or state.get("horizon"),
            "assumptions": state.get("assumptions", [])}

    if check and not check.get("ok"):
        return {**base, "status": check["status"], "summary": check["reason"], "horizon_hours": None,
                "assumptions": []}
    if state.get("clarify_question"):
        return {**base, "status": "needs_clarification", "summary": state["clarify_question"]}

    escalation: list[str] = []
    if state.get("demand_error"):
        escalation.append(f"Demand forecast unavailable: {state['demand_error']}")
    if state.get("operations_error"):
        escalation.append(f"Critical data missing: {state['operations_error']}")
    if state.get("llm_escalation"):
        escalation.append(f"Orchestrator escalated: {state['llm_escalation']}")
    forecast, assessment = state.get("forecast"), state.get("assessment")
    if forecast and forecast["needs_review"]:
        escalation.append(f"Low forecast confidence ({forecast['confidence']}), please double-check")

    suggestions = []
    if assessment:
        suggestions = check_suggestions(assessment["proposed_actions"], trace)
        for suggestion in suggestions:
            if not suggestion["policy"]["allowed"]:
                escalation.append(f"Rule broken by '{suggestion['description']}': "
                                  + "; ".join(suggestion["policy"]["violations"]))

    problems = assessment and (any(d["cannot_serve"] > 0 for d in assessment["dishes"])
                               or any(s["action_type"] != policy_engine.ADVISORY for s in suggestions)
                               or any(k["status"] == "bottleneck" for k in assessment["kitchen"]))
    status = "escalated" if escalation else ("action_needed" if problems else "ok")
    result = {**base, "status": status, "suggestions": suggestions, "escalation_reasons": escalation}
    result["summary"] = summarise(result, state.get("llm_summary"))
    return result


def persist(state: dict, result: dict, trace: list[dict], llm_info: dict) -> int:
    latency_ms = round((time.time() - state["started_at"]) * 1000, 1)
    return memory_store.save_decision(
        trace_id=state["trace_id"], request=state["request"], horizon_hours=result["horizon_hours"],
        status=result["status"], summary=result["summary"], escalation_reasons=result["escalation_reasons"],
        forecast=result["forecast"], assessment=result["assessment"], suggestions=result["suggestions"],
        trace=trace, latency_ms=latency_ms, assumptions=result["assumptions"], llm=llm_info,
    )
