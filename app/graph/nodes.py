"""LangGraph nodes.

guardrail -> supervisor <-> (demand_agent | operations_agent) -> finalize -> END

The supervisor node is the Orchestrator Agent: an LLM chooses the next step through LangChain
structured output (RouteDecision). Code enforces dependencies, the step limit and fallback.
The finalize node checks every suggestion against the rules and saves the answer.
"""

import json
import logging

from pydantic import ValidationError

from app.agents import guardrails
from app.agents.base import AgentError, call_llm, call_tool
from app.agents.demand_agent import DemandAgent
from app.agents.operations_agent import OperationsAgent
from app.config import settings
from app.graph import finalize
from app.llm import prompts
from app.llm.client import LLMUsage
from app.llm.schemas import RouteDecision
from app.tools import data_tools

logger = logging.getLogger(__name__)

DONE = "done"


# --- Observation helpers (what the supervisor LLM sees after each agent call) -------------

def forecast_observation(forecast: dict) -> dict:
    keys = ("horizon_hours", "expected_orders", "normal_orders", "uplift_pct", "spike", "confidence",
            "needs_review", "reason_codes", "reasoning")
    return {k: forecast.get(k) for k in keys}


def assessment_observation(assessment: dict) -> dict:
    return {
        "sla_risk": assessment["sla_risk"],
        "inventory_risks": [{"name": i["name"], "status": i["status"], "shortfall": i["net_requirement"],
                             "unit": i["unit"], "arrives_in_time": i["arrives_in_time"]}
                            for i in assessment["inventory"] if i["status"] != "ok"],
        "kitchen_bottlenecks": [{"name": s["name"], "utilisation_pct": s["utilisation_pct"]}
                                for s in assessment["kitchen"] if s["status"] != "ok"],
        "orders_we_cannot_fulfil": [{"dish": d["dish"], "cannot_serve": d["cannot_serve"],
                                     "short_ingredient": d["short_ingredient"], "runs_out_at": d["runs_out_at"]}
                                    for d in assessment["dishes"] if d["cannot_serve"] > 0],
        "suggestions": [{"type": p["action_type"], "description": p["description"]}
                        for p in assessment["proposed_actions"]],
        "risk_summary": assessment.get("risk_summary", ""),
        "prep_plan": assessment.get("prep_plan", ""),
    }


def _with_observation(steps: list, observation: dict) -> list:
    """Return a copy of steps with the observation set on the last step."""
    if not steps:
        return steps
    updated = [dict(s) for s in steps]
    updated[-1]["observation"] = observation
    return updated


def _safe_horizon(value, default: int) -> int:
    try:
        hours = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(settings.max_horizon_hours, hours))


# --- Node factory -----------------------------------------------------------------------

def make_nodes(llm):
    """Build node functions bound to an LLM adapter (or None for rule-based mode)."""

    def guardrail_node(state: dict) -> dict:
        """Perceive: deterministic input guardrail before any LLM sees the request."""
        trace: list[dict] = []
        check = guardrails.check_request(state["request"])
        trace.append({"component": "input_guardrail", "kind": "tool",
                      "status": "ok" if check["ok"] else check["status"],
                      "detail": check["reason"], "duration_ms": 0})
        current = call_tool(trace, "order_api.current", data_tools.get_current_orders)
        horizon, assumptions = guardrails.parse_horizon(
            check["cleaned"], state.get("explicit_horizon"), current["current_hour"],
            settings.default_horizon_hours, settings.max_horizon_hours)
        return {
            "check": check, "current": current, "horizon": horizon, "assumptions": assumptions,
            "mode": "llm" if llm is not None else "rule_based", "fallback_reason": None,
            "steps": [], "step_count": 0, "forecast": None, "assessment": None,
            "demand_error": None, "operations_error": None, "llm_summary": None,
            "clarify_question": None, "llm_escalation": None, "trace": trace,
            "next_node": "supervisor" if check["ok"] else "finalize",
        }

    def _rule_route(state: dict) -> dict:
        """Fixed sequence used without an LLM (or after the LLM planner failed)."""
        if not state.get("forecast") and not state.get("demand_error"):
            return {"next_node": "demand_agent", "pending_horizon": state["horizon"]}
        if state.get("forecast") and not state.get("assessment") and not state.get("operations_error"):
            return {"next_node": "operations_agent"}
        return {"next_node": "finalize"}

    def _ask_llm(state: dict, usage: LLMUsage, trace: list[dict]) -> RouteDecision:
        current = state["current"]
        system = prompts.orchestrator_system(
            settings.default_horizon_hours, settings.max_horizon_hours,
            current_time=f"{current['business_date']} {current['current_hour']:02d}:00")
        situation = {
            "current_time": f"{current['business_date']} {current['current_hour']:02d}:00",
            "kitchen_closes": "23:00",
            "horizon_parsed_from_request": state["horizon"],
            "explicit_horizon_from_ui": state.get("explicit_horizon"),
        }
        user = (f"<request>{state['check']['cleaned']}</request>\n"
                f"<situation>{json.dumps(situation)}</situation>\n"
                f"<steps_so_far>{json.dumps(state.get('steps', []), default=str)}</steps_so_far>\n"
                "Decide the next action.")
        step_label = f"plan step {state.get('step_count', 0) + 1}"
        invoke = None
        if hasattr(llm, "complete_structured"):
            # LangChain structured output, validated against the Pydantic schema.
            def invoke(system_prompt, user_prompt):
                return llm.complete_structured(RouteDecision, system_prompt, user_prompt)
        data = call_llm(trace, usage, "orchestrator", llm, system, user, step_label, invoke=invoke)
        return RouteDecision.model_validate(data)

    def supervisor_node(state: dict) -> dict:
        """Orchestrator Agent: decide the next node."""
        trace: list[dict] = []
        usage = LLMUsage()
        if state["mode"] != "llm" or llm is None:
            return {**_rule_route(state), "trace": trace}

        steps = [dict(s) for s in state.get("steps", [])]
        step_number = state.get("step_count", 0) + 1
        if step_number > settings.orchestrator_max_steps:
            steps.append({"step": "limit", "thought": "Step limit reached", "action": "stop",
                          "args": {}, "observation": None})
            update = {"steps": steps, "next_node": "finalize", "trace": trace}
            if not state.get("forecast"):
                update["llm_escalation"] = "Planner reached the step limit without a forecast"
            return update

        try:
            decision = _ask_llm(state, usage, trace)
        except ValidationError as error:
            steps.append({"step": step_number, "thought": "", "action": "invalid", "args": {},
                          "observation": {"error": f"Invalid decision format: {error.errors()[0]['msg']}"}})
            return {"steps": steps, "step_count": step_number, "next_node": "supervisor",
                    "trace": trace, "usage": usage.as_dict()}
        except Exception as error:  # noqa: BLE001 - LLM failure: continue with the fixed sequence
            logger.warning("LLM planner failed, switching to rule-based routing: %s", error)
            trace.append({"component": "orchestrator", "kind": "agent", "status": "fallback",
                          "detail": f"LLM planner failed: {error}", "duration_ms": 0})
            return {**_rule_route(state), "mode": "llm_fallback", "fallback_reason": str(error),
                    "trace": trace, "usage": usage.as_dict()}

        args = {k: v for k, v in decision.model_dump().items()
                if k not in ("thought", "action") and v is not None}
        step = {"step": step_number, "thought": decision.thought[:400], "action": decision.action,
                "args": args, "observation": None}
        update = {"step_count": step_number, "trace": trace, "usage": usage.as_dict()}

        if decision.action == "call_demand_agent":
            horizon = _safe_horizon(state.get("explicit_horizon") or decision.horizon_hours, state["horizon"])
            if state.get("forecast") and state.get("forecast_horizon") == horizon:
                step["observation"] = {"note": "Forecast for this horizon is already available.",
                                       "forecast": forecast_observation(state["forecast"])}
                update["next_node"] = "supervisor"
            else:
                update.update({"next_node": "demand_agent", "pending_horizon": horizon})
        elif decision.action == "call_operations_agent":
            if not state.get("forecast"):
                step["observation"] = {"error": "A demand forecast is required first. Call call_demand_agent."}
                update["next_node"] = "supervisor"
            elif state.get("assessment"):
                step["observation"] = {"note": "Assessment already available.",
                                       "assessment": assessment_observation(state["assessment"])}
                update["next_node"] = "supervisor"
            else:
                update["next_node"] = "operations_agent"
        elif decision.action == "clarify":
            update.update({"next_node": "finalize",
                           "clarify_question": (decision.question or "Could you clarify your request?")[:400]})
        elif decision.action == "escalate":
            update.update({"next_node": "finalize",
                           "llm_escalation": (decision.reason or "Escalated by orchestrator")[:300],
                           "llm_summary": (decision.summary or "")[:800] or None})
        else:  # finish
            if not state.get("forecast"):
                step["observation"] = {"error": "You cannot finish without a forecast. Call call_demand_agent "
                                                "first, or escalate if data is unavailable."}
                update["next_node"] = "supervisor"
            else:
                update.update({"next_node": "finalize", "llm_summary": (decision.summary or "")[:800] or None})

        update["steps"] = steps + [step]
        return update

    def demand_agent_node(state: dict) -> dict:
        trace: list[dict] = []
        usage = LLMUsage()
        horizon = state.get("pending_horizon") or state["horizon"]
        try:
            forecast = DemandAgent(trace, llm, usage).run({"horizon_hours": horizon}, trace)
            update = {"forecast": forecast, "forecast_horizon": horizon, "demand_error": None,
                      "assessment": None, "operations_error": None}
            observation = {"forecast": forecast_observation(forecast)}
        except AgentError as error:
            update = {"demand_error": str(error)}
            observation = {"error": str(error)}
        if state["mode"] == "llm":
            update["steps"] = _with_observation(state.get("steps", []), observation)
        return {**update, "trace": trace, "usage": usage.as_dict()}

    def operations_agent_node(state: dict) -> dict:
        trace: list[dict] = []
        usage = LLMUsage()
        try:
            assessment = OperationsAgent(trace, llm, usage).run({"forecast": state["forecast"]}, trace)
            update = {"assessment": assessment, "operations_error": None}
            observation = {"assessment": assessment_observation(assessment)}
        except AgentError as error:
            update = {"operations_error": str(error)}
            observation = {"error": str(error)}
        if state["mode"] == "llm":
            update["steps"] = _with_observation(state.get("steps", []), observation)
        return {**update, "trace": trace, "usage": usage.as_dict()}

    def finalize_node(state: dict) -> dict:
        """Deterministic rules: escalations, policy checks, summary, persistence."""
        trace: list[dict] = []
        result = finalize.build_result(state, trace)
        usage = state.get("usage") or {}
        llm_info = {
            "framework": "langgraph",
            "mode": state.get("mode", "rule_based"),
            "provider": getattr(llm, "provider", None),
            "model": getattr(llm, "model", None),
            "calls": int(usage.get("calls", 0)),
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "llm_latency_ms": usage.get("latency_ms", 0),
            "fallback_reason": state.get("fallback_reason"),
            "planner_steps": state.get("steps", []),
        }
        all_trace = list(state.get("trace", [])) + trace
        decision_id = finalize.persist(state, result, all_trace, llm_info)
        return {"decision_id": decision_id, "trace": trace, "next_node": DONE}

    return {
        "guardrail": guardrail_node,
        "supervisor": supervisor_node,
        "demand_agent": demand_agent_node,
        "operations_agent": operations_agent_node,
        "finalize": finalize_node,
    }


def route_next(state: dict) -> str:
    return state.get("next_node", DONE)

