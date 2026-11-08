"""Operational metrics computed from the stored decisions and request logs."""

import json
import statistics

from app.db import get_connection


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[max(0, int(round(0.95 * len(ordered))) - 1)], 1)


def compute_metrics() -> dict:
    with get_connection() as conn:
        decisions = [dict(r) for r in conn.execute(
            "SELECT status, latency_ms, llm_json FROM decisions").fetchall()]
        outcomes = [dict(r) for r in conn.execute(
            "SELECT forecast_orders, actual_orders FROM forecast_outcomes").fetchall()]
        requests = [dict(r) for r in conn.execute(
            "SELECT status, latency_ms FROM request_metrics WHERE path LIKE '/api/%'").fetchall()]

    status_counts: dict[str, int] = {}
    for decision in decisions:
        status_counts[decision["status"]] = status_counts.get(decision["status"], 0) + 1
    planning = [d for d in decisions if d["status"] not in ("refused", "needs_clarification")]

    errors = [abs(o["actual_orders"] - o["forecast_orders"]) / o["actual_orders"]
              for o in outcomes if o["actual_orders"] > 0]
    mape = statistics.mean(errors) * 100 if errors else None

    llm_infos = [json.loads(d["llm_json"] or "{}") for d in planning]
    llm_runs = [i for i in llm_infos if i.get("mode") in ("llm", "llm_fallback")]
    llm_tokens = sum(i.get("input_tokens", 0) + i.get("output_tokens", 0) for i in llm_infos)
    server_errors = [r for r in requests if r["status"] >= 500]
    return {
        "decisions_total": len(decisions),
        "decisions_by_status": status_counts,
        "escalation_rate_pct": round(100 * status_counts.get("escalated", 0) / len(planning), 1) if planning else 0.0,
        "decision_latency_p95_ms": _p95([d["latency_ms"] for d in planning]),
        "forecast_accuracy_pct": round(100 - mape, 1) if mape is not None else None,
        "forecast_outcomes_recorded": len(outcomes),
        "llm_calls_total": sum(i.get("calls", 0) for i in llm_infos),
        "llm_tokens_per_decision": round(llm_tokens / len(llm_runs), 1) if llm_runs else 0.0,
        "llm_fallback_rate_pct": round(100 * sum(i.get("mode") == "llm_fallback" for i in llm_runs) / len(llm_runs), 1)
        if llm_runs else 0.0,
        "api_error_rate_pct": round(100 * len(server_errors) / len(requests), 1) if requests else 0.0,
        "api_latency_p95_ms": _p95([r["latency_ms"] for r in requests]),
    }
