"""Memory store (shared supporting capability, not an agent).

Long-term memory: every decision (question, forecast, assessment, suggestions, trace), the actual
order counts recorded later (used for learning), and an audit log. All of it is structured data,
so a relational database is enough (no vector database needed here).
"""

import json

from app.db import get_connection

DECISION_JSON_FIELDS = ("forecast_json", "assessment_json", "suggestions_json", "trace_json",
                        "escalation_reasons", "assumptions_json", "llm_json")


def audit(conn, trace_id: str, actor: str, event: str, details: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO audit_log (trace_id, actor, event, details_json) VALUES (?, ?, ?, ?)",
        (trace_id, actor, event, json.dumps(details or {}, default=str)),
    )


def save_decision(trace_id: str, request: str, horizon_hours: int | None, status: str, summary: str,
                  escalation_reasons: list[str], forecast: dict | None, assessment: dict | None,
                  suggestions: list[dict], trace: list[dict], latency_ms: float,
                  assumptions: list[str] | None = None, llm: dict | None = None) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO decisions (trace_id, request, horizon_hours, status, summary, escalation_reasons,
                   assumptions_json, llm_json, forecast_json, assessment_json, suggestions_json,
                   trace_json, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trace_id, request, horizon_hours, status, summary, json.dumps(escalation_reasons),
             json.dumps(assumptions or []), json.dumps(llm or {}, default=str),
             json.dumps(forecast, default=str), json.dumps(assessment, default=str),
             json.dumps(suggestions, default=str), json.dumps(trace, default=str), latency_ms),
        )
        decision_id = cursor.lastrowid
        audit(conn, trace_id, "orchestrator", "decision_created",
              {"decision_id": decision_id, "status": status, "suggestions": len(suggestions)})
    return decision_id


def get_decision(decision_id: int) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
    if row is None:
        return None
    decision = dict(row)
    for field in DECISION_JSON_FIELDS:
        if decision.get(field):
            decision[field] = json.loads(decision[field])
    return decision


def list_decisions(limit: int = 20) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, created_at, trace_id, request, horizon_hours, status, summary, latency_ms "
            "FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def record_forecast_outcome(decision_id: int, actual_orders: float, trace_id: str) -> dict:
    """Learning feedback: the real number of orders, compared later with the forecast."""
    decision = get_decision(decision_id)
    if decision is None or not decision.get("forecast_json"):
        raise ValueError("Decision not found or has no forecast")
    forecast_orders = decision["forecast_json"]["expected_orders"]
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO forecast_outcomes (decision_id, forecast_orders, actual_orders) VALUES (?, ?, ?) "
            "ON CONFLICT(decision_id) DO UPDATE SET actual_orders = excluded.actual_orders",
            (decision_id, forecast_orders, actual_orders),
        )
        audit(conn, trace_id, "supervisor", "outcome_recorded",
              {"decision_id": decision_id, "forecast": forecast_orders, "actual": actual_orders})
    return {"decision_id": decision_id, "forecast_orders": forecast_orders, "actual_orders": actual_orders}


def get_forecast_outcomes(limit: int = 20) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT forecast_orders, actual_orders FROM forecast_outcomes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def list_audit(limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    entries = []
    for row in rows:
        entry = dict(row)
        entry["details_json"] = json.loads(entry["details_json"] or "{}")
        entries.append(entry)
    return entries
