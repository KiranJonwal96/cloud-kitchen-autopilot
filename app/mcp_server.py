"""MCP server: lets other AI apps (e.g. Claude Desktop, MCP Inspector) use the kitchen tools.

Uses the official MCP Python SDK (v2: MCPServer).
Run:  python -m app.mcp_server        (stdio transport)

Exposed tools
- forecast_demand      : statistical demand forecast for the next N hours
- check_kitchen_status : current stock and station capacity
- search_procedures    : RAG search over the kitchen SOPs (ChromaDB)
- plan_for_demand      : run the full multi-agent workflow (LangGraph)

MCP clients get the same read-only information and suggestions as the web page; nothing here changes
stock, orders or the menu.
"""

import logging
import sys

from mcp.server.mcpserver import MCPServer

from app.agents.demand_agent import DemandAgent
from app.db import get_connection, init_db
from app.graph import workflow
from app.rag import knowledge_base

# stdout carries the MCP protocol, so logs must go to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
init_db()

mcp = MCPServer(
    "cloud-kitchen-autopilot",
    instructions="Tools for a cloud kitchen: demand forecast, stock and capacity, kitchen procedures, "
                 "and multi-agent suggestions.",
)


@mcp.tool()
def forecast_demand(horizon_hours: int = 2) -> dict:
    """Forecast the number of orders for the next 1-6 hours and say whether a demand spike is expected."""
    horizon_hours = max(1, min(6, int(horizon_hours)))
    forecast = DemandAgent(trace=[]).run({"horizon_hours": horizon_hours}, trace=[])
    return {key: forecast[key] for key in ("horizon_hours", "expected_orders", "normal_orders",
                                           "uplift_pct", "spike", "confidence", "reason_codes", "per_hour")}


@mcp.tool()
def check_kitchen_status() -> dict:
    """Current stock of every ingredient and the hourly capacity of every kitchen station."""
    with get_connection() as conn:
        stock = [dict(r) for r in conn.execute(
            "SELECT name, stock, unit, safety_stock FROM ingredients ORDER BY id").fetchall()]
        stations = [dict(r) for r in conn.execute(
            "SELECT name, capacity_per_hour FROM stations ORDER BY id").fetchall()]
    return {"ingredients": stock, "stations": stations}


@mcp.tool()
def search_procedures(query: str) -> list[dict]:
    """Search the kitchen's standard operating procedures and policies (RAG). Returns cited excerpts."""
    return [{"id": h["id"], "doc_id": h["doc_id"], "title": h["title"], "section": h["section"], "score": h["score"], "text": h["text"]}
            for h in knowledge_base.search(query)]


@mcp.tool()
def plan_for_demand(request: str) -> dict:
    """Ask the multi-agent autopilot a question, e.g. 'Can we handle the next 2 hours?'.
    Returns the answer, the orders that can't be fulfilled and the suggestions."""
    decision = workflow.run_plan(request)
    assessment = decision.get("assessment_json") or {}
    return {
        "decision_id": decision["id"],
        "status": decision["status"],
        "answer": decision["summary"],
        "orders_we_cannot_fulfil": [d for d in assessment.get("dishes", []) if d["cannot_serve"] > 0],
        "suggestions": [s["description"] for s in decision.get("suggestions_json") or []],
    }


if __name__ == "__main__":
    mcp.run()
