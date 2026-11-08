"""Pydantic schemas for structured LLM output."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """The Orchestrator's next step in the LangGraph supervisor loop."""

    thought: str = Field(description="One or two sentences of reasoning")
    action: Literal["call_demand_agent", "call_operations_agent", "clarify", "escalate", "finish"] = Field(
        description="The next step")
    horizon_hours: Optional[int] = Field(default=None, description="Forecast horizon for call_demand_agent")
    question: Optional[str] = Field(default=None, description="Clarifying question for clarify")
    reason: Optional[str] = Field(default=None, description="Why, for escalate")
    summary: Optional[str] = Field(default=None, description="Answer for the supervisor, for finish or escalate")
