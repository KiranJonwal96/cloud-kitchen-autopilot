"""Base class implementing the PRAL cycle (Perceive -> Reason -> Act, plus Learn)."""

import time
from abc import ABC, abstractmethod

try:  # LangSmith @traceable: records tool and agent runs when LANGSMITH_TRACING=true
    from langsmith import traceable
except ImportError:  # keeps the module importable without langsmith (no-op decorator)
    def traceable(*_args, **_kwargs):
        return lambda func: func


class AgentError(Exception):
    """Raised when an agent cannot produce a usable result (for example, critical data is missing)."""


def call_tool(trace: list[dict], tool_name: str, func, *args, **kwargs):
    """Call a tool and record the call (status and duration) in the decision trace."""
    started = time.perf_counter()
    try:
        result = traceable(name=tool_name, run_type="tool")(func)(*args, **kwargs)
    except Exception as error:
        trace.append({"component": tool_name, "kind": "tool", "status": "failed", "detail": str(error),
                      "duration_ms": round((time.perf_counter() - started) * 1000, 1)})
        raise
    trace.append({"component": tool_name, "kind": "tool", "status": "ok", "detail": "",
                  "duration_ms": round((time.perf_counter() - started) * 1000, 1)})
    return result


def call_llm(trace: list[dict], usage, agent_name: str, llm, system: str, user: str, step: str,
             invoke=None) -> dict:
    """Call the LLM, record tokens/latency and the agent's reasoning in the trace, return the data.

    invoke defaults to llm.complete_json; pass another callable (system, user) -> LLMResult
    to use a different output mode, e.g. structured output.
    """
    started = time.perf_counter()
    try:
        result = (invoke or llm.complete_json)(system, user)
    except Exception as error:
        trace.append({"component": f"{agent_name}.llm", "kind": "llm", "status": "failed",
                      "detail": f"{step}: {error}",
                      "duration_ms": round((time.perf_counter() - started) * 1000, 1)})
        raise
    if usage is not None:
        usage.add(result.input_tokens, result.output_tokens, result.latency_ms)
    reasoning = result.data.get("thought") or result.data.get("reasoning") or result.data.get("risk_summary") or ""
    trace.append({
        "component": f"{agent_name}.llm", "kind": "llm", "status": "ok",
        "detail": f"{step}: {reasoning}".strip(),
        "tokens": result.input_tokens + result.output_tokens,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    })
    return result.data


class BaseAgent(ABC):
    name = "agent"

    @abstractmethod
    def perceive(self, task: dict) -> dict:
        """Gather the context the agent needs (via tools only)."""

    @abstractmethod
    def reason(self, context: dict) -> dict:
        """Interpret the context and reach conclusions."""

    @abstractmethod
    def act(self, conclusions: dict) -> dict:
        """Produce the structured output handed back to the caller."""

    def learn(self) -> dict:
        """Return the adaptation currently applied from past outcomes (default: none)."""
        return {}

    def run(self, task: dict, trace: list[dict]) -> dict:
        started = time.perf_counter()
        try:
            output = traceable(name=self.name, run_type="chain")(self._pral)(task)
        except Exception as error:
            trace.append(self._trace_entry(started, "failed", str(error)))
            raise
        trace.append(self._trace_entry(started, "ok", output.get("headline", "")))
        return output

    def _pral(self, task: dict) -> dict:
        context = self.perceive(task)
        conclusions = self.reason(context)
        return self.act(conclusions)

    def _trace_entry(self, started: float, status: str, detail: str) -> dict:
        return {
            "component": self.name,
            "kind": "agent",
            "status": status,
            "detail": detail,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
