"""LLM access through LangChain (ChatOpenAI).

Agents talk to a small adapter with two methods:
- complete_json(system, user): JSON-mode reply, parsed into a dict (used by the specialists)
- complete_structured(schema, system, user): LangChain structured output validated by a
  Pydantic schema (used by the Orchestrator's routing decision)

LangChain automatically sends every model call to LangSmith when LANGSMITH_TRACING=true.
"""

import json
import re
import time
from dataclasses import dataclass, field

from app.config import settings


class LLMError(Exception):
    """Raised when the model call fails or does not return a usable result."""


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, latency_ms: float) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.latency_ms += latency_ms

    def as_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "latency_ms": round(self.latency_ms, 1)}


@dataclass
class LLMResult:
    data: dict
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw_text: str = field(repr=False, default="")


def extract_json(text: str) -> dict:
    """Parse a JSON object from model text, tolerating code fences or surrounding prose."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("Model response did not contain a JSON object") from None
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as error:
            raise LLMError(f"Model returned invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise LLMError("Model JSON must be an object")
    return value


def _usage(message) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


def _text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, list):  # some models return content blocks
        return "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    return str(content)


class LangChainLLM:
    """Adapter around a LangChain chat model."""

    def __init__(self, chat_model, provider: str, model: str):
        self.chat = chat_model
        self.provider = provider
        self.model = model
        # OpenAI JSON mode: the model must reply with a single JSON object.
        self._json_chat = chat_model.bind(response_format={"type": "json_object"})

    @staticmethod
    def _messages(system: str, user: str) -> list:
        from langchain_core.messages import HumanMessage, SystemMessage
        return [SystemMessage(content=system), HumanMessage(content=user)]

    def complete_json(self, system: str, user: str) -> LLMResult:
        last_error: Exception | None = None
        for attempt in range(2):
            prompt = user if attempt == 0 else (
                user + "\n\nYour previous reply was not valid JSON. Reply with ONE JSON object only.")
            started = time.perf_counter()
            try:
                message = self._json_chat.invoke(self._messages(system, prompt))
            except Exception as error:  # noqa: BLE001 - provider SDKs raise many error types
                raise LLMError(f"{self.provider} call failed: {error}") from error
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            input_tokens, output_tokens = _usage(message)
            text = _text(message)
            try:
                return LLMResult(extract_json(text), input_tokens, output_tokens, latency_ms, text)
            except LLMError as error:
                last_error = error
        raise last_error

    def complete_structured(self, schema, system: str, user: str) -> LLMResult:
        """LangChain structured output: the reply is validated against a Pydantic schema."""
        structured = self.chat.with_structured_output(schema, method="function_calling", include_raw=True)
        started = time.perf_counter()
        try:
            output = structured.invoke(self._messages(system, user))
        except Exception as error:  # noqa: BLE001
            raise LLMError(f"{self.provider} call failed: {error}") from error
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if output.get("parsing_error") or output.get("parsed") is None:
            raise LLMError(f"Structured output invalid: {output.get('parsing_error')}")
        input_tokens, output_tokens = _usage(output.get("raw"))
        return LLMResult(output["parsed"].model_dump(), input_tokens, output_tokens, latency_ms)


def build_llm() -> LangChainLLM | None:
    """Create the OpenAI chat model, or None when the LLM is off or no key is set."""
    if settings.llm_provider in ("", "none") or not settings.llm_api_key:
        return None
    if settings.llm_provider != "openai":
        raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use openai or none.")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise LLMError("langchain-openai is not installed. Run: pip install -r requirements.txt") from error
    options = {}
    if settings.llm_base_url:
        options["base_url"] = settings.llm_base_url  # e.g. a course proxy such as Vocareum
    chat = ChatOpenAI(model=settings.llm_model, api_key=settings.llm_api_key,
                      temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens,
                      timeout=settings.llm_timeout_seconds, max_retries=2, **options)
    return LangChainLLM(chat, "openai", settings.llm_model)


def describe_llm_mode() -> dict:
    """LLM and tracing status for the UI and /health (never includes keys)."""
    status = {"framework": "LangGraph + LangChain", "langsmith_tracing": settings.langsmith_tracing,
              "langsmith_project": settings.langsmith_project if settings.langsmith_tracing else None}
    try:
        llm = build_llm()
    except (LLMError, ValueError) as error:
        return {**status, "enabled": False, "provider": settings.llm_provider, "model": None, "reason": str(error)}
    if llm is None:
        reason = ("LLM_PROVIDER is 'none'" if settings.llm_provider in ("", "none")
                  else "OPENAI_API_KEY is not set")
        return {**status, "enabled": False, "provider": settings.llm_provider, "model": None, "reason": reason}
    return {**status, "enabled": True, "provider": llm.provider, "model": llm.model, "reason": ""}
