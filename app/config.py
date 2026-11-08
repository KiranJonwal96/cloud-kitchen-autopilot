"""Application configuration loaded from environment variables (and an optional .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding real environment variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # On a hosting platform (Render sets PORT) listen on all interfaces; locally only on this computer.
    app_host: str = os.getenv("APP_HOST") or ("0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    # PORT is set by hosting platforms such as Render; APP_PORT is for local use.
    app_port: int = int(os.getenv("PORT") or os.getenv("APP_PORT", "8000"))
    app_version: str = os.getenv("APP_VERSION", "4.0.0")
    # Relative paths are resolved from the project root so the app runs from any folder.
    db_path: str = str(PROJECT_ROOT / os.getenv("DB_PATH", "data/kitchen.db"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # Optional HTTP Basic Auth for the whole site (set both when deploying publicly, e.g. on Render).
    basic_auth_user: str = os.getenv("BASIC_AUTH_USER", "")
    basic_auth_password: str = os.getenv("BASIC_AUTH_PASSWORD", "")

    # Agent / orchestration thresholds
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    spike_ratio: float = float(os.getenv("SPIKE_RATIO", "1.3"))
    capacity_warning_utilisation: float = float(os.getenv("CAPACITY_WARNING_UTILISATION", "0.85"))
    max_horizon_hours: int = int(os.getenv("MAX_HORIZON_HOURS", "6"))
    default_horizon_hours: int = int(os.getenv("DEFAULT_HORIZON_HOURS", "2"))

    # Policy limits (INR)
    max_spend_per_po: float = float(os.getenv("MAX_SPEND_PER_PO", "25000"))
    daily_procurement_budget: float = float(os.getenv("DAILY_PROCUREMENT_BUDGET", "60000"))

    # Learning boundaries: automatic bias correction is clamped to this range.
    learning_min_factor: float = float(os.getenv("LEARNING_MIN_FACTOR", "0.85"))
    learning_max_factor: float = float(os.getenv("LEARNING_MAX_FACTOR", "1.15"))

    # Tool reliability
    tool_max_retries: int = int(os.getenv("TOOL_MAX_RETRIES", "2"))

    # Live data sources (default: live). "simulated" = demo values only.
    weather_source: str = os.getenv("WEATHER_SOURCE", "open_meteo").strip().lower()     # or "simulated"
    kitchen_latitude: float = float(os.getenv("KITCHEN_LATITUDE", "12.9716"))            # default: Bengaluru
    kitchen_longitude: float = float(os.getenv("KITCHEN_LONGITUDE", "77.5946"))
    holiday_source: str = os.getenv("HOLIDAY_SOURCE", "calendarific").strip().lower()   # or "simulated"
    calendarific_api_key: str = os.getenv("CALENDARIFIC_API_KEY", "")
    holiday_country: str = os.getenv("HOLIDAY_COUNTRY", "IN")
    holiday_location: str = os.getenv("HOLIDAY_LOCATION", "")                           # e.g. "in-ka" (optional)
    # The kitchen's own time zone, so the live clock is right on a server in another zone (Render runs on UTC).
    kitchen_timezone: str = os.getenv("KITCHEN_TIMEZONE", "Asia/Kolkata")

    # LLM settings (LangChain ChatOpenAI). Provider: openai | none (none = rule-based agents)
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    llm_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    # Only for course/proxy keys (e.g. Vocareum "voc-" keys): the OpenAI-compatible endpoint to use.
    llm_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1200"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    orchestrator_max_steps: int = int(os.getenv("ORCHESTRATOR_MAX_STEPS", "6"))

    # RAG: ChromaDB vector store over the kitchen SOP documents in knowledge/
    knowledge_dir: str = str(PROJECT_ROOT / os.getenv("KNOWLEDGE_DIR", "knowledge"))
    chroma_path: str = str(PROJECT_ROOT / os.getenv("CHROMA_PATH", "data/chroma"))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "kitchen_sops")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.25"))


    # LangSmith tracing is read directly by LangChain/LangGraph from LANGSMITH_TRACING,
    # LANGSMITH_API_KEY and LANGSMITH_PROJECT. Shown here only for status reporting.
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "cloud-kitchen-autopilot")

settings = Settings()
