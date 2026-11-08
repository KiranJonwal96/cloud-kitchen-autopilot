"""Start Cloud Kitchen Autopilot: python run.py [--reset]"""

import argparse
import sys

from app.db import init_db
from app.logging_setup import configure_logging


def check_dependencies() -> None:
    missing = []
    for module in ("fastapi", "uvicorn", "langgraph", "langchain_core", "langchain_openai", "chromadb"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        print(f"Missing packages: {', '.join(missing)}\n"
              "Install them with:  pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)


def index_knowledge() -> None:
    """Index the SOP documents into ChromaDB at start-up (first run downloads a small embedding model)."""
    from app.rag import knowledge_base
    try:
        print("Indexing kitchen knowledge base...", knowledge_base.ingest(), file=sys.stderr)
    except Exception as error:  # noqa: BLE001 - the app still works without RAG
        print(f"Knowledge base not available ({error}); continuing without RAG.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cloud Kitchen Autopilot server")
    parser.add_argument("--reset", action="store_true", help="recreate the database with demo data")
    args = parser.parse_args()

    check_dependencies()
    configure_logging()
    init_db(reset=args.reset)
    index_knowledge()

    import uvicorn

    from app.api.main import app  # imported after logging is configured
    from app.config import settings
    print(f"Open http://{settings.app_host}:{settings.app_port}  (API docs: /docs)", file=sys.stderr)
    uvicorn.run(app, host=settings.app_host, port=settings.app_port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
