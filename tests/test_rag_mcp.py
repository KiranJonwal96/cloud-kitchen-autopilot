"""Tests for RAG (ChromaDB knowledge base), dynamic prompts and the MCP server.

Run with: python -m unittest discover -s tests -t . -v
"""

import asyncio
import os
import tempfile
import unittest

if "DB_PATH" not in os.environ:
    os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_rag.db")
    os.environ["CHECKPOINT_DB_PATH"] = os.path.join(os.path.dirname(os.environ["DB_PATH"]), "checkpoints.db")
os.environ["LLM_PROVIDER"] = "none"
os.environ.setdefault("WEATHER_SOURCE", "simulated")
os.environ.setdefault("HOLIDAY_SOURCE", "simulated")

from app.db import init_db  # noqa: E402
FIXED_CLOCK = {"simulated_date": "2026-11-07", "simulated_hour": 19}  # a Saturday evening, so results never depend on today
from app.graph import workflow  # noqa: E402
from app.llm import prompts  # noqa: E402
from app.rag import knowledge_base  # noqa: E402
from app.tools import data_tools  # noqa: E402

SURGE = {"incidents": {"rain": True, "cricket_match": True, "order_rush": True}}


class TestRag(unittest.TestCase):
    def setUp(self):
        init_db(reset=True)
        data_tools.update_sim_state(FIXED_CLOCK)

    def test_rag_01_documents_are_chunked_with_metadata(self):
        chunks = knowledge_base.load_chunks()
        self.assertGreaterEqual(len(chunks), 15)
        first = chunks[0]
        self.assertRegex(first["id"], r"^[A-Z]+-\d+#\d+$")
        for key in ("doc_id", "title", "section", "version", "source", "checksum"):
            self.assertIn(key, first["metadata"])

    def test_rag_02_search_finds_the_right_procedure(self):
        hits = knowledge_base.search("chicken stockout risk: stock will run out before the next delivery")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["doc_id"], "SOP-01")

    def test_rag_03_no_answer_for_unrelated_query(self):
        self.assertEqual(knowledge_base.search("best football team in Europe"), [])

    def test_rag_04_reindex_skips_unchanged_chunks(self):
        knowledge_base.ingest()
        self.assertEqual(knowledge_base.ingest()["updated"], 0)

    def test_rag_05_operations_agent_uses_procedures(self):
        data_tools.update_sim_state(SURGE)
        assessment = workflow.run_plan("Can we handle the next 2 hours?", llm=None)["assessment_json"]
        doc_ids = {k["doc_id"] for k in assessment["knowledge"]}
        self.assertIn("SOP-01", doc_ids)  # shortage procedure
        self.assertIn("SOP-04", doc_ids)  # peak-hour playbook

    def test_rag_06_no_risks_means_no_retrieval(self):
        assessment = workflow.run_plan("Can we handle the next 2 hours?", llm=None)["assessment_json"]
        self.assertEqual(assessment["knowledge"], [])


class TestDynamicPrompts(unittest.TestCase):
    def test_prompt_01_orchestrator_includes_live_context(self):
        prompt = prompts.orchestrator_system(2, 6, "2026-11-07 19:00")
        self.assertIn("It is now 2026-11-07 19:00", prompt)

    def test_prompt_02_demand_prompt_includes_recent_accuracy(self):
        self.assertEqual(prompts.demand_system({"samples": 0}), prompts.DEMAND_SYSTEM)
        prompt = prompts.demand_system({"samples": 4, "mape_pct": 12.0, "bias_pct": 8.0})
        self.assertIn("under-forecast", prompt)

    def test_prompt_03_operations_prompt_depends_on_retrieval(self):
        self.assertIn("Cite only ids", prompts.operations_system(True))
        self.assertIn("Do not invent procedures", prompts.operations_system(False))


class TestMcpServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import mcp_server
        cls.server = mcp_server

    def setUp(self):
        init_db(reset=True)
        data_tools.update_sim_state(FIXED_CLOCK)

    def test_mcp_01_tools_are_registered(self):
        tools = asyncio.run(self.server.mcp.list_tools())
        self.assertEqual({t.name for t in tools},
                         {"forecast_demand", "check_kitchen_status", "search_procedures", "plan_for_demand"})
        self.assertTrue(all(t.description for t in tools))  # descriptions tell clients when to use each tool

    def test_mcp_02_no_tool_changes_orders(self):
        names = [n for n in dir(self.server) if "approve" in n.lower() or "execute" in n.lower()]
        self.assertEqual(names, [])

    def test_mcp_03_call_tool_through_mcp(self):
        data_tools.update_sim_state(SURGE)
        result = asyncio.run(self.server.mcp.call_tool("forecast_demand", {"horizon_hours": 2}))
        self.assertIn("expected_orders", str(result))

    def test_mcp_04_tools_return_data(self):
        data_tools.update_sim_state(SURGE)
        forecast = self.server.forecast_demand(2)
        self.assertTrue(forecast["spike"])
        self.assertEqual(len(self.server.check_kitchen_status()["stations"]), 4)
        self.assertEqual(self.server.search_procedures("station over capacity during peak")[0]["doc_id"], "SOP-04")
        plan = self.server.plan_for_demand("Can we handle the next 2 hours?")
        self.assertEqual(plan["status"], "action_needed")
        self.assertTrue(plan["orders_we_cannot_fulfil"])
        self.assertTrue(plan["suggestions"])


if __name__ == "__main__":
    unittest.main()
