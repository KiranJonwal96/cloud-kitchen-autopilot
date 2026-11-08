"""Operations Agent: inventory and kitchen-capacity assessment against the demand forecast.

Perceive: forecast + inventory, recipes, kitchen capacity (tools).
Reason:   deterministic tools compute shortages, normal loss and station load (inventory and
          kitchen checks run in parallel) and build candidate actions; the LLM then selects
          and explains the actions and summarises operational risk.
Act:      returns the assessment and recommended actions to the Orchestrator.

Boundaries: it assesses and proposes. It does not forecast demand, choose or negotiate
with suppliers, authorise purchases or execute anything.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from app.agents.base import AgentError, BaseAgent, call_llm, call_tool
from app.config import settings
from app.llm import prompts
from app.rag import knowledge_base
from app.tools import calculations, data_tools, policy_engine

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


class OperationsAgent(BaseAgent):
    name = "operations_agent"

    def __init__(self, trace: list[dict], llm=None, usage=None):
        self.trace = trace
        self.llm = llm
        self.usage = usage

    # --- Perceive ---------------------------------------------------------------------
    def perceive(self, task: dict) -> dict:
        try:
            inventory = call_tool(self.trace, "inventory_api", data_tools.call_with_retry,
                                  data_tools.get_inventory, settings.tool_max_retries)
        except data_tools.ToolError as error:
            raise AgentError(f"Inventory data unavailable: {error}") from error
        return {
            "forecast": task["forecast"],
            "inventory": inventory,
            "recipes": call_tool(self.trace, "recipe_db", data_tools.get_recipes),
            "kitchen": call_tool(self.trace, "kitchen_api", data_tools.get_kitchen_capacity),
            "current": call_tool(self.trace, "order_api.current", data_tools.get_current_orders),
        }

    # --- Reason -----------------------------------------------------------------------
    def reason(self, context: dict) -> dict:
        # Inventory and kitchen assessments are independent once the forecast exists.
        with ThreadPoolExecutor(max_workers=2) as pool:
            inventory_future = pool.submit(self._assess_inventory, context)
            kitchen_future = pool.submit(self._assess_kitchen, context)
            inventory = inventory_future.result()
            kitchen = kitchen_future.result()
        return {"inventory": inventory, "kitchen": kitchen, "context": context}

    def _available_item_forecast(self, context: dict) -> dict[int, float]:
        available = {item["id"] for item in context["kitchen"]["menu_items"] if item["available"]}
        return {int(item_id): orders for item_id, orders in context["forecast"]["per_item"].items()
                if int(item_id) in available}

    @staticmethod
    def _clock(start_hour: int, minutes: float) -> str:
        total = int(start_hour * 60 + minutes)
        return f"{(total // 60) % 24:02d}:{total % 60:02d}"

    def _runout_minutes(self, ingredient: dict, context: dict, item_forecast: dict) -> float | None:
        """Minutes from now until stock runs out (ignoring safety stock), or None if it lasts the window."""
        total_orders = sum(item_forecast.values()) or 1
        hourly_use_per_order = sum(
            item_forecast.get(r["menu_item_id"], 0) / total_orders * r["qty_per_order"]
            for r in context["recipes"] if r["ingredient_id"] == ingredient["id"]
        ) * (1 + ingredient["normal_loss_pct"])
        stock, elapsed = ingredient["stock"], 0.0
        for hour in context["forecast"]["per_hour"]:
            use = hour["expected"] * hourly_use_per_order
            if use > 0 and use >= stock:
                return elapsed + stock / use * 60
            stock -= use
            elapsed += 60
        return None

    def _assess_inventory(self, context: dict) -> list[dict]:
        item_forecast = self._available_item_forecast(context)
        consumption = calculations.ingredient_requirements(item_forecast, context["recipes"])
        horizon_minutes = context["forecast"]["horizon_hours"] * 60
        start_hour = context["current"]["current_hour"]
        items_by_ingredient: dict[int, list[int]] = {}
        for recipe in context["recipes"]:
            items_by_ingredient.setdefault(recipe["ingredient_id"], []).append(recipe["menu_item_id"])

        results = []
        for ingredient in context["inventory"]:
            used = consumption.get(ingredient["id"], 0.0)
            calc = calculations.replenishment_quantity(
                used, ingredient["normal_loss_pct"], ingredient["safety_stock"], ingredient["stock"]
            )
            needed_for_demand = calc["consumption"] + calc["normal_loss"]
            if ingredient["stock"] < needed_for_demand:
                status = "stockout_risk"
            elif calc["net_requirement"] > 0:
                status = "safety_stock_risk"
            else:
                status = "ok"
            runout = self._runout_minutes(ingredient, context, item_forecast)
            results.append({
                "ingredient_id": ingredient["id"],
                "name": ingredient["name"],
                "runs_out_at": self._clock(start_hour, runout) if runout is not None else None,
                "runs_out_in_minutes": round(runout) if runout is not None else None,
                "restock_at": self._clock(start_hour, ingredient["lead_time_minutes"]),
                "unit": ingredient["unit"],
                "stock": ingredient["stock"],
                "safety_stock": ingredient["safety_stock"],
                **calc,
                "status": status,
                "arrives_in_time": ingredient["lead_time_minutes"] <= horizon_minutes,
                "lead_time_minutes": ingredient["lead_time_minutes"],
                "supplier_id": ingredient["supplier_id"],
                "supplier_name": ingredient["supplier_name"],
                "unit_cost": ingredient["unit_cost"],
                "max_order_qty": ingredient["max_order_qty"],
                "used_by_items": items_by_ingredient.get(ingredient["id"], []),
            })
        return results

    def _dish_availability(self, inventory: list[dict], context: dict) -> list[dict]:
        """How many expected orders of each dish the current stock can cover.

        Stock shared by several dishes is split in proportion to their expected demand, so a dish
        can serve (expected orders x coverage of its scarcest ingredient).
        """
        by_id = {i["ingredient_id"]: i for i in inventory}
        names = {m["id"]: m["name"] for m in context["kitchen"]["menu_items"]}
        dishes = []
        for item_id, expected in self._available_item_forecast(context).items():
            limiting, coverage = None, 1.0
            for recipe in context["recipes"]:
                if recipe["menu_item_id"] != item_id:
                    continue
                ing = by_id[recipe["ingredient_id"]]
                need = ing["consumption"] + ing["normal_loss"]
                if need > 0 and ing["stock"] / need < coverage:
                    limiting, coverage = ing, ing["stock"] / need
            expected_orders = round(expected)
            can_serve = expected_orders if limiting is None else min(expected_orders, int(expected * coverage))
            dishes.append({
                "menu_item_id": item_id, "dish": names[item_id], "expected_orders": expected_orders,
                "can_serve": can_serve, "cannot_serve": expected_orders - can_serve,
                "short_ingredient": limiting["name"] if limiting else None,
                "short_by": round(limiting["consumption"] + limiting["normal_loss"] - limiting["stock"], 2)
                if limiting else 0,
                "unit": limiting["unit"] if limiting else None,
                "runs_out_at": limiting["runs_out_at"] if limiting else None,
                "restock_at": limiting["restock_at"] if limiting else None,
                "restock_in_time": bool(limiting and limiting["runs_out_in_minutes"] is not None
                                        and limiting["lead_time_minutes"] <= limiting["runs_out_in_minutes"]),
            })
        return sorted(dishes, key=lambda d: d["cannot_serve"], reverse=True)

    def _assess_kitchen(self, context: dict) -> list[dict]:
        forecast = context["forecast"]
        item_forecast = self._available_item_forecast(context)
        total_forecast = sum(item_forecast.values()) or 1
        items = context["kitchen"]["menu_items"]
        queued = context["current"]["queued_orders"]
        peak_hour_orders = max((h["expected"] for h in forecast["per_hour"]), default=0)

        results = []
        for station in context["kitchen"]["stations"]:
            station_items = [i["id"] for i in items if i["station_id"] == station["id"]]
            share = sum(item_forecast.get(i, 0) for i in station_items) / total_forecast
            # Queued orders are added to the first (peak-risk) hour.
            peak_load = share * (peak_hour_orders + queued)
            utilisation = peak_load / station["capacity_per_hour"]
            if utilisation > 1.0:
                status = "bottleneck"
            elif utilisation >= settings.capacity_warning_utilisation:
                status = "warning"
            else:
                status = "ok"
            results.append({
                "station_id": station["id"], "name": station["name"],
                "capacity_per_hour": station["capacity_per_hour"],
                "peak_load_per_hour": round(peak_load, 1),
                "utilisation_pct": round(utilisation * 100, 1),
                "status": status,
                "items": station_items,
            })
        return results

    # --- Act --------------------------------------------------------------------------
    def act(self, conclusions: dict) -> dict:
        inventory = conclusions["inventory"]
        kitchen = conclusions["kitchen"]
        context = conclusions["context"]
        item_names = {i["id"]: i["name"] for i in context["kitchen"]["menu_items"]}
        dishes = self._dish_availability(inventory, context)
        proposals: list[dict] = []
        notes: list[str] = []

        for ing in inventory:
            if ing["order_qty"] <= 0:
                continue
            quantity = min(ing["order_qty"], ing["max_order_qty"])
            if quantity < ing["order_qty"]:
                notes.append(f"{ing['name']} order capped at max order quantity {ing['max_order_qty']}")
            timing = (f"arrives about {ing['restock_at']}, before it runs out at {ing['runs_out_at']}"
                      if ing["runs_out_at"] and ing["lead_time_minutes"] <= ing["runs_out_in_minutes"]
                      else f"arrives about {ing['restock_at']}"
                      + (f", after it runs out at {ing['runs_out_at']}" if ing["runs_out_at"] else ""))
            proposals.append({
                "action_type": policy_engine.PURCHASE_ORDER,
                "description": (f"Reorder {quantity} {ing['unit']} {ing['name']} from {ing['supplier_name']} "
                                f"now ({timing})"),
                "payload": {
                    "ingredient_id": ing["ingredient_id"], "ingredient": ing["name"],
                    "quantity": quantity, "unit": ing["unit"], "unit_cost": ing["unit_cost"],
                    "total_cost": round(quantity * ing["unit_cost"], 2),
                    "supplier_id": ing["supplier_id"], "supplier": ing["supplier_name"],
                    "lead_time_minutes": ing["lead_time_minutes"],
                    "reason": ing["status"],
                },
            })

        for dish in dishes:
            if dish["cannot_serve"] > 0 and not dish["restock_in_time"]:
                proposals.append({
                    "action_type": policy_engine.MENU_THROTTLE,
                    "description": (f"Stop taking {dish['dish']} orders after about {dish['can_serve']} more: "
                                    f"{dish['short_ingredient']} runs out around {dish['runs_out_at']} and the "
                                    f"next delivery arrives about {dish['restock_at']}"),
                    "payload": {"menu_item_id": dish["menu_item_id"], "menu_item": dish["dish"],
                                "ingredient": dish["short_ingredient"], "can_serve": dish["can_serve"]},
                })

        for station in kitchen:
            if station["status"] == "bottleneck":
                proposals.append({
                    "action_type": policy_engine.ADVISORY,
                    "description": (f"{station['name']} at {station['utilisation_pct']:.0f}% of capacity: "
                                    "move one cook from the least-loaded station and start batch prep now"),
                    "payload": {"station_id": station["station_id"]},
                })

        priorities = sorted(
            ((item_names[int(i)], qty) for i, qty in context["forecast"]["per_item"].items()),
            key=lambda pair: pair[1], reverse=True,
        )
        proposals.append({
            "action_type": policy_engine.ADVISORY,
            "description": "Prep priority: " + ", ".join(f"{name} (~{qty:.0f})" for name, qty in priorities),
            "payload": {"type": "prep_priority"},
        })

        shortages = [i for i in inventory if i["status"] != "ok"]
        bottlenecks = [s for s in kitchen if s["status"] == "bottleneck"]
        stockouts = [i for i in inventory if i["status"] == "stockout_risk"]
        if bottlenecks or any(not i["arrives_in_time"] for i in stockouts):
            sla_risk = "high"
        elif shortages or any(s["status"] == "warning" for s in kitchen):
            sla_risk = "medium"
        else:
            sla_risk = "low"

        for index, proposal in enumerate(proposals, start=1):
            proposal["candidate_id"] = f"A{index}"
            proposal["mandatory"] = (
                proposal["action_type"] == policy_engine.MENU_THROTTLE
                or (proposal["action_type"] == policy_engine.PURCHASE_ORDER
                    and proposal["payload"]["reason"] == "stockout_risk")
            )

        knowledge, knowledge_error = self._retrieve_knowledge(inventory, kitchen, context["forecast"])
        result = {
            "inventory": inventory, "kitchen": kitchen, "sla_risk": sla_risk,
            "proposed_actions": proposals, "notes": notes, "mode": "rule_based", "dishes": dishes,
            "risk_summary": "", "prep_plan": "", "dropped_actions": [], "llm_fallback": None,
            "knowledge": [{k: h[k] for k in ("id", "doc_id", "title", "section", "score")} for h in knowledge],
            "knowledge_error": knowledge_error,
            "citations": [h["id"] for h in knowledge],
        }
        self._knowledge_text = {h["id"]: h["text"] for h in knowledge}
        if self.llm is not None:
            try:
                result = self._llm_review(result, context["forecast"])
            except Exception as error:  # noqa: BLE001 - any LLM problem falls back to deterministic output
                result["llm_fallback"] = str(error)

        result["headline"] = (f"{len(shortages)} inventory risk(s), {len(bottlenecks)} kitchen bottleneck(s), "
                              f"SLA risk {result['sla_risk']}")
        return result

    def _retrieve_knowledge(self, inventory: list[dict], kitchen: list[dict], forecast: dict):
        """RAG: build search queries from the risks found and retrieve matching SOP sections."""
        queries = []
        for item in inventory:
            if item["status"] == "ok":
                continue
            query = f"{item['name']} {item['status'].replace('_', ' ')} shortage purchase order supplier lead time"
            if not item["arrives_in_time"]:
                query += " delivery cannot arrive in time pause menu dish"
            queries.append(query)
        for station in kitchen:
            if station["status"] != "ok":
                queries.append(f"{station['name']} {station['utilisation_pct']:.0f}% of capacity peak hour "
                               "station overload staff batch preparation")
        if forecast.get("spike"):
            queries.append("demand spike peak hour signals sports match rain preparation delivery promise")
        if not queries:
            return [], None
        try:
            return call_tool(self.trace, "knowledge_base.search", knowledge_base.search_many, queries), None
        except knowledge_base.KnowledgeBaseError as error:
            return [], str(error)  # RAG is helpful but not critical: continue without it

    def _llm_review(self, result: dict, forecast: dict) -> dict:
        """Let the LLM select and explain candidate actions. Mandatory ones are always kept."""
        payload = {
            "forecast": {k: forecast[k] for k in ("horizon_hours", "expected_orders", "normal_orders",
                                                  "uplift_pct", "spike", "confidence", "reason_codes")},
            "inventory_risks": [
                {k: i[k] for k in ("name", "unit", "stock", "consumption", "normal_loss", "safety_stock",
                                   "net_requirement", "status", "arrives_in_time", "lead_time_minutes")}
                for i in result["inventory"] if i["status"] != "ok"
            ],
            "kitchen": [{k: s[k] for k in ("name", "capacity_per_hour", "peak_load_per_hour",
                                           "utilisation_pct", "status")} for s in result["kitchen"]],
            "orders_we_cannot_fulfil": [d for d in result["dishes"] if d["cannot_serve"] > 0],
            "computed_sla_risk": result["sla_risk"],
            "candidate_actions": [
                {"id": p["candidate_id"], "type": p["action_type"], "description": p["description"],
                 "mandatory": p["mandatory"]} for p in result["proposed_actions"]
            ],
        }
        knowledge = [{"id": k["id"], "title": k["title"], "section": k["section"],
                      "text": self._knowledge_text.get(k["id"], "")} for k in result["knowledge"]]
        user = ("Assess the situation and choose the actions. Reply with the JSON object.\n"
                f"<operations_context>{json.dumps(payload)}</operations_context>\n"
                f"<knowledge>{json.dumps(knowledge)}</knowledge>")
        answer = call_llm(self.trace, self.usage, self.name, self.llm,
                          prompts.operations_system(has_knowledge=bool(knowledge)), user, "action selection")

        selected = answer.get("selected_action_ids", [])
        if not isinstance(selected, list):
            raise ValueError("selected_action_ids must be a list")
        selected_ids = {str(i) for i in selected}
        notes = answer.get("action_notes") if isinstance(answer.get("action_notes"), dict) else {}

        kept, dropped = [], []
        for proposal in result["proposed_actions"]:
            note = str(notes.get(proposal["candidate_id"], ""))[:300]
            if note:
                proposal["payload"]["rationale"] = note
            if proposal["mandatory"] or proposal["candidate_id"] in selected_ids:
                kept.append(proposal)
            else:
                dropped.append({"id": proposal["candidate_id"], "description": proposal["description"],
                                "note": note})

        llm_risk = str(answer.get("sla_risk", "")).lower()
        if llm_risk in RISK_ORDER and RISK_ORDER[llm_risk] > RISK_ORDER[result["sla_risk"]]:
            result["sla_risk"] = llm_risk  # the LLM may raise the risk level, never lower it

        retrieved_ids = {k["id"] for k in result["knowledge"]}
        cited = answer.get("citations") if isinstance(answer.get("citations"), list) else []
        result.update({
            "citations": [str(c) for c in cited if str(c) in retrieved_ids],  # only real, retrieved sources
            "proposed_actions": kept, "dropped_actions": dropped, "mode": "llm",
            "risk_summary": str(answer.get("risk_summary", ""))[:600],
            "prep_plan": str(answer.get("prep_plan", ""))[:400],
        })
        return result
