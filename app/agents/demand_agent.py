"""Demand Agent: near-term demand forecast and demand-spike detection.

Perceive: order history, current orders, weather and events (tools).
Reason:   a statistical forecast tool computes the baseline; the LLM reviews it and may
          apply a bounded adjustment, set confidence and explain the drivers.
Act:      returns forecast, spike flag, confidence and reason codes to the Orchestrator.
Learn:    bias correction from recorded outcomes, clamped to safe bounds.

Boundaries: forecasts only. It does not decide replenishment, suppliers, kitchen
priorities or execute anything.
"""

import json
import statistics
from concurrent.futures import ThreadPoolExecutor

from app.agents.base import AgentError, BaseAgent, call_llm, call_tool
from app.config import settings
from app.llm import prompts
from app.tools import data_tools, memory_store

OPENING_HOUR = 10  # first hour in which orders are accepted
CLOSING_HOUR = 23  # last hour in which orders are accepted
DEFAULT_DRIVER_EFFECTS = {"rain": 1.2, "cricket_match": 1.35, "festival": 1.25}
MIN_DRIVER_SAMPLES = 3
LLM_MIN_ADJUSTMENT = 0.8
LLM_MAX_ADJUSTMENT = 1.25


class DemandAgent(BaseAgent):
    name = "demand_agent"

    def __init__(self, trace: list[dict], llm=None, usage=None):
        self.trace = trace
        self.llm = llm
        self.usage = usage

    # --- Learn ------------------------------------------------------------------------
    def learn(self) -> dict:
        """Bias correction from recorded outcomes (actual / forecast), clamped to safe bounds."""
        outcomes = memory_store.get_forecast_outcomes(limit=10)
        ratios = [o["actual_orders"] / o["forecast_orders"] for o in outcomes if o["forecast_orders"] > 0]
        if len(ratios) < 2:
            return {"bias_factor": 1.0, "samples": len(ratios), "clamped": False}
        raw = statistics.mean(ratios)
        factor = min(settings.learning_max_factor, max(settings.learning_min_factor, raw))
        return {"bias_factor": round(factor, 3), "raw_factor": round(raw, 3),
                "samples": len(ratios), "clamped": factor != raw}

    @staticmethod
    def _recent_accuracy() -> dict:
        """Recent forecast error and bias, used to build the dynamic part of the prompt."""
        outcomes = [o for o in memory_store.get_forecast_outcomes(limit=10) if o["actual_orders"] > 0]
        if not outcomes:
            return {"samples": 0}
        errors = [abs(o["actual_orders"] - o["forecast_orders"]) / o["actual_orders"] for o in outcomes]
        bias = [(o["actual_orders"] - o["forecast_orders"]) / o["actual_orders"] for o in outcomes]
        return {"samples": len(outcomes), "mape_pct": 100 * statistics.mean(errors),
                "bias_pct": 100 * statistics.mean(bias)}

    # --- Perceive ---------------------------------------------------------------------
    def perceive(self, task: dict) -> dict:
        current = call_tool(self.trace, "order_api.current", data_tools.get_current_orders)
        start_hour = current["current_hour"]
        hours = [h for h in range(start_hour, start_hour + task["horizon_hours"]) if OPENING_HOUR <= h <= CLOSING_HOUR]
        if not hours:
            raise AgentError(f"The kitchen is closed at {start_hour:02d}:00 (open 10:00-23:00). "
                             "Pick a time in 'Simulate a time' to plan for an open hour")
        history_hours = sorted(set(hours + [max(start_hour - 1, 0)]))

        # Independent retrievals run in parallel; demand drivers may fail and fall back.
        with ThreadPoolExecutor(max_workers=4) as pool:
            history_future = pool.submit(call_tool, self.trace, "order_history_db",
                                         data_tools.get_all_history_for_hours, history_hours)
            item_future = pool.submit(call_tool, self.trace, "order_history_db.items",
                                      data_tools.get_historical_orders, current["weekday"], hours)
            weather_future = pool.submit(self._optional_tool, "weather_api", data_tools.get_weather)
            events_future = pool.submit(self._optional_tool, "events_api", data_tools.get_events)
            history = history_future.result()
            item_history = item_future.result()
            weather = weather_future.result()
            events = events_future.result()

        if not history:
            raise AgentError("No historical order data available")
        return {
            "task": task, "current": current, "hours": hours,
            "requested_horizon": task["horizon_hours"],
            "history": history, "item_history": item_history,
            "weather": weather, "events": events, "learning": self.learn(),
        }

    def _optional_tool(self, name: str, func):
        try:
            return call_tool(self.trace, name, data_tools.call_with_retry, func, settings.tool_max_retries)
        except data_tools.ToolError:
            return None

    # --- Reason -----------------------------------------------------------------------
    def reason(self, context: dict) -> dict:
        conclusions = self._statistical_forecast(context)
        conclusions["mode"] = "rule_based"
        if self.llm is not None:
            try:
                conclusions = self._llm_review(context, conclusions)
            except Exception as error:  # noqa: BLE001 - any LLM problem falls back to statistics
                conclusions["llm_fallback"] = str(error)
        return conclusions

    def _llm_review(self, context: dict, stats: dict) -> dict:
        """Ask the LLM to review the statistical forecast; validate and bound its answer."""
        payload = {
            "current_time": f"{context['current']['business_date']} {context['current']['current_hour']}:00",
            "weekday": context["current"]["weekday"],
            "weather": context["weather"] if context["weather"] else "UNKNOWN (weather API failed)",
            "events": context["events"] if context["events"] else "UNKNOWN (events API failed)",
            "last_hour_orders": context["current"]["last_hour_orders"],
            "statistical_forecast": {
                "per_hour": stats["per_hour"],
                "expected_total": round(stats["expected_total"], 1),
                "normal_total": round(stats["normal_total"], 1),
                "uplift_pct": round((stats["uplift"] - 1) * 100, 1),
                "factors": stats["factors"],
                "statistical_confidence": stats["confidence"],
                "reason_codes": stats["reason_codes"],
            },
        }
        user = ("Review this statistical forecast and reply with the JSON object.\n"
                f"<forecast_context>{json.dumps(payload)}</forecast_context>")
        answer = call_llm(self.trace, self.usage, self.name, self.llm,
                          prompts.demand_system(self._recent_accuracy()), user, "forecast review")

        try:
            requested = float(answer.get("adjustment_factor", 1.0))
            llm_confidence = float(answer.get("confidence", stats["confidence"]))
        except (TypeError, ValueError) as error:
            raise ValueError("LLM returned non-numeric adjustment or confidence") from error
        factor = min(LLM_MAX_ADJUSTMENT, max(LLM_MIN_ADJUSTMENT, requested))
        llm_confidence = min(1.0, max(0.0, llm_confidence))

        reviewed = dict(stats)
        reviewed["per_hour"] = [{**h, "expected": round(h["expected"] * factor, 1)} for h in stats["per_hour"]]
        reviewed["per_item"] = {k: round(v * factor, 1) for k, v in stats["per_item"].items()}
        reviewed["expected_total"] = stats["expected_total"] * factor
        reviewed["uplift"] = (reviewed["expected_total"] / stats["normal_total"]) if stats["normal_total"] else 1.0
        reviewed["spike"] = reviewed["uplift"] >= settings.spike_ratio  # recomputed, not taken from the LLM
        # Conservative: never more confident than the statistics allow.
        reviewed["confidence"] = round(min(llm_confidence, stats["confidence"] + 0.05), 2)
        codes = [str(c).upper()[:40] for c in answer.get("reason_codes", []) if isinstance(c, str)]
        reviewed["reason_codes"] = list(dict.fromkeys(stats["reason_codes"] + codes))
        if factor != 1.0:
            reviewed["reason_codes"].append("LLM_ADJUSTED")
        reviewed["llm_needs_review"] = bool(answer.get("needs_review", False))
        reviewed["reasoning"] = str(answer.get("reasoning", ""))[:600]
        reviewed["factors"] = {**stats["factors"], "llm_adjustment": round(factor, 3),
                               "llm_requested_adjustment": requested,
                               "statistical_expected_total": round(stats["expected_total"], 1)}
        reviewed["mode"] = "llm"
        return reviewed

    def _statistical_forecast(self, context: dict) -> dict:
        """Statistical forecast tool: baseline x demand drivers x momentum x learned bias."""
        current = context["current"]
        weekday = current["weekday"]
        history = context["history"]
        reason_codes: list[str] = []

        baseline_by_slot = self._normal_baseline(history)
        driver_effects = self._estimate_driver_effects(history, baseline_by_slot)

        active_drivers = []
        if context["weather"] is None:
            reason_codes.append("WEATHER_DATA_MISSING")
        elif context["weather"]["condition"] == "rain":
            active_drivers.append("rain")
            reason_codes.append("RAIN_UPLIFT")
        if context["events"] is None:
            reason_codes.append("EVENT_DATA_MISSING")
        else:
            if context["events"]["event"] == "cricket_match":
                active_drivers.append("cricket_match")
                reason_codes.append("EVENT_CRICKET_MATCH")
            if context["events"]["festival"]:
                active_drivers.append("festival")
                reason_codes.append("FESTIVAL")
        if weekday >= 4:
            reason_codes.append("WEEKEND_PATTERN")

        driver_factor = 1.0
        for driver in active_drivers:
            driver_factor *= driver_effects[driver]

        # Momentum: compare the last hour's actual orders with what was expected for it.
        previous_hour = max(current["current_hour"] - 1, 0)
        expected_previous = baseline_by_slot.get((weekday, previous_hour), 0) * driver_factor
        momentum_ratio = 1.0
        if expected_previous > 0:
            momentum_ratio = min(1.6, max(0.8, current["last_hour_orders"] / expected_previous))
        momentum_factor = 1 + 0.5 * (momentum_ratio - 1)
        if momentum_ratio >= 1.15:
            reason_codes.append("MOMENTUM_HIGH")
        elif momentum_ratio <= 0.9:
            reason_codes.append("MOMENTUM_LOW")

        bias = context["learning"]["bias_factor"]
        if bias != 1.0:
            reason_codes.append("LEARNED_BIAS_APPLIED")
        if len(context["hours"]) < context["requested_horizon"]:
            reason_codes.append("HORIZON_TRUNCATED_AT_CLOSING")

        per_hour = []
        normal_total = 0.0
        for hour in context["hours"]:
            normal = baseline_by_slot.get((weekday, hour), 0.0)
            normal_total += normal
            per_hour.append({"hour": hour, "normal": round(normal, 1),
                             "expected": round(normal * driver_factor * momentum_factor * bias, 1)})
        expected_total = sum(h["expected"] for h in per_hour)

        item_share = self._item_share(context["item_history"])
        # String keys keep the forecast plain JSON when it is saved.
        per_item = {str(item_id): round(expected_total * share, 1) for item_id, share in item_share.items()}

        uplift = expected_total / normal_total if normal_total else 1.0
        spike = uplift >= settings.spike_ratio
        confidence = self._confidence(history, weekday, context["hours"], reason_codes, momentum_ratio)

        return {
            "inputs": {"weather": context["weather"], "events": context["events"]},
            "hours": context["hours"], "per_hour": per_hour, "per_item": per_item,
            "expected_total": expected_total, "normal_total": normal_total,
            "uplift": uplift, "spike": spike, "confidence": confidence,
            "reason_codes": reason_codes,
            "factors": {
                "driver_factor": round(driver_factor, 3),
                "driver_effects_used": {d: round(driver_effects[d], 3) for d in active_drivers},
                "momentum_ratio": round(momentum_ratio, 3),
                "momentum_factor": round(momentum_factor, 3),
                "learned_bias": bias,
            },
            "learning": context["learning"],
        }

    @staticmethod
    def _is_normal(row: dict) -> bool:
        return row["weather"] == "clear" and row["event"] == "none" and not row["festival"]

    def _normal_baseline(self, history: list[dict]) -> dict:
        """Average total orders per (weekday, hour) on days without demand drivers."""
        buckets: dict[tuple, list[float]] = {}
        fallback: dict[tuple, list[float]] = {}
        for row in history:
            key = (row["weekday"], row["hour"])
            fallback.setdefault(key, []).append(row["orders"])
            if self._is_normal(row):
                buckets.setdefault(key, []).append(row["orders"])
        baseline = {}
        for key, values in fallback.items():
            normal_values = buckets.get(key, [])
            baseline[key] = statistics.mean(normal_values if len(normal_values) >= 2 else values)
        return baseline

    def _estimate_driver_effects(self, history: list[dict], baseline: dict) -> dict:
        """Estimate each driver's multiplier from history where only that driver was present."""
        effects = {}
        for driver in DEFAULT_DRIVER_EFFECTS:
            ratios = []
            for row in history:
                active = {
                    "rain": row["weather"] == "rain",
                    "cricket_match": row["event"] == "cricket_match",
                    "festival": bool(row["festival"]),
                }
                if active[driver] and sum(active.values()) == 1:
                    normal = baseline.get((row["weekday"], row["hour"]))
                    if normal:
                        ratios.append(row["orders"] / normal)
            effects[driver] = (statistics.mean(ratios) if len(ratios) >= MIN_DRIVER_SAMPLES
                               else DEFAULT_DRIVER_EFFECTS[driver])
        return effects

    @staticmethod
    def _item_share(item_history: list[dict]) -> dict[int, float]:
        totals: dict[int, float] = {}
        for row in item_history:
            totals[row["menu_item_id"]] = totals.get(row["menu_item_id"], 0) + row["orders"]
        grand_total = sum(totals.values()) or 1
        return {item_id: value / grand_total for item_id, value in totals.items()}

    def _confidence(self, history, weekday, hours, reason_codes, momentum_ratio) -> float:
        normal_totals = [r["orders"] for r in history
                         if r["weekday"] == weekday and r["hour"] in hours and self._is_normal(r)]
        confidence = 0.9
        if len(normal_totals) >= 2:
            variation = statistics.pstdev(normal_totals) / (statistics.mean(normal_totals) or 1)
            confidence -= min(0.2, variation * 0.5)
        if len(normal_totals) < 4:
            confidence -= 0.1
        confidence -= 0.15 * sum(code.endswith("_DATA_MISSING") for code in reason_codes)
        if momentum_ratio >= 1.4 or momentum_ratio <= 0.85:
            confidence -= 0.1  # recent orders differ strongly from history: unusual situation
        return round(min(0.95, max(0.1, confidence)), 2)

    # --- Act --------------------------------------------------------------------------
    def act(self, conclusions: dict) -> dict:
        uplift_pct = round((conclusions["uplift"] - 1) * 100, 1)
        needs_review = (conclusions["confidence"] < settings.confidence_threshold
                        or conclusions.get("llm_needs_review", False))
        headline = (
            f"{conclusions['expected_total']:.0f} orders expected over {len(conclusions['hours'])}h "
            f"({uplift_pct:+.0f}% vs normal), spike={conclusions['spike']}, "
            f"confidence={conclusions['confidence']}"
        )
        return {
            "headline": headline,
            "horizon_hours": len(conclusions["hours"]),
            "hours": conclusions["hours"],
            "expected_orders": round(conclusions["expected_total"], 1),
            "normal_orders": round(conclusions["normal_total"], 1),
            "uplift_pct": uplift_pct,
            "spike": conclusions["spike"],
            "confidence": conclusions["confidence"],
            "needs_review": needs_review,
            "per_hour": conclusions["per_hour"],
            "per_item": conclusions["per_item"],
            "reason_codes": conclusions["reason_codes"],
            "factors": conclusions["factors"],
            "learning": conclusions["learning"],
            "inputs": conclusions.get("inputs", {}),
            "mode": conclusions["mode"],
            "reasoning": conclusions.get("reasoning", ""),
            "llm_fallback": conclusions.get("llm_fallback"),
        }
