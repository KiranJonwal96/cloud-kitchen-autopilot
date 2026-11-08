"""Deterministic calculation tools: ingredient requirement, normal loss and replenishment."""

import math


def ingredient_requirements(item_forecast: dict[int, float], recipes: list[dict]) -> dict[int, float]:
    """Return expected consumption per ingredient id for the forecast order counts per menu item."""
    totals: dict[int, float] = {}
    for recipe in recipes:
        orders = item_forecast.get(recipe["menu_item_id"], 0.0)
        totals[recipe["ingredient_id"]] = (
            totals.get(recipe["ingredient_id"], 0.0) + orders * recipe["qty_per_order"]
        )
    return totals


def normal_loss_impact(consumption: float, loss_pct: float) -> float:
    """Extra quantity lost to trimming/spoilage for a given consumption."""
    return consumption * loss_pct


def replenishment_quantity(consumption: float, loss_pct: float, safety_stock: float,
                           current_stock: float, order_step: float = 0.5) -> dict:
    """Net requirement = consumption + normal loss + safety stock - current stock.

    The order quantity is rounded up to the supplier's order step.
    """
    loss = normal_loss_impact(consumption, loss_pct)
    gross_requirement = consumption + loss + safety_stock
    net_requirement = max(0.0, gross_requirement - current_stock)
    order_qty = math.ceil(net_requirement / order_step) * order_step if net_requirement > 0 else 0.0
    return {
        "consumption": round(consumption, 2),
        "normal_loss": round(loss, 2),
        "gross_requirement": round(gross_requirement, 2),
        "net_requirement": round(net_requirement, 2),
        "order_qty": round(order_qty, 2),
    }
