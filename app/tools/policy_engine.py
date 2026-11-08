"""Policy / Rule Engine: deterministic business rules, checked on every suggestion.

The system only suggests; the supervisor decides. The policy engine makes sure a suggestion
never breaks a company rule, and flags it (and escalates the decision) when it would.
"""

from app.config import settings

PURCHASE_ORDER = "PURCHASE_ORDER"   # reorder an ingredient
MENU_THROTTLE = "MENU_THROTTLE"     # stop or limit a dish
ADVISORY = "ADVISORY"               # kitchen staffing / prep advice


def validate_action(conn, action_type: str, payload: dict, planned_spend: float = 0.0) -> dict:
    """Return {allowed, violations, checks} for one suggestion.

    planned_spend is the cost of the other reorder suggestions in the same plan, so the daily
    budget is checked against the plan as a whole.
    """
    violations: list[str] = []
    checks: list[str] = []

    if action_type == PURCHASE_ORDER:
        ingredient = conn.execute(
            "SELECT i.max_order_qty, s.approved FROM ingredients i "
            "JOIN suppliers s ON s.id = i.supplier_id WHERE i.id = ?",
            (payload["ingredient_id"],),
        ).fetchone()
        if ingredient is None:
            violations.append("Unknown ingredient")
        else:
            if payload["quantity"] > ingredient["max_order_qty"]:
                violations.append(f"Quantity {payload['quantity']} exceeds max order quantity "
                                  f"{ingredient['max_order_qty']}")
            else:
                checks.append("Within max order quantity")
            if not ingredient["approved"]:
                violations.append("Supplier is not on the approved list")
            else:
                checks.append("Supplier approved")

        cost = payload["total_cost"]
        if cost > settings.max_spend_per_po:
            violations.append(f"Order value {cost:.0f} exceeds the per-order limit {settings.max_spend_per_po:.0f}")
        else:
            checks.append("Within per-order spend limit")
        if planned_spend + cost > settings.daily_procurement_budget:
            violations.append(f"Daily procurement budget {settings.daily_procurement_budget:.0f} would be exceeded")
        else:
            checks.append("Within daily procurement budget")

    elif action_type == MENU_THROTTLE:
        if conn.execute("SELECT 1 FROM menu_items WHERE id = ?", (payload["menu_item_id"],)).fetchone():
            checks.append("Menu item exists")
        else:
            violations.append("Unknown menu item")

    elif action_type != ADVISORY:
        violations.append(f"Unsupported suggestion type {action_type}")

    return {"allowed": not violations, "violations": violations, "checks": checks}
