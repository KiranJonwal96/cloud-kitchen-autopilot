"""Input guardrails for user requests: validation, prompt-injection detection and intent parsing.

User text is always treated as data. It can ask a question, but it can never change
policy, rules or the agents' instructions.
"""

import re

MAX_REQUEST_LENGTH = 500

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)",
    r"disregard (the |all )?(rules|policy|policies|instructions)",
    r"(bypass|skip|disable|override) (the )?(approval|policy|policies|rules|limits?|guardrails?)",
    r"without (human |supervisor )?approval",
    r"auto[- ]?approve",
    r"you are now",
    r"system prompt",
    r"(drop|delete|truncate) (table|database|all)",
]

DOMAIN_KEYWORDS = [
    "order", "demand", "surge", "spike", "rush", "busy", "peak", "handle", "cope", "manage",
    "stock", "inventory", "ingredient", "capacity", "kitchen", "prepare", "prep", "evening",
    "tonight", "dinner", "lunch", "next", "hour", "plan", "replan", "shortage", "run out",
]


def check_request(text: str) -> dict:
    """Return {ok, status, reason, cleaned}. status is 'ok', 'refused' or 'needs_clarification'."""
    cleaned = (text or "").strip()
    if not cleaned:
        return {"ok": False, "status": "needs_clarification", "reason": "Request is empty.",
                "cleaned": cleaned}
    if len(cleaned) > MAX_REQUEST_LENGTH:
        return {"ok": False, "status": "refused",
                "reason": f"Request is longer than {MAX_REQUEST_LENGTH} characters.", "cleaned": cleaned}
    lowered = cleaned.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return {"ok": False, "status": "refused",
                    "reason": ("The request tries to change the system's rules or instructions. "
                               "These cannot be changed from a request."),
                    "cleaned": cleaned}
    if not any(keyword in lowered for keyword in DOMAIN_KEYWORDS):
        return {"ok": False, "status": "needs_clarification",
                "reason": ("I can help with near-term demand, inventory and kitchen capacity. "
                           "Could you say what you want to check and for how many hours? "
                           "For example: 'Can we handle demand for the next 2 hours?'"),
                "cleaned": cleaned}
    return {"ok": True, "status": "ok", "reason": "", "cleaned": cleaned}


def parse_horizon(text: str, explicit_hours: int | None, current_hour: int,
                  default_hours: int, max_hours: int) -> tuple[int, list[str]]:
    """Work out the forecast horizon in hours and list any assumptions made."""
    assumptions: list[str] = []
    if explicit_hours:
        hours = int(explicit_hours)
    else:
        lowered = text.lower()
        match = re.search(r"(\d+)\s*(hours|hour|hrs|hr|h)\b", lowered)
        word_numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        word_match = re.search(r"\b(one|two|three|four|five|six)\s+hours?\b", lowered)
        if match:
            hours = int(match.group(1))
        elif word_match:
            hours = word_numbers[word_match.group(1)]
        elif "tonight" in lowered or "rest of the evening" in lowered:
            hours = max(1, 24 - current_hour)
            assumptions.append(f"'Tonight' interpreted as until closing ({hours}h).")
        else:
            hours = default_hours
            assumptions.append(f"No time window given; assumed the next {default_hours} hours.")
    if hours < 1:
        hours = 1
        assumptions.append("Horizon below 1 hour; using 1 hour.")
    if hours > max_hours:
        assumptions.append(f"Horizon limited to {max_hours} hours (near-term forecasting only).")
        hours = max_hours
    return hours, assumptions
