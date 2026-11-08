"""Deterministic data-retrieval tools.

Each tool stands in for an external API (Order, Inventory, Kitchen, Weather, Events).
Orders, inventory and kitchen data are simulated in SQLite. Weather and holidays can be LIVE:
- WEATHER_SOURCE=open_meteo   -> real weather from Open-Meteo (free, no key)
- HOLIDAY_SOURCE=calendarific -> real public holidays and festivals from Calendarific (free key)
Otherwise they come from the demo situation buttons. The UI's "data down" situations make a tool
fail so the fallback and escalation paths can be shown.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime

from app.config import settings
from app.db import DEFAULT_SIM_STATE, HOURLY_BASE_ORDERS, WEEKDAY_FACTOR, get_connection

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CALENDARIFIC_URL = "https://calendarific.com/api/v2/holidays"
RAIN_WEATHER_CODES = set(range(51, 68)) | set(range(80, 83)) | set(range(95, 100))  # WMO rain codes
WEATHER_CACHE_SECONDS = 600

_weather_cache: dict = {}   # {"at": timestamp, "value": dict}
_holiday_cache: dict = {}   # {(country, location, date): [holiday names]}


class ToolError(Exception):
    """Raised when a tool/API is unavailable or returns unusable data."""

    def __init__(self, tool_name: str, message: str):
        super().__init__(f"{tool_name}: {message}")
        self.tool_name = tool_name


# --- Simulation state (stands in for the live operating environment) ---------------------

def get_sim_state() -> dict:
    """The kitchen's date and hour, and which incidents are switched on.

    Date and hour come from the system clock, unless a date or hour is being simulated.
    """
    with get_connection() as conn:
        row = conn.execute("SELECT state_json FROM sim_state WHERE id = 1").fetchone()
    stored = json.loads(row["state_json"])
    incidents = {**DEFAULT_SIM_STATE["incidents"], **(stored.get("incidents") or {})}
    state = {**DEFAULT_SIM_STATE, **stored, "incidents": incidents}
    now = datetime.now()
    state["business_date"] = state["simulated_date"] or now.date().isoformat()
    state["current_hour"] = now.hour if state["simulated_hour"] is None else int(state["simulated_hour"])
    return state


def update_sim_state(changes: dict) -> dict:
    """Change the simulated date and hour and/or incidents, e.g. {"incidents": {"rain": True}}.

    {"simulated_date": "2026-11-08", "simulated_hour": 19} simulates that day and hour;
    None for either goes back to the system clock.
    """
    state = get_sim_state()
    for key in ("business_date", "current_hour"):
        state.pop(key)  # worked out from the clock on every read, never stored
    if "simulated_hour" in changes:
        hour = changes["simulated_hour"]
        state["simulated_hour"] = None if hour is None else int(hour)
    if "simulated_date" in changes:
        day = changes["simulated_date"]
        state["simulated_date"] = date.fromisoformat(str(day)).isoformat() if day else None
    for name, flag in (changes.get("incidents") or {}).items():
        if name in state["incidents"]:
            state["incidents"][name] = bool(flag)
    with get_connection() as conn:
        conn.execute("UPDATE sim_state SET state_json = ? WHERE id = 1", (json.dumps(state),))
    return get_sim_state()


def _check_outage(state: dict, incident: str, tool_name: str) -> None:
    if state["incidents"].get(incident):
        raise ToolError(tool_name, "service unavailable (simulated outage)")


# --- Retrieval tools -------------------------------------------------------------------

def get_current_orders() -> dict:
    """Order API: current operating hour, recent order rate and kitchen queue."""
    state = get_sim_state()
    business_date = date.fromisoformat(state["business_date"])
    rush = state["incidents"]["order_rush"]
    # Demo order feed: a typical last hour for this weekday and time (38 on a Saturday at 19:00).
    previous_hour = HOURLY_BASE_ORDERS.get(int(state["current_hour"]) - 1, 0)
    normal = round(previous_hour * WEEKDAY_FACTOR[business_date.weekday()] * 1.05)
    return {
        "business_date": state["business_date"],
        "weekday": business_date.weekday(),
        "current_hour": int(state["current_hour"]),
        "last_hour_orders": round(normal * 1.7) if rush else normal,
        "queued_orders": 18 if rush else 10,
    }


def get_historical_orders(weekday: int, hours: list[int]) -> list[dict]:
    """Order history DB: per-day, per-hour, per-item orders for the given weekday and hours."""
    placeholders = ",".join("?" for _ in hours)
    query = (
        "SELECT order_date, hour, menu_item_id, orders, weather, event, festival "
        f"FROM historical_orders WHERE weekday = ? AND hour IN ({placeholders})"
    )
    with get_connection() as conn:
        rows = conn.execute(query, (weekday, *hours)).fetchall()
    return [dict(row) for row in rows]


def get_all_history_for_hours(hours: list[int]) -> list[dict]:
    """Order history across all weekdays, used to estimate demand-driver effects."""
    placeholders = ",".join("?" for _ in hours)
    query = (
        "SELECT order_date, weekday, hour, SUM(orders) AS orders, weather, event, festival "
        f"FROM historical_orders WHERE hour IN ({placeholders}) "
        "GROUP BY order_date, hour"
    )
    with get_connection() as conn:
        rows = conn.execute(query, tuple(hours)).fetchall()
    return [dict(row) for row in rows]


def get_inventory() -> list[dict]:
    """Inventory API: current stock with safety stock, loss rate and supplier terms."""
    _check_outage(get_sim_state(), "inventory_api_down", "inventory_api")
    query = """
        SELECT i.id, i.name, i.unit, i.stock, i.safety_stock, i.unit_cost, i.normal_loss_pct,
               i.max_order_qty, s.id AS supplier_id, s.name AS supplier_name,
               s.lead_time_minutes, s.approved AS supplier_approved
        FROM ingredients i JOIN suppliers s ON s.id = i.supplier_id
        ORDER BY i.id
    """
    with get_connection() as conn:
        return [dict(row) for row in conn.execute(query).fetchall()]


def get_kitchen_capacity() -> dict:
    """Kitchen API: stations, hourly capacity and which menu items each station prepares."""
    with get_connection() as conn:
        stations = [dict(r) for r in conn.execute("SELECT * FROM stations ORDER BY id").fetchall()]
        items = [dict(r) for r in conn.execute("SELECT * FROM menu_items ORDER BY id").fetchall()]
    return {"stations": stations, "menu_items": items}


def get_recipes() -> list[dict]:
    with get_connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM recipes").fetchall()]


def _get_json(url: str, tool_name: str, timeout: float = 10) -> dict:
    """GET a JSON API. Errors never include the URL, so API keys can't leak into logs."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise ToolError(tool_name, f"HTTP {error.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise ToolError(tool_name, f"unavailable ({type(error).__name__})") from None


def get_weather() -> dict:
    """Weather API: is it raining now or likely in the next 3 hours ("rain") or not ("clear").

    Order of precedence: simulated outage -> simulated rain incident -> live Open-Meteo -> demo (clear).
    """
    state = get_sim_state()
    _check_outage(state, "weather_api_down", "weather_api")
    if state["incidents"]["rain"]:
        return {"condition": "rain", "source": "simulated incident"}
    if settings.weather_source != "open_meteo":
        return {"condition": "clear", "source": "demo"}

    cached = _weather_cache.get("value")
    if cached and time.time() - _weather_cache["at"] < WEATHER_CACHE_SECONDS:
        return cached
    query = urllib.parse.urlencode({
        "latitude": settings.kitchen_latitude, "longitude": settings.kitchen_longitude,
        "current": "precipitation,weather_code,temperature_2m",
        "hourly": "precipitation_probability", "forecast_hours": 3,
    })
    data = _get_json(f"{OPEN_METEO_URL}?{query}", "weather_api")
    current = data.get("current") or {}
    rain_chance = max((p or 0 for p in (data.get("hourly") or {}).get("precipitation_probability") or [0]))
    raining = (current.get("precipitation") or 0) > 0.1 or current.get("weather_code") in RAIN_WEATHER_CODES
    value = {
        "condition": "rain" if raining or rain_chance >= 60 else "clear",
        "source": "Open-Meteo (live)",
        "temperature_c": current.get("temperature_2m"),
        "rain_chance_next_3h_pct": rain_chance,
    }
    _weather_cache.update({"at": time.time(), "value": value})
    return value


def get_holidays(day: str) -> list[str]:
    """Calendarific: the main festivals and national holidays on this date (cached per day).

    Only type=national is asked for (e.g. Diwali, Holi, Eid, Christmas, Independence Day), so small
    observances and optional/restricted holidays do not count as a festival.
    """
    key = (settings.holiday_country, settings.holiday_location, day)
    if key in _holiday_cache:
        return _holiday_cache[key]
    year, month, dom = day.split("-")
    params = {"api_key": settings.calendarific_api_key, "country": settings.holiday_country,
              "year": int(year), "month": int(month), "day": int(dom), "type": "national"}
    if settings.holiday_location:
        params["location"] = settings.holiday_location
    data = _get_json(f"{CALENDARIFIC_URL}?{urllib.parse.urlencode(params)}", "events_api")
    if (data.get("meta") or {}).get("code") != 200:
        raise ToolError("events_api", f"Calendarific error {(data.get('meta') or {}).get('error_type', '')}")
    response = data.get("response") or {}
    holidays = response.get("holidays", []) if isinstance(response, dict) else []
    names = [h["name"] for h in holidays if str((h.get("date") or {}).get("iso", ""))[:10] == day]
    _holiday_cache[key] = names
    return names


def get_events() -> dict:
    """Events data: a major sports event (incident switch) and whether today is a holiday/festival.

    Order of precedence: simulated outage -> simulated festival incident -> live Calendarific -> demo (none).
    """
    state = get_sim_state()
    incidents = state["incidents"]
    _check_outage(state, "holiday_api_down", "events_api")
    event = "cricket_match" if incidents["cricket_match"] else "none"
    if incidents["festival"]:
        return {"event": event, "festival": True, "holidays": ["Festival (simulated)"], "source": "simulated incident"}
    if settings.holiday_source == "calendarific" and settings.calendarific_api_key:
        holidays = get_holidays(state["business_date"])
        return {"event": event, "festival": bool(holidays), "holidays": holidays, "source": "Calendarific (live)"}
    return {"event": event, "festival": False, "holidays": [], "source": "demo"}


def call_with_retry(func, retries: int, *args, **kwargs):
    """Call a tool, retrying on ToolError. Re-raises the last error when retries run out."""
    last_error = None
    for _ in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except ToolError as error:
            last_error = error
    raise last_error
