"""SQLite persistence: schema, connection helper and demo seed data.

The workbook design names PostgreSQL. SQLite is used for the local prototype so it runs
with zero setup; the schema uses plain SQL and can be moved to PostgreSQL unchanged in shape.
"""

import json
import random
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS suppliers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    lead_time_minutes INTEGER NOT NULL,
    approved INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS ingredients (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    unit TEXT NOT NULL,
    stock REAL NOT NULL,
    safety_stock REAL NOT NULL,
    unit_cost REAL NOT NULL,
    normal_loss_pct REAL NOT NULL,
    max_order_qty REAL NOT NULL,
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id)
);
CREATE TABLE IF NOT EXISTS stations (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    capacity_per_hour INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS menu_items (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    station_id INTEGER NOT NULL REFERENCES stations(id),
    price REAL NOT NULL,
    available INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS recipes (
    menu_item_id INTEGER NOT NULL REFERENCES menu_items(id),
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    qty_per_order REAL NOT NULL,
    PRIMARY KEY (menu_item_id, ingredient_id)
);
CREATE TABLE IF NOT EXISTS historical_orders (
    order_date TEXT NOT NULL,
    weekday INTEGER NOT NULL,
    hour INTEGER NOT NULL,
    menu_item_id INTEGER NOT NULL REFERENCES menu_items(id),
    orders INTEGER NOT NULL,
    weather TEXT NOT NULL,
    event TEXT NOT NULL,
    festival INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hist_slot ON historical_orders (weekday, hour);
CREATE TABLE IF NOT EXISTS sim_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    state_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    trace_id TEXT NOT NULL,
    request TEXT NOT NULL,
    horizon_hours INTEGER,
    status TEXT NOT NULL,
    summary TEXT,
    escalation_reasons TEXT,
    assumptions_json TEXT,
    llm_json TEXT,
    forecast_json TEXT,
    assessment_json TEXT,
    suggestions_json TEXT,
    trace_json TEXT,
    latency_ms REAL
);
CREATE TABLE IF NOT EXISTS forecast_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER NOT NULL UNIQUE REFERENCES decisions(id),
    forecast_orders REAL NOT NULL,
    actual_orders REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    trace_id TEXT,
    actor TEXT NOT NULL,
    event TEXT NOT NULL,
    details_json TEXT
);
CREATE TABLE IF NOT EXISTS request_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT (datetime('now')),
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status INTEGER NOT NULL,
    latency_ms REAL NOT NULL
);
"""

# Incidents you can switch on from the UI to simulate a situation or an outage on demand.
INCIDENTS = {
    "rain": "Heavy rain (overrides live weather)",
    "festival": "Festival / holiday (overrides live holidays)",
    "cricket_match": "Major cricket match tonight",
    "order_rush": "Order rush (orders running hot right now)",
    "weather_api_down": "Weather API down",
    "holiday_api_down": "Holiday API down",
    "inventory_api_down": "Inventory API down",
}

DEFAULT_SIM_STATE = {
    "simulated_date": None,  # None = use the system date; "YYYY-MM-DD" = simulate that day
    "simulated_hour": None,  # None = use the system time; 10-23 = simulate that hour
    "incidents": {name: False for name in INCIDENTS},
}

# --- Demo master data ------------------------------------------------------------------

SUPPLIERS = [
    (1, "FreshMeat Co", 60, 1),
    (2, "DairyDirect", 180, 1),
    (3, "AgroMart Wholesale", 90, 1),
    (4, "CheeseHub", 240, 1),
]

# id, name, unit, stock, safety_stock, unit_cost, normal_loss_pct, max_order_qty, supplier_id
INGREDIENTS = [
    (1, "Chicken", "kg", 18.0, 5.0, 260.0, 0.05, 40.0, 1),
    (2, "Paneer", "kg", 7.5, 2.0, 380.0, 0.04, 20.0, 2),
    (3, "Basmati Rice", "kg", 30.0, 8.0, 90.0, 0.02, 100.0, 3),
    (4, "Pizza Flour", "kg", 15.0, 4.0, 45.0, 0.02, 60.0, 3),
    (5, "Mozzarella", "kg", 4.8, 1.5, 520.0, 0.03, 15.0, 4),
    (6, "Tomato", "kg", 20.0, 5.0, 40.0, 0.08, 60.0, 3),
    (7, "Onion", "kg", 18.0, 4.0, 35.0, 0.05, 60.0, 3),
    (8, "Fresh Cream", "l", 6.0, 1.5, 220.0, 0.03, 20.0, 2),
    (9, "Mixed Vegetables", "kg", 10.0, 3.0, 60.0, 0.10, 40.0, 3),
    (10, "Cooking Oil", "l", 15.0, 4.0, 150.0, 0.01, 40.0, 3),
]

STATIONS = [
    (1, "Biryani Station", 30),
    (2, "Curry Station", 50),
    (3, "Wok Station", 25),
    (4, "Pizza Oven", 30),
]

# id, name, station_id, price, available
MENU_ITEMS = [
    (1, "Chicken Biryani", 1, 320.0, 1),
    (2, "Paneer Butter Masala", 2, 280.0, 1),
    (3, "Butter Chicken", 2, 340.0, 1),
    (4, "Veg Fried Rice", 3, 220.0, 1),
    (5, "Margherita Pizza", 4, 300.0, 1),
]

# menu_item_id, ingredient_id, qty_per_order (kg or l)
RECIPES = [
    (1, 1, 0.18), (1, 3, 0.15), (1, 7, 0.05), (1, 10, 0.02),
    (2, 2, 0.15), (2, 6, 0.10), (2, 8, 0.03), (2, 7, 0.04), (2, 10, 0.02),
    (3, 1, 0.20), (3, 6, 0.10), (3, 8, 0.04), (3, 10, 0.02),
    (4, 3, 0.15), (4, 9, 0.10), (4, 10, 0.02),
    (5, 4, 0.15), (5, 5, 0.12), (5, 6, 0.06),
]

HOURLY_BASE_ORDERS = {
    10: 8, 11: 15, 12: 35, 13: 40, 14: 28, 15: 14, 16: 12,
    17: 16, 18: 28, 19: 45, 20: 55, 21: 50, 22: 32, 23: 15,
}
WEEKDAY_FACTOR = {0: 1.0, 1: 0.95, 2: 1.0, 3: 1.05, 4: 1.15, 5: 1.3, 6: 1.25}
ITEM_MIX = {1: 0.28, 2: 0.20, 3: 0.18, 4: 0.14, 5: 0.20}
HISTORY_WEEKS = 8


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_connection(db_path: str | None = None):
    """Yield a connection that commits on success and rolls back on error."""
    conn = _connect(db_path or settings.db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _generate_history(seed: int, end_date: date) -> list[tuple]:
    rows = []
    start = end_date - timedelta(weeks=HISTORY_WEEKS)
    day = start
    while day < end_date:
        weekday = day.weekday()
        # Seeded by weekday and "weeks ago", so e.g. the Saturday history is the same whatever day it is made.
        rng = random.Random(f"{seed}-{weekday}-{((end_date - day).days - 1) // 7}")
        weather = "rain" if rng.random() < 0.2 else "clear"
        event = "cricket_match" if rng.random() < 0.12 else "none"
        festival = 1 if rng.random() < 0.05 else 0
        for hour, base in HOURLY_BASE_ORDERS.items():
            factor = WEEKDAY_FACTOR[weekday]
            if weather == "rain" and hour >= 17:
                factor *= 1.25
            if event == "cricket_match" and hour >= 18:
                factor *= 1.4
            if festival:
                factor *= 1.3
            total = base * factor * rng.uniform(0.9, 1.1)
            for item_id, share in ITEM_MIX.items():
                orders = max(0, round(total * share * rng.uniform(0.85, 1.15)))
                rows.append((day.isoformat(), weekday, hour, item_id, orders,
                             weather if hour >= 17 else "clear",
                             event if hour >= 18 else "none", festival))
        day += timedelta(days=1)
    return rows


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first version so existing databases keep working."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(decisions)")}
    for column in ("assumptions_json", "llm_json", "suggestions_json"):
        if column not in existing:
            conn.execute(f"ALTER TABLE decisions ADD COLUMN {column} TEXT")


def init_db(db_path: str | None = None, reset: bool = False) -> None:
    """Create schema and load demo data. With reset=True all data is recreated."""
    path = Path(db_path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(str(path)) as conn:
        if reset:
            # Drop tables in place (rather than deleting the file) so a running server keeps working.
            conn.execute("PRAGMA foreign_keys = OFF")
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
            for table in tables:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        _migrate(conn)
        if conn.execute("SELECT COUNT(*) FROM suppliers").fetchone()[0]:
            return  # already seeded
        conn.executemany("INSERT INTO suppliers VALUES (?, ?, ?, ?)", SUPPLIERS)
        conn.executemany("INSERT INTO ingredients VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", INGREDIENTS)
        conn.executemany("INSERT INTO stations VALUES (?, ?, ?)", STATIONS)
        conn.executemany("INSERT INTO menu_items VALUES (?, ?, ?, ?, ?)", MENU_ITEMS)
        conn.executemany("INSERT INTO recipes VALUES (?, ?, ?)", RECIPES)
        business_date = date.today()  # history ends at the system date
        conn.executemany(
            "INSERT INTO historical_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            _generate_history(42, business_date),
        )
        conn.execute("INSERT INTO sim_state (id, state_json) VALUES (1, ?)",
                     (json.dumps(DEFAULT_SIM_STATE),))
