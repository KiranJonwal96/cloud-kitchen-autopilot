# Code guide: how the code works, in plain language

This guide walks through the code in the order a request travels through it. Each stop explains
**what the file does**, shows a **short piece of the real code** (some trimmed for length), and gives
you **one sentence to say in class**.

---

## The big picture

Think of the project as a small company:

| Folder | "Department" | What it does |
|---|---|---|
| `static/` | Reception | The web page the supervisor uses |
| `app/api/` | Front desk | FastAPI: receives requests from the web page |
| `app/graph/` | Manager's office | LangGraph: decides which agent works next |
| `app/agents/` | The specialists | Demand Agent and Operations Agent |
| `app/llm/` | The brain | Talks to OpenAI (LangChain) and holds the prompts |
| `app/rag/` | The library | Finds company procedures in ChromaDB |
| `app/tools/` | The workshop | Data, calculations, rules, memory |
| `knowledge/` | The rule books | 6 kitchen procedure documents |
| `tests/` | Quality control | 51 automated tests |

**One-line summary:** the web page sends a question → FastAPI passes it to LangGraph → the Orchestrator
(LLM) calls the agents → the agents use tools and RAG → rules check the suggestions → the supervisor
reads the answer and decides what to do.

---

## Stop 1: starting the app (`run.py`)

```python
check_dependencies()      # are the packages installed?
init_db(reset=args.reset) # create the database with demo data
index_knowledge()         # put the procedure documents into ChromaDB
uvicorn.run(app, ...)     # start the FastAPI web server
```

**Say:** "run.py prepares the database and the knowledge base, then starts the web server."

---

## Stop 2: the front desk (`app/api/main.py`, FastAPI)

Each URL is a small Python function. FastAPI checks the input with a Pydantic model.

```python
class PlanRequest(BaseModel):
    request: str = Field(max_length=2000)
    horizon_hours: Optional[int] = Field(default=None, ge=1, le=24)

@app.post("/api/plan")
def create_plan(body: PlanRequest):
    return workflow.run_plan(body.request, horizon_hours=body.horizon_hours)
```

If someone sends a bad value, such as `horizon_hours: 99`, FastAPI rejects it automatically (HTTP 422).
FastAPI also creates interactive documentation at `/docs`.

**Say:** "FastAPI receives the question, validates it, and hands it to the multi-agent workflow."

---

## Stop 3: the workflow map (`app/graph/workflow.py`, LangGraph)

This is the most important file to show. It draws the "map" of who runs after whom:

```python
graph = StateGraph(KitchenState)
graph.add_edge(START, "guardrail")                       # 1. check the question
graph.add_conditional_edges("supervisor", route_next, {  # 2. the Orchestrator chooses:
    "demand_agent": "demand_agent",                      #    call the Demand Agent
    "operations_agent": "operations_agent",              #    call the Operations Agent
    "finalize": "finalize",                              #    or finish
})
graph.add_edge("demand_agent", "supervisor")             # agents always report back
graph.add_edge("operations_agent", "supervisor")
graph.add_conditional_edges("finalize", route_next, {DONE: END})          # done
```

- A **node** is one step (a Python function).
- An **edge** says what comes next. A **conditional edge** means "decide at run time".
- The **state** (`app/graph/state.py`) is a shared dictionary every node reads and updates: the question,
  the forecast, the assessment and so on.

**Say:** "LangGraph is the map. The Orchestrator picks the route at run time, and agents always come back to it."

---

## Stop 4: the Orchestrator decides (`app/graph/nodes.py` → `supervisor_node`)

The LLM gets the question plus everything done so far, and must reply in a fixed format
(`RouteDecision` in `app/llm/schemas.py`):

```python
class RouteDecision(BaseModel):
    thought: str          # its reasoning, shown in the UI
    action: Literal["call_demand_agent", "call_operations_agent", "clarify", "escalate", "finish"]
    horizon_hours: Optional[int] = None
    summary: Optional[str] = None
```

Code then checks the decision before following it:

```python
if decision.action == "call_operations_agent" and not state.get("forecast"):
    step["observation"] = {"error": "A demand forecast is required first."}  # sent back to the LLM
```

**Say:** "The LLM chooses the next step, but code enforces the rules. It can't skip the forecast or loop forever."

---

## Stop 5: the Demand Agent (`app/agents/demand_agent.py`)

Every agent follows **PRAL**, defined in `app/agents/base.py`:

```python
def _pral(self, task):
    context = self.perceive(task)       # P: collect data with tools
    conclusions = self.reason(context)  # R: statistics, then LLM review
    return self.act(conclusions)        # A: return the result
# L: learn() reads past outcomes and adjusts future forecasts
```

The LLM may adjust the statistical forecast, but **only within limits**:

```python
factor = min(1.25, max(0.8, requested))                    # LLM can change it by -20% .. +25%
reviewed["spike"] = reviewed["uplift"] >= settings.spike_ratio  # spike is recalculated by code
```

**Say:** "Statistics calculate the forecast; the LLM reviews and explains it, but can't produce wild numbers."

---

## Stop 6: the Operations Agent and RAG (`operations_agent.py`, `app/rag/knowledge_base.py`)

1. Tools calculate shortages and station load (`app/tools/calculations.py`):

   ```python
   gross_requirement = consumption + normal_loss + safety_stock
   net_requirement = max(0, gross_requirement - current_stock)
   ```

2. **RAG** finds the matching company procedure in ChromaDB:

   ```python
   result = collection.query(query_texts=[query], n_results=4)  # search by meaning
   score = 1 - distance                                         # 1 = perfect match
   if score >= 0.25: hits.append(...)                           # ignore weak matches
   ```

3. The LLM chooses actions from the calculated list, explains them and cites the procedures
   (for example "per SOP-04#2"). Critical actions can't be dropped.

**Say:** "RAG means: retrieve the right procedure, add it to the prompt, and the LLM answers using it."

---

## Stop 7: prompts and the LLM connection (`app/llm/prompts.py`, `app/llm/client.py`)

**Dynamic prompts:** a fixed part plus live context added at run time:

```python
def operations_system(has_knowledge):
    # different rules depending on whether RAG found procedures
    return OPERATIONS_SYSTEM.replace("KNOWLEDGE_RULES",
                                     KNOWLEDGE_RULES_FOUND if has_knowledge else KNOWLEDGE_RULES_NONE)
```

**LangChain** sends the prompt to OpenAI:

```python
chat = ChatOpenAI(model="gpt-4o-mini", api_key=..., temperature=0.2)
structured = chat.with_structured_output(RouteDecision)  # reply must match the schema
```

**Say:** "Prompts are templates filled with live data; LangChain sends them to OpenAI and checks the format of the reply."

---

## Stop 8: orders we can't fulfil, and safety (`operations_agent.py`, `policy_engine.py`)

For each dish, the scarcest ingredient decides how many expected orders can be served:

```python
coverage = stock / need                       # e.g. chicken: 18 kg / 22.1 kg = 0.81
can_serve = expected_orders * coverage        # 67 biryani expected -> 54 can be served
cannot_serve = expected_orders - can_serve    # 13 can't
```

It also works out **when** the ingredient runs out and whether a delivery can arrive in time. In time →
suggest "reorder now". Too late → suggest "stop taking this dish after N more orders".

The **policy engine** then checks every suggestion with plain code, with no AI involved:

```python
if quantity > ingredient["max_order_qty"]:  violations.append("Quantity exceeds max order quantity")
if cost > settings.max_spend_per_po:        violations.append("Order value exceeds the per-order limit")
```

**Say:** "Code calculates which orders we can't fulfil, the AI explains what to do, rules check the
suggestions, and the supervisor decides."

---

## Stop 9: memory, MCP and tests

- **Live data and incidents** (`app/tools/data_tools.py`): `get_weather()` calls Open-Meteo and
  `get_holidays()` calls Calendarific by default, asking only for main festivals (`type=national`) for the system date, or the date picked in **Simulate a date**. `get_sim_state()` takes the date and hour from the system clock (`datetime.now()`) unless **Simulate a date** / **Simulate a time** is set. A simulated incident (rain, festival, an API being down)
  takes priority, so you can stage a situation on demand:
  ```python
  _check_outage(state, "weather_api_down", "weather_api")   # simulated outage first
  if state["incidents"]["rain"]:                             # then a simulated incident
      return {"condition": "rain", "source": "simulated incident"}
  # ...otherwise call the real Open-Meteo API
  ```
- **Memory** (`app/tools/memory_store.py`, `app/db.py`): SQLite tables for every answer (with its
  suggestions), the actual order counts used for learning, and the audit log.
- **MCP** (`app/mcp_server.py`): four functions marked with `@mcp.tool()` so other AI apps can call them:
  ```python
  @mcp.tool()
  def forecast_demand(horizon_hours: int = 2) -> dict: ...
  ```
- **Tests** (`tests/`): 54 tests, including the happy path, a demand spike, the can't-fulfil numbers, unsafe
  prompts, API outages, RAG and MCP. Run them with `python -m unittest discover -s tests -t . -v`.

---

## Follow one request through the code

| # | What happens | File → function |
|---|---|---|
| 1 | Supervisor clicks **Run plan** | `static/app.js` → `api("/api/plan")` |
| 2 | FastAPI validates and starts the workflow | `app/api/main.py` → `create_plan` |
| 3 | Unsafe or off-topic text is blocked | `app/agents/guardrails.py` → `check_request` |
| 4 | Orchestrator LLM picks "call Demand Agent" | `app/graph/nodes.py` → `supervisor_node` |
| 5 | Forecast is calculated and reviewed | `app/agents/demand_agent.py` |
| 6 | Orchestrator picks "call Operations Agent" | `supervisor_node` |
| 7 | Stock and capacity are checked, procedures retrieved | `operations_agent.py`, `knowledge_base.py` |
| 8 | Orchestrator picks "finish" and writes the answer | `supervisor_node` |
| 9 | Rules check every action, the decision is saved | `app/graph/finalize.py` |
| 10 | The page shows the answer, orders we can't fulfil and suggestions | `static/app.js` → `render` |

---

## Words you may be asked about

| Word | Simple meaning |
|---|---|
| **Agent** | A program that perceives, reasons and acts toward a goal, using tools |
| **Orchestrator / supervisor** | The agent that decides which other agent works next |
| **Node / edge (LangGraph)** | A step in the workflow / the arrow to the next step |
| **State** | The shared notebook every step reads from and writes to |
| **Structured output** | Forcing the LLM to reply in a fixed format (a Pydantic model) |
| **RAG** | Retrieve relevant documents, Augment the prompt with them, Generate the answer |
| **Embedding** | Turning text into numbers so similar meanings are close together |
| **Chunk** | One section of a document, stored and searched separately |
| **Dynamic prompt** | A prompt template filled with live information each time |
| **MCP** | A standard way for AI apps to use another program's tools |
| **Guardrail** | A check that blocks unsafe or off-topic input |
| **Idempotent** | Doing it twice has the same effect as doing it once |

---

## Questions your class may ask

**Why not let the LLM calculate the forecast?**
LLMs are bad at precise arithmetic and can invent numbers. Code calculates; the LLM judges and explains.

**What if OpenAI is down or the key is wrong?**
The workflow falls back to a fixed route (Demand → Operations) and still gives an answer, marked "LLM fallback".

**Can someone trick it with a prompt like "ignore your rules"?**
No. The guardrail blocks it before the LLM sees it, and the system can only suggest, never change anything.

**Why ChromaDB and also SQLite?**
SQLite holds exact numbers (stock, orders). ChromaDB holds text documents and searches them by meaning.

**What makes it agentic and not a chatbot?**
It plans its own steps, uses tools, coordinates several agents, gives concrete suggestions and learns from outcomes.
