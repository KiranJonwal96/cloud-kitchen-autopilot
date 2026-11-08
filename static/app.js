"use strict";

// Incidents that can be switched on to simulate a situation or an outage on demand.
const INCIDENTS = ["rain", "festival", "cricket_match", "order_rush",
  "weather_api_down", "holiday_api_down", "inventory_api_down"];
const BUSY_EVENING = { rain: true, cricket_match: true, order_rush: true };

const STATUS_LABELS = {
  ok: "All good",
  action_needed: "Action needed",
  escalated: "Needs your attention",
  refused: "Refused",
  needs_clarification: "Please clarify",
};

const SUGGESTION_LABELS = {
  PURCHASE_ORDER: "Reorder",
  MENU_THROTTLE: "Limit a dish",
  ADVISORY: "Kitchen",
};

const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
}

function showError(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(showError.timer);
  showError.timer = setTimeout(() => toast.classList.add("hidden"), 5000);
}

async function api(path, body) {
  const response = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data;
}

// ---------- Rendering ----------

function describeInputs(inputs) {
  if (!inputs) return "";
  const w = inputs.weather;
  const e = inputs.events;
  const weather = w
    ? `Weather: ${escapeHtml(w.condition)}${w.temperature_c != null ? `, ${Math.round(w.temperature_c)}°C` : ""} (${escapeHtml(w.source || "demo")})`
    : "Weather: unavailable";
  const holiday = e
    ? `Holiday/festival: ${e.holidays && e.holidays.length ? escapeHtml(e.holidays.join(", ")) : (e.festival ? "yes" : "none")} · Cricket match: ${e.event === "cricket_match" ? "yes" : "no"} (${escapeHtml(e.source || "demo")})`
    : "Holidays and events: unavailable";
  return `<p class="muted data-used">Data used: ${weather} · ${holiday}</p>`;
}

function renderAnswer(decision) {
  const forecast = decision.forecast_json;
  const notices = [
    ...(decision.assumptions_json || []).map((a) => `<div class="notice info">${escapeHtml(a)}</div>`),
    ...(decision.escalation_reasons || []).map((r) => `<div class="notice bad">${escapeHtml(r)}</div>`),
  ].join("");
  const kpis = forecast ? `
    <div class="kpis">
      <div class="kpi"><div class="label">Expected orders (next ${forecast.horizon_hours}h)</div>
        <div class="value">${Math.round(forecast.expected_orders)}</div></div>
      <div class="kpi"><div class="label">Compared with a normal day</div>
        <div class="value">${forecast.uplift_pct > 0 ? "+" : ""}${Math.round(forecast.uplift_pct)}%</div></div>
      <div class="kpi"><div class="label">Forecast confidence</div>
        <div class="value">${forecast.confidence.toFixed(2)}</div></div>
    </div>` : "";
  return `
    <section class="card">
      <span class="badge s-${escapeHtml(decision.status)}">${STATUS_LABELS[decision.status] || escapeHtml(decision.status)}</span>
      <p class="summary">${escapeHtml(decision.summary)}</p>
      ${notices}${kpis}
      ${forecast ? describeInputs(forecast.inputs) : ""}
    </section>`;
}

function renderCannotFulfil(assessment) {
  if (!assessment) return "";
  const rows = assessment.dishes.filter((d) => d.cannot_serve > 0);
  if (!rows.length) {
    return `<section class="card"><h2>Orders we can't fulfil</h2>
      <p class="muted">None. Current stock covers every expected order.</p></section>`;
  }
  return `
    <section class="card">
      <h2>Orders we can't fulfil</h2>
      <table>
        <thead><tr><th>Dish</th><th class="num">Expected</th><th class="num">Can serve</th>
          <th class="num">Can't serve</th><th>Short ingredient</th><th>Runs out</th><th>Next delivery</th></tr></thead>
        <tbody>${rows.map((d) => `
          <tr>
            <td>${escapeHtml(d.dish)}</td>
            <td class="num">${d.expected_orders}</td>
            <td class="num">${d.can_serve}</td>
            <td class="num bad">${d.cannot_serve}</td>
            <td>${escapeHtml(d.short_ingredient)} <span class="muted">(short ${d.short_by} ${escapeHtml(d.unit)})</span></td>
            <td>${escapeHtml(d.runs_out_at || "-")}</td>
            <td>${escapeHtml(d.restock_at || "-")}
              <span class="badge ${d.restock_in_time ? "s-ok" : "s-escalated"}">${d.restock_in_time ? "in time" : "too late"}</span></td>
          </tr>`).join("")}</tbody>
      </table>
    </section>`;
}

function renderSuggestions(decision) {
  const suggestions = decision.suggestions_json || [];
  if (!suggestions.length) return "";
  const assessment = decision.assessment_json || {};
  const cited = new Set(assessment.citations || []);
  const procedures = (assessment.knowledge || []).filter((k) => !cited.size || cited.has(k.id));
  return `
    <section class="card">
      <h2>Suggestions</h2>
      ${suggestions.map((s) => `
        <div class="suggestion">
          <div class="kind">${SUGGESTION_LABELS[s.action_type] || escapeHtml(s.action_type)}</div>
          <div class="text">${escapeHtml(s.description)}</div>
          ${s.payload.rationale ? `<div class="why">Why: ${escapeHtml(s.payload.rationale)}</div>` : ""}
          ${s.policy.violations.map((v) => `<div class="warning">Breaks a rule: ${escapeHtml(v)}</div>`).join("")}
        </div>`).join("")}
      ${procedures.length ? `<div class="why">Based on company procedures:
        ${procedures.map((k) => `<span class="chip" title="${escapeHtml(k.title)}">${escapeHtml(k.id)} ${escapeHtml(k.section)}</span>`).join("")}</div>` : ""}
      <p class="muted">These are suggestions only. You decide what to do.</p>
    </section>`;
}

function renderDetails(decision) {
  const info = decision.llm_json || {};
  const forecast = decision.forecast_json;
  const assessment = decision.assessment_json;
  const mode = info.mode === "llm"
    ? `AI agents (${escapeHtml(info.model)}), ${info.calls} AI calls, ${(info.input_tokens || 0) + (info.output_tokens || 0)} tokens`
    : info.mode === "llm_fallback" ? `AI failed, fixed route used: ${escapeHtml(info.fallback_reason)}`
    : "Rule-based (no AI key set)";
  const steps = (info.planner_steps || []).map((s) => `
    <div class="step"><div class="who">Orchestrator · step ${escapeHtml(s.step)}</div>
      ${escapeHtml(s.thought)}<div class="call">→ ${escapeHtml(s.action)}</div></div>`).join("");
  const demand = forecast && forecast.reasoning
    ? `<div class="step"><div class="who">Demand agent</div>${escapeHtml(forecast.reasoning)}</div>` : "";
  const operations = assessment && assessment.risk_summary
    ? `<div class="step"><div class="who">Operations agent</div>${escapeHtml(assessment.risk_summary)}
        ${assessment.prep_plan ? `<br>Prep plan: ${escapeHtml(assessment.prep_plan)}` : ""}</div>` : "";
  const trace = (decision.trace_json || []).map((t) =>
    `<span class="chip">${escapeHtml(t.kind)}: ${escapeHtml(t.component)} ${t.duration_ms} ms</span>`).join("");
  return `
    <section class="card">
      <details>
        <summary>How the agents worked this out</summary>
        <p class="muted">${mode}</p>
        ${steps}${demand}${operations}
        <p class="muted">Steps run:</p>${trace}
      </details>
    </section>`;
}

function render(decision) {
  $("#result").innerHTML = renderAnswer(decision) + renderCannotFulfil(decision.assessment_json)
    + renderSuggestions(decision) + (decision.forecast_json ? renderDetails(decision) : "");
}

// ---------- Data sources and incidents ----------

function markIncidents(incidents) {
  document.querySelectorAll("[data-incident]").forEach((b) => b.classList.toggle("active", Boolean(incidents[b.dataset.incident])));
}

async function setIncidents(changes, extra = {}) {
  try {
    await api("/api/simulation", { incidents: changes, ...extra });
    $("#result").innerHTML = "";
    loadStatus();
  } catch (error) {
    showError(error.message);
  }
}

async function loadStatus() {
  try {
    const [health, state] = await Promise.all([api("/health"), api("/api/simulation")]);
    const badge = $("#llm-status");
    badge.textContent = health.llm.enabled ? `AI: ${health.llm.model}` : "AI off (rule-based)";
    badge.className = `badge ${health.llm.enabled ? "s-ok" : "s-action_needed"}`;
    badge.title = health.llm.enabled ? "LangGraph agents using OpenAI" : health.llm.reason;
    const sources = health.data_sources;
    const day = new Date(`${state.business_date}T00:00:00`).toLocaleDateString("en-IN",
      { weekday: "short", day: "numeric", month: "short", year: "numeric" });
    const clock = state.simulated_date || state.simulated_hour !== null ? "simulated" : "live";
    $("#sources").textContent = `Date: ${day}, ${state.current_hour}:00 (${clock}) · Weather: ${sources.weather} · `
      + `Holidays: ${sources.holidays} · Orders and stock: demo data`;
    $("#sim-date").value = state.simulated_date || "";
    $("#sim-hour").value = state.simulated_hour === null ? "" : String(state.simulated_hour);
    markIncidents(state.incidents);
  } catch (error) {
    showError(error.message);
  }
}

// ---------- Events ----------

document.querySelectorAll("[data-incident]").forEach((button) => {
  button.addEventListener("click", () => setIncidents({ [button.dataset.incident]: !button.classList.contains("active") }));
});
$("#busy-evening").addEventListener("click", () => setIncidents(BUSY_EVENING, { simulated_hour: 19 }));
$("#back-to-live").addEventListener("click",
  () => setIncidents(Object.fromEntries(INCIDENTS.map((n) => [n, false])), { simulated_date: null, simulated_hour: null }));
$("#sim-date").addEventListener("change", (event) => setIncidents({}, { simulated_date: event.target.value || null }));
$("#sim-hour").addEventListener("change",
  (event) => setIncidents({}, { simulated_hour: event.target.value === "" ? null : Number(event.target.value) }));

$("#ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("#ask-button");
  button.disabled = true;
  button.textContent = "Thinking…";
  try {
    render(await api("/api/plan", { request: $("#question").value }));
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Get suggestions";
  }
});

loadStatus();
