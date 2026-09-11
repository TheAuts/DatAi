# DatAi — Options Analytics Dashboard

**Product spec (v0.1)**  
**Audience:** specialist agents (data, quant, frontend)  
**Status:** roadmap only — no implementation in this document  
**Owner:** Andrew  
**Last updated:** 2026-09-11

---

## 1. Problem

Retail traders can usually see *live* Greeks on a broker quote screen. They almost never get a clean **historical time series of Greeks** (Delta, Gamma, Theta, Vega, Rho) for a specific contract or strike over days/weeks. That history is what this product visualizes.

DatAi is a **historical (and later near-real-time) Greeks analytics dashboard** for listed equity/ETF options. It is not a broker, not an execution engine, and not a full options chain scanner in v1.

**Primary user:** a single power user (Andrew) who wants to inspect how Greeks evolved for a chosen underlying and contract.

**North-star moment:** open the app, pick a ticker + expiration + strike, see a graph of a Greek over time.

---

## 2. Product principles

1. **History first, live later.** The scarce asset is historical chain + reconstructed Greeks, not another live quote widget.
2. **One contract, then many.** Depth of a single series beats a 500-strike heatmap until the pipeline is trustworthy.
3. **Reproducible numbers.** Every plotted point must be traceable to: market inputs, model, and parameters (rate, calendar, IV method).
4. **Cheap to iterate.** Prefer local files and a thin UI until the graph is correct. Do not stand up a cluster for MVP.
5. **Honest about model limits.** Equity options are American; Black–Scholes is a European approximation. Ship BS for MVP; document the error, do not hide it.

---

## 3. Data strategy

### 3.1 Decision (MVP)

| Concern | Choice | Why |
|---|---|---|
| Market data vendor | **Polygon.io Options** (REST) | Retail-accessible API, historical option aggregates + underlying bars, simpler auth than Databento OPRA. Sufficient for EOD / minute bars on a handful of underlyings. |
| Fallback / later upgrade | Databento (OPRA) | Use if Polygon gaps, rate limits, or tick-level reconstruction become the bottleneck. Treat as Phase 2 ingest, same storage schema. |
| Serving store (MVP) | **Parquet on local disk** (partitioned) | Zero ops, cheap, columnar, Python-native (`pandas` / `polars`). Perfect until query volume or concurrent users exist. |
| Serving store (post-MVP) | TimescaleDB (or Postgres + hypertables) | Add when we need SQL time-range filters, multiple underlyings, and a persistent API. Do **not** introduce a database before the first graph. |
| Cache | On-disk Parquet + in-process cache in the UI process | Avoid re-hitting the vendor for the same `(ticker, date range, contract)`. |

**Explicit non-goals for ingest in MVP:** live websocket Greeks, full OPRA tick replay, unusual-options-activity scanners, multi-vendor blending.

### 3.2 What we actually fetch

Vendors rarely give “historical Delta” as a first-class, cheap series. We **reconstruct** Greeks from historical prices.

For each trading day `t` in the requested window, we need:

| Input | Source (Polygon MVP) | Notes |
|---|---|---|
| Underlying spot `S_t` | Underlying daily (or minute) aggregate close | Use official close for EOD graphs; VWAP optional later. |
| Option last / close `C_t` or `P_t` | Options daily aggregate for that `OCC` symbol | Prefer close; if missing, last trade. Skip the day if unusable. |
| Strike `K`, type (C/P), expiration `T_exp` | Contract metadata | Parse from OCC symbol or contracts endpoint; store as columns, never re-parse in the model. |
| Risk-free rate `r_t` | Static default for MVP (e.g. 4–5%), later a Treasury/SOFR series | Do not block MVP on a rates vendor. Parameterize it. |
| Dividend yield `q` | `0` for MVP; later dividend schedule or implied yield | Call this out on the chart footer. |
| Implied vol `σ_t` | **Inverted from `C_t`/`P_t`** via Black–Scholes | Do not depend on vendor-supplied IV for MVP (nice-to-have overlay later). |

**Universe for MVP:** one underlying, default **SPY**. One expiration. A small set of strikes (ATM ± a few) or a single contract selected by the user.

**History window for MVP:** last **30–60 trading days** (or from listed date if the contract is younger). Do not ingest the entire options universe.

### 3.3 Storage layout (Parquet)

Root: `data/` (gitignored). Partition so later Timescale migration is a mechanical load, not a redesign.

```
data/
  underlyings/
    ticker=SPY/date=YYYY-MM-DD.parquet     # OHLCV
  contracts/
    ticker=SPY/contracts.parquet           # static metadata: occ, strike, cp, expiration
  quotes/
    ticker=SPY/expiration=YYYY-MM-DD/date=YYYY-MM-DD.parquet
      # occ, close, volume, oi (if available)
  greeks/
    ticker=SPY/expiration=YYYY-MM-DD/occ=.../date=YYYY-MM-DD.parquet
      # computed: iv, delta, gamma, theta, vega, rho, model_version, params_hash
  runs/
    run_id=.../manifest.json               # what was fetched, API errors, row counts
```

**Rules:**

- Raw quotes and computed Greeks live in **separate** trees. Recalculating the model must not require re-downloading the chain.
- Every Greeks file stores `model_version` and a `params_hash` (rate, q, calendar convention, IV solver settings).
- IDs are **OCC option symbols** (e.g. `SPY250919C00650000`), not vendor-internal IDs, so Databento can map later.

### 3.4 Ingest pipeline (logical stages)

Specialist agents should implement these as separate jobs, even if each is a single Python module:

1. **Discover** — resolve ticker → listed expirations → contracts (filter: expiration, optional strike band).
2. **Pull underlying** — daily bars for `[start, end]`.
3. **Pull option bars** — daily bars per OCC in universe.
4. **Normalize** — one row per `(occ, date)` with typed columns; drop obviously bad prints (zero/negative prices, `S <= 0`).
5. **Compute** — IV + Greeks (see §4); write `data/greeks/`.
6. **Manifest** — row counts, skipped days, API status. Required for debugging “why is the graph empty?”

Rate-limit and retry belong in the pull layer only. The compute layer is offline and deterministic.

### 3.5 Later: TimescaleDB

When Parquet querying becomes painful (many tickers, interactive filters):

- Table `option_quotes(time, occ, ticker, expiration, strike, cp, close, volume, oi)`
- Table `option_greeks(time, occ, model_version, iv, delta, gamma, theta, vega, rho, params_hash)`
- Hypertable on `time`; indexes on `(occ, time)` and `(ticker, expiration, time)`
- Load from existing Parquet; Parquet remains cold backup

---

## 4. Calculation engine

### 4.1 Model (MVP)

**Black–Scholes–Merton** (European, continuous yield `q`).

Let  
`S` = underlying, `K` = strike, `T` = years to expiration, `r` = continuous risk-free rate, `q` = continuous dividend yield, `σ` = implied volatility.

```
d1 = [ln(S/K) + (r - q + 0.5 σ²) T] / (σ √T)
d2 = d1 - σ √T
```

**Call**

- Price: `S e^{-qT} N(d1) - K e^{-rT} N(d2)`
- Delta: `e^{-qT} N(d1)`
- Gamma: `e^{-qT} n(d1) / (S σ √T)`
- Vega: `S e^{-qT} n(d1) √T`  (per 1.0 vol; **display as per 1 vol point** = value / 100)
- Theta: use the standard BSM theta (per year), **display as per calendar day** = value / 365 (document this)
- Rho: `K T e^{-rT} N(d2)` (call); display per 1% rate = value / 100

**Put** — put-call parity / standard BSM put formulas (Delta `e^{-qT}(N(d1)-1)`, etc.). Do not derive puts from calls in the time series unless both quotes exist and we are running a quality check.

`N` = standard normal CDF, `n` = PDF.

### 4.2 Time to expiry

- `T = max(calendar_days_remaining, 0) / 365`
- On expiration day, `T` may be ~0; **do not compute** Greeks when `T < T_min` (recommend `1e-4` years ≈ 1 hour). Mark the point as `expired` / skip.
- MVP uses **calendar** year (365). Trading-day year (252) is a later toggle. Store which convention was used.

### 4.3 Implied volatility

Vendor close is the market price. Invert BS for `σ`:

1. Bracket / Newton–Raphson (or Brent) on `BS(σ) - market_price = 0`.
2. Bounds: `σ ∈ (1e-4, 5.0)` (0.01% to 500%).
3. If no root (price outside arb bounds, e.g. call below intrinsic for European — possible with American + discrete dividends + wide bid/ask): **leave IV/Greeks null** for that day; do not force a number.
4. Seed Newton with previous day’s IV when available (warms the solver along the series).

**Intrinsic (European) sanity checks** before solving:

- Call: `max(0, S e^{-qT} - K e^{-rT})` ≤ price ≤ `S e^{-qT}`
- Put: `max(0, K e^{-rT} - S e^{-qT})` ≤ price ≤ `K e^{-rT}`

Slightly loose epsilon (~1–2 ticks) is allowed so noisy closes still solve.

### 4.4 Parameter defaults (must be visible in UI footer)

| Parameter | MVP default | Later |
|---|---|---|
| `r` | `0.045` | daily SOFR or 3M T-bill series |
| `q` | `0.0` | dividend yield or discrete dividends |
| Day count | 365 | 365 vs 252 toggle |
| Exercise style | European BS | American (binomial / Bjerksund–Stensland) |
| Price field | daily close | mid = (bid+ask)/2 when available |

### 4.5 Output contract (what the graph consumes)

One row per `(occ, date)`:

```
date, occ, ticker, cp, strike, expiration,
S, option_price, T, r, q,
iv, delta, gamma, theta_per_day, vega_per_vol_point, rho_per_percent,
solve_status, model_version
```

`solve_status`: `ok` | `no_price` | `bound_violation` | `solver_fail` | `expired`

The frontend plots only `solve_status == ok`. A small “dropped points” count belongs on the chart.

### 4.6 Validation (blocking for “correct graph”)

Quant specialist must include a tiny fixture, not a research paper:

- Known textbook BS price/Greeks for a canned `(S,K,T,r,q,σ)` — assert within tight tolerance.
- Put-call parity on synthetic prices.
- Spot-check one live SPY contract: Delta in `[-1, 1]`, Gamma ≥ 0, Vega ≥ 0, IV in a sane range for that name.

### 4.7 Known limitations (product copy, not just comments)

- American early-exercise premium is ignored; short-dated ITM puts are the worst case.
- Using **last/close** (not mid) biases IV, especially in wide markets or low volume.
- Overnight jumps in `S` without a matching option print will create holes, not interpolated Greeks.
- Theta sign and day-count must be labeled; “theta = -0.12” without “per day” is a product bug.

---

## 5. Frontend architecture

### 5.1 Decision (MVP)

**Streamlit + Plotly** (Python, single process).

| Need | How Streamlit covers it |
|---|---|
| First Greek graph fast | Native Plotly charts, no JS build |
| Historical rendering | `st.plotly_chart` on a Polars/Pandas frame from Parquet |
| “Real-time” later | `st.fragment` / timed rerun, or a “Refresh” button that re-runs compute on latest bar — **not** a websocket mesh in MVP |
| Performance | Cache Parquet reads (`@st.cache_data`); never recompute BS on every widget tick if Greeks are already on disk |

**Post-MVP (if the Streamlit app feels like a notebook):** React (Vite) + Plotly.js (or uPlot if we hit tens of thousands of points) talking to a thin FastAPI that reads Parquet/Timescale. Same compute engine; only the shell changes.

**Explicit non-goals for UI in MVP:** user accounts, mobile app, dark-pool flow heatmaps, 3D vol surfaces (vol surface is Phase 2+).

### 5.2 Information architecture (one screen)

**Page: Contract Greeks**

1. **Controls (top)**  
   Ticker (default SPY) · Expiration date · Strike · Call/Put · Greek metric (Delta default) · Date range.
2. **Primary chart**  
   X = date, Y = selected Greek. Optional second trace: underlying `S` on a **secondary y-axis** (off by default).
3. **Context strip**  
   Contract OCC, DTE, last IV, last Delta, `r`, `q`, model version.
4. **Data quality**  
   Count of `ok` vs dropped days; link or expander with `solve_status` table.

No dashboard chrome beyond this until the graph is trusted.

### 5.3 Performance rules

- Chart data is **precomputed Parquet**, not BS in the request path, except a “Recalculate” action.
- MVP series length is ~30–60 points: Plotly is fine. If we later plot minute bars (thousands of points), downsample for display; keep full resolution on disk.
- Do not load all strikes’ Greeks into memory on page load; read one `occ` partition.

### 5.4 Near-real-time (Phase 1.5, not MVP)

After EOD graphs work:

- Poll Polygon snapshot (or last aggregate) on an interval (e.g. 15–60s) for the **selected contract only**.
- Append one row: invert IV → Greeks → update Plotly.
- If polling exceeds plan limits, degrade to manual refresh.

True streaming (websocket) waits for a dedicated API process; Streamlit is a viewer, not a market-data server.

---

## 6. MVP scope — smallest version that shows a Greek graph

This is the **only** thing v0 must do.

### 6.1 In

- Config: Polygon API key via environment variable (never committed).
- CLI or script: `ingest` SPY, **one** expiration (nearest monthly with ≥ 7 DTE), **one** strike nearest ATM as of ingest day, calls **or** puts (pick calls).
- Pull ~30–60 trading days of underlying + that single option’s daily closes (or from contract listing date if shorter).
- Write Parquet under `data/`.
- Run BS IV inversion + Delta (and store other Greeks while we are there — cheap — but the UI may only chart Delta).
- Streamlit page: dropdowns can be **hardcoded** to that one contract if widgets slow us down.
- Plotly line chart: **Delta vs date**.
- Footer: model, `r`, `q`, OCC symbol.

### 6.2 Out (do not build)

- Databento, TimescaleDB, Docker compose with a database
- Multi-ticker search, watchlists, auth
- Full chain heatmaps, IV surface, P&L scenarios, strategy builder
- American model, dividend curves, live websocket
- Bid/ask mid, greeks-at-the-touch
- Tests beyond the small BS fixture and a smoke ingest
- Pixel-perfect design system

### 6.3 Definition of done (MVP)

A local user can:

1. Set `POLYGON_API_KEY`, run ingest, run Streamlit.
2. See a non-empty Delta time series for one SPY call.
3. Hover a point and read date + Delta.
4. Explain any missing days via `solve_status` (not silent gaps).

If that works, the product exists. Everything else is iteration.

### 6.4 Suggested build order for specialist agents

| Order | Agent | Deliverable |
|---|---|---|
| 1 | Quant | Pure functions: BS price, Greeks, IV solver + fixture tests. No I/O. |
| 2 | Data | Polygon client → normalize → Parquet layout + manifest. Uses quant functions. |
| 3 | Frontend | Streamlit page that reads one Greeks Parquet and draws Plotly. |

Do not start UI until a fixture Greeks series can be written without the network (synthetic `S`/`C` path) so the chart can be developed offline.

---

## 7. Success metrics (early)

- Time-to-first-graph from a clean clone (after API key): target **< 30 minutes** of ingest + run.
- Share of trading days with `solve_status == ok` for the MVP contract: target **≥ 80%** over 30 days (thin names may miss this; SPY should not).
- Recompute without re-fetch: changing `r` regenerates `data/greeks/` only.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Polygon options entitlement / plan does not include historical option aggregates | Spike ingest on day 1 with a single OCC; if blocked, swap vendor adapter to Databento without changing Parquet schema. |
| American vs European error looks like a “bug” | Label the model on the chart; add American engine only after Delta series looks plausible. |
| Bad prints → wild IV | Bound checks + nulls; never interpolate IV across a fail. |
| Scope creep (surfaces, scanners) | §6.2 is the kill list until DoD is met. |

---

## 9. Phase map (after MVP)

- **P1:** User-selectable expiration + strike list from ingested metadata; chart any Greek; overlay `S`.
- **P1.5:** Polling refresh for selected contract.
- **P2:** Multiple underlyings; Timescale; Databento adapter; mid-price IV.
- **P3:** Term-structure / smile snapshots; American pricing; strategy-level aggregate Greeks.

---

## 10. Open questions (do not block MVP)

1. Polygon plan: daily option aggregates vs minute — confirm entitlement before optimizing cadence.
2. Default theta day-count: 365 vs 252 — pick 365 and document.
3. Whether to show **signed** Delta for shorts later (portfolio view) — not in MVP.

When these are decided, update this spec; do not fork undocumented conventions in code.
