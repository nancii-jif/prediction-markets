# v0 Implementation Plan — internal call-auction market (cold start, minimal)

This file **supersedes all previous IMPLEMENTATION.md versions and any
conflicting part of spec.txt**. Agents quote two-sided on an internal
exchange cleared by per-round call auctions. **No Kalshi price information
is ever visible to any agent.** Kalshi is used for exactly three things:
selecting/refilling the live markets, detecting resolutions for settlement,
and (data collection only — nothing is computed in v0) the evaluation
benchmark. The quote is the entire agent output; there is no separate
belief elicitation, no baselines, no scoring pipeline. Analysis happens
later as ad-hoc queries against the DB.

**`stream.py` is the ONLY module that talks to Kalshi.** Everything else —
including live-market membership — is downstream of the DB that stream
writes. Kalshi is queried **on demand**: the routine load is a targeted
fetch of live ∪ held tickers; full candidate sweeps happen only at seed and
when a resolution demands a refill.

Two independent cadences, both in `config.py`:

- **Cadence A** (`stream_interval`, default 1h) — `stream.refresh_live()`,
  cron.
- **Cadence B** (`round_interval`, default 1h) — auction rounds, runner.
  The runner triggers `stream.refresh_live()` at round start, so B never
  depends on A's phase.

Python 3.10+, stdlib `sqlite3`, LiteLLM. No agent framework, no daemons:
cron + one-shot scripts + the runner. Execute stages in order; run each
**done when** before moving on.

## Core invariant (enforced)

No agent-visible surface may contain Kalshi price data: no `yes_bid`,
`yes_ask`, `last`, `volume_24h`, no derived spread/mid. Agents see market
metadata (title, rules, close time), the internal exchange's own history,
and their own ledger state. A test (Stage 7) greps modules for
`kalshi_client` imports (only `stream.py` may have one) AND asserts a
rendered brief contains none of the snapshot price fields.

```
prediction_markets/
  config.py         # every knob; config JSON logged to DB at run start
  kalshi_client.py  # unchanged from previous codebase
  database.py
  stream.py         # SOLE Kalshi caller: refresh_live() + sweep_candidates()
  live_markets.py   # membership logic; pure DB downstream of stream
  ledger.py         # cash, signed YES positions, max-loss margin
  auction.py        # call-auction clearing, pure functions
  agents.py         # brief -> one LLM call -> parsed quote
  tools.py          # v1 STUBS only: web_search, get_news -> NotImplementedError
  runner.py         # round loop
data/market.db      # gitignored
```

---

## Stage 1 — kalshi_client (unchanged)

Public REST (`https://api.elections.kalshi.com/trade-api/v2`), no
credentials. `get_markets` follows cursor pagination; `get_markets_by_tickers`
batches by **encoded URL length, not count** (HTTP 414 otherwise); backoff on
5xx, long sleep on 429; always `mve_filter=exclude` on universe queries.
Imported by `stream.py` and nothing else.

Status vocabulary (verified live; request values ≠ response values —
`status == "settled"` never matches):

| request `status=` | response `status` | `result` |
|---|---|---|
| `open` | `active` | `''` |
| `closed` | `determined` | `yes`/`no` |
| `settled` | `finalized` | `yes`/`no` |
| `unopened` | `initialized` | `''` |

Resolved ⟺ **`result` non-empty**. `result` may be `"scalar"` with the payout
in `settlement_value_dollars` (~2.6% of resolutions); not pre-filterable by
`market_type`.

**Done when:** existing checks pass (no-op if intact).

## Stage 2 — database

WAL mode (cron stream and runner may overlap). All internal money in integer
**cents**, sizes in integer contracts, prices on the 1–99 grid, contracts
pay 100¢.

```sql
-- Kalshi-side (stream writes; live_markets reads DB only; agents NEVER read prices)
markets      (ticker TEXT PRIMARY KEY, title TEXT, rules TEXT, sector TEXT,
              open_ts INT, close_ts INT, first_seen INT)
live_markets (ticker TEXT PRIMARY KEY REFERENCES markets, sector TEXT,
              added_at INT, retired_at INT)
snapshots    (ticker TEXT, captured_at INT, yes_bid REAL, yes_ask REAL,
              bid_size REAL, ask_size REAL, last REAL, volume_24h REAL,
              status TEXT, result TEXT, settlement_value REAL,
              PRIMARY KEY (ticker, captured_at))

-- Internal exchange (always YES space, regardless of quote frame)
cash      (agent TEXT PRIMARY KEY, balance_cents INT)
positions (agent TEXT, ticker TEXT, signed_qty INT, cost_basis_cents INT,
           PRIMARY KEY (agent, ticker))
quotes    (round_id INT, agent TEXT, ticker TEXT,
           bid_cents INT, bid_size INT, ask_cents INT, ask_size INT,
           status TEXT, reject_reason TEXT,
           PRIMARY KEY (round_id, agent, ticker))
auctions  (ticker TEXT, round_id INT, ts INT, clear_cents INT, -- NULL = no print
           cleared_qty INT, tie_lo INT, tie_hi INT,
           PRIMARY KEY (ticker, round_id))
fills     (id INTEGER PRIMARY KEY, round_id INT, agent TEXT, ticker TEXT,
           signed_qty INT, price_cents INT)
settlements (agent TEXT, ticker TEXT, round_id INT, qty INT,
             payout_cents INT, scalar INT DEFAULT 0)

-- Provenance
calls     (agent TEXT, ticker TEXT, round_id INT, ts INT, brief TEXT,
           raw_response TEXT, parse_ok INT,
           PRIMARY KEY (agent, ticker, round_id))
run_config (run_id TEXT, ts INT, config_json TEXT)
```

Active membership = `live_markets WHERE retired_at IS NULL`. No-trade rounds
get an `auctions` row with `clear_cents NULL, cleared_qty 0`. Unmatched
remainders are cancelled; nothing rests between rounds.

**Done when:** clean create; round-trip test; stream + reader run
concurrently without `database is locked`.

## Stage 3 — stream (SOLE Kalshi caller; on-demand)

Two entry points, both idempotent:

**`refresh_live()`** — the routine path. Cron every `stream_interval` (1h)
via `python -m prediction_markets.stream`, and called by the runner at
round start.

1. Targeted fetch of live ∪ held tickers via `get_markets_by_tickers`
   (one or two batched requests for ~50–100 tickers; this is the entire
   routine Kalshi load).
2. Change-detection writes (snapshot row only if a tracked field changed).
3. If any fetched member's `result` turned non-empty → **refill demand**:
   call `sweep_candidates()` now, so retire/refill decisions read candidate
   data seconds old.
4. `live_markets.maintain()` — always last.

**`sweep_candidates()`** — the candidate-pool path. Full horizon query:
open markets, `close_time` in `[expiry_min_h, expiry_max_h]` (default
24–168h), `mve_filter=exclude`; upsert `markets` metadata verbatim (the
only entry point for agent-visible text) + snapshots (~25k markets, ~92s
measured). Sector narrowing happens client-side off the response's category
field — do not assume a server-side category filter without verifying
against current API docs at build time; if one exists, narrow, else sweep
and filter. Invoked: at seed; on refill demand from `refresh_live()`;
optionally on `archive_sweep_interval` (default `None` = off — a cheap
archival knob if pre-admission Kalshi history or selection-audit data is
ever wanted; data not captured is gone).

A targeted-fetch exception must not prevent `maintain()` from running on
whatever data is fresh; a sweep exception downgrades that run to
retire-without-refill (refill happens on the next successful sweep — log
loudly).

**Done when:** routine runs touch Kalshi with only targeted fetches
(assert request count); a resolution between runs triggers sweep + refill
within the same run; forced sweep failure retires without refill and
recovers on the next run.

## Stage 4 — live_markets (pure DB downstream; no Kalshi access)

Membership logic only. Reads `markets` + `snapshots`, writes `live_markets`.
**Does not import `kalshi_client`** — if data seems missing here, the fix is
in stream, never a direct API call.

- Strata = Kalshi sectors from a pinned `sectors` allowlist in config
  (discover the live vocabulary once, pin, log).
- `top_k(sector, k)` over latest snapshots: qualifying = in-window expiry
  AND 24h volume ≥ `volume_floor` AND Kalshi mid within `[uncertain_lo,
  uncertain_hi]` (default 20–80¢). Rank by 24h volume; ties lexicographic.
  (Selection reads Kalshi prices from the DB; agents never do.)
- `maintain()`, invoked by stream as its last step:
  - **Seed** (first run): per sector admit `top_k(sector,
    markets_per_sector)` (default 8).
  - **Retire on non-empty `result` — and on nothing else.** All admission
    criteria (expiry window, volume floor, uncertainty band) are checked at
    ADD time only. A member whose Kalshi mid later drifts to 5¢ or whose
    volume dries up stays until resolution. Two reasons, recorded here so
    nobody "fixes" this: (a) drifting toward 0/100 is the market
    converging — the phase the internal exchange most needs to live
    through; ejecting converging markets selects the resolution dynamics
    out of the data; (b) membership changes are agent-visible events, so
    band-based retirement would broadcast Kalshi price state into the
    agents' world through the back door — admission leaks one static bit
    (in-band at entry), continuous eviction would leak a running signal.
  - **Refill** each retirement with the top qualifying never-before-admitted
    market from the same sector. Composition constant, identities rotate.
    No fresh candidates available (sweep failed / sector empty) → leave the
    seat open, log, fill on the next successful sweep.
- Deterministic given the DB; log every add/retire with its screening stats.
- Note for later analysis: markets enter at varying ages — any cross-market
  aggregation should align by **time-to-close**, never round index.

**Done when:** seed fills `sectors × markets_per_sector` inside the band; a
simulated resolved snapshot (DB row only, no network) triggers retire +
same-sector refill; a member pushed out of the band by a simulated snapshot
is NOT retired; maintains without resolutions are no-ops.

## Stage 5 — ledger

Signed YES contracts. Fill on own bid = +qty, on own ask = −qty (selling
without inventory = short YES ≡ long NO). Integer cents.

- `initial_cash_cents` (default 100_000).
- **Margin (max-loss, worse-side):** `bid < ask` strictly (Stage 7) means at
  most one side of a quote executes at a uniform price. Admissibility of an
  agent's full quote set: assume per market the worse of {bid fills fully,
  ask fills fully}, jointly across markets vs the pre-round ledger:
  `cash_after + Σ_markets min(value_if_yes, value_if_no) ≥ 0`. Violation →
  reject the whole set, record reason, no partial repair.
- Cash at fill: bid q at p → `−p·q`; ask → `+p·q`.
- `max_position` (±200), `max_lot` (50) per side per market. No cooldowns.
- `settle(round_id)`: yes → 100¢·(+qty); no → 100¢·(−qty); scalar →
  `settlement_value` per YES, `100 − settlement_value` per NO; missing
  scalar value → leave open, never pay zero.
- `pnl(agent)`: realized from settlements + fills; unrealized marked at the
  last internal clear (never at Kalshi). Ledger bookkeeping for the brief,
  not a scoring pipeline.

**Done when:** tests cover worse-side margin, joint cross-market rejection,
short collateral, netting, three settlement cases, hand-computed scalar.

## Stage 6 — auction

Pure: `clear(quotes, last_clear_cents | None)`. Each two-sided quote →
one buy (bid) + one sell (ask); `bid < ask` upstream ⇒ no self-trade is
possible at a uniform price.

- `D(p)` = Σ bid sizes with bid ≥ p; `S(p)` = Σ ask sizes with ask ≤ p;
  maximize `V(p) = min(D, S)` over the grid (standard EOD-auction rule).
- `V_max = 0` → no print, record, cancel all.
- Tie interval: first auction → midpoint rounded toward 50; after → closest
  to last clear (exact tie → toward 50). No external reference price, ever.
- Uniform price; strictly-inside orders fill fully; marginal side pro-rata
  by size rounded down, remainders one each in lexicographic agent-id order.
- Deterministic: replaying `quotes` reproduces `fills` exactly.

**Done when:** brute-force volume-max property test; both tie-breaks;
pro-rata determinism; no-print; proof-test that no price fills both sides of
one agent; golden replay.

## Stage 7 — agents

One structured LLM call per (agent, market) per round. No tool loop.

- Roster in config: `agent_id -> {model, persona, temperature}` =
  families × personas; design goal `n_agents > live market count`.
  Temperature 0 default.
- **Quote frame:** `quote_side ∈ {yes, no}`, one value per run (default
  `yes`). Under `no`, the brief presents the NO contract and the agent
  quotes NO; the parser maps at the boundary (bid NO at p ≡ ask YES at
  100−p, and vice versa) and **everything downstream — ledger, auction,
  storage — stays in YES space**. Comparing a `yes` run and a `no` run on
  matched markets is the framing-bias check. The frame appears in exactly
  two places: brief renderer and response parser.
- Every agent quotes every active market every round, two-sided, mandatory.
  No abstain: "no view" = wide spread.
- v0 prompt: minimal — the mechanism (call auction, uniform price,
  settlement at 100/0), the required JSON, the frame's contract definition.
  Prompt text versioned in config.
- Brief — the agent's inherited history, nothing computed beyond it:
  title/rules/close time, time to close; **full market history** (each past
  round: clear or NO TRADE + volume); **full own history** (past
  bid/ask/sizes, fills); current position + cost basis, cash, realized/
  unrealized PnL from `ledger.pnl`. Other agents' individual quotes are
  never shown.
- Static prefix first (persona, protocol, rules), volatile suffix last —
  prompt caching is a first-order cost lever at this volume and needs a
  stable prefix.
- Required JSON:
  `{"bid_cents": int, "bid_size": int, "ask_cents": int, "ask_size": int}`.
  Validation: `1 ≤ bid_cents < ask_cents ≤ 99` (strict), `1 ≤ sizes ≤
  max_lot`.
- Parse/validation failure → one retry with the error appended → else
  `parse_ok = 0`, no quote this round (recorded).
- `tools.py`: v1 headers only (`web_search`, `get_news` →
  `NotImplementedError`); absent from the v0 prompt and call path.
- Brief + raw response stored verbatim in `calls`; the run must be fully
  reconstructable from the DB.

**Done when:** isolation test passes (`kalshi_client` imported by
`stream.py` alone; brief contains no snapshot price fields); crossed quote
rejected with reason; parse-fail → retry → recorded skip; under
`quote_side=no` a synthetic NO quote round-trips to the correct YES-space
order and back in the brief; one real call per family returns valid JSON.

## Stage 8 — runner (cadence B)

Monotonic schedule at `round_interval`. Per round:

1. `stream.refresh_live()` — fresh `result` state (and any demanded
   sweep + refill) at the round boundary regardless of cron phase; still
   stream code, the runner itself never touches Kalshi.
2. `ledger.settle(round_id)`.
3. Briefs for every (agent, active unresolved market).
4. All calls concurrently, `asyncio.gather` + semaphore (`max_concurrency`
   16). Per-call failure → log, skip; the round continues.
5. Worse-side joint margin validation per agent; write quote rows.
6. Auction per market; apply fills; write rows.
7. Idempotent per `round_id` (PKs enforce); rerunning a crashed round
   continues, never duplicates.

At start: compute + log expected daily calls (`n_agents × live markets ×
rounds/day`) and token estimate; refuse to start above `max_daily_calls`.

**Done when:** 3-round scripted run complete; settlement in the right
round; a refilled market shows empty internal history; kill-and-rerun a
round_id → no duplicates.

---

## Cross-cutting rules

- **`stream.py` is the sole importer of `kalshi_client`**; only it and
  `live_markets.py` may read `snapshots` price fields (live_markets from
  the DB only). Enforced by the Stage 7 test.
- Internal exchange: integer cents/contracts, 1–99 grid; YES space in
  storage regardless of frame. Kalshi floats stay as returned.
- Timestamps Unix seconds UTC. Every constant in `config.py`; full config
  (incl. pinned sectors, expected daily calls) logged to `run_config`.
- Determinism: all randomness from `run_seed`; replays match.
- Zero internal fees in v0 (`fee_cents = 0` stub).
- No mocking Kalshi in integration tests; auction/ledger unit tests use
  synthetic quotes.
- No baselines, no scoring/report code in v0. The DB is the deliverable:
  snapshots (Kalshi benchmark), auctions/quotes/fills (market history),
  cash/positions/settlements (per-agent PnL) — analysis is ad-hoc SQL/
  pandas later.
- Anchored arm (Kalshi prices shown to agents): v1, not built, no dormant
  flag. It enters through the brief renderer alone, when it comes.