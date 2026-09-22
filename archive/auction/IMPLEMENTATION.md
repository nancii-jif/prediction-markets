# v0 Implementation Plan

Execute in stage order. Each stage has a **done when** check — run it before
moving on. [spec.txt](spec.txt) is the authority on design; this file is the
build sequence. Python 3.10+, stdlib `sqlite3`, no framework.

Layout:

```
prediction_markets/
  config.py         # all knobs from spec.txt Config, one place
  kalshi_client.py
  database.py
  listener.py       # runs as its own process
  universe.py
  broker.py
  tools.py
  runner.py
  baselines.py
data/market.db      # gitignored
```

---

## Stage 1 — kalshi_client

Thin wrapper over public REST (`https://api.elections.kalshi.com/trade-api/v2`).
No credentials anywhere in the project.

- `get_markets(**params)` — GET `/markets`, follows `cursor` pagination to
exhaustion, returns list of market dicts.
- `get_markets_by_tickers(tickers)` — batches into GET `/markets?tickers=a,b,..`.
Batch by **encoded URL length, not ticker count**: names run 11–48 chars, and a
fixed count tuned for short ones returns HTTP 414 on long ones.
- Retry with exponential backoff on 5xx/timeouts; raise after ~5 attempts.
Respect 429 with a longer sleep.
- **Always pass** `mve_filter=exclude` on universe-wide queries. Without it,
auto-generated parlay combo markets (~6:1 vs real markets) flood results and
pagination effectively never terminates.

**Done when:** a script fetches all open markets with
`min_close_ts=now, max_close_ts=now+7d, mve_filter=exclude` and prints a count
(expect tens of thousands) with pagination terminating.

**Status vocabulary — request values differ from response values.** Verified
against the live API; a literal `status == "settled"` check never matches:

| request `status=` | response `status` field | `result` |
|---|---|---|
| `open`     | `active`      | `''`           |
| `closed`   | `determined`  | `yes` / `no`   |
| `settled`  | `finalized`   | `yes` / `no`   |
| `unopened` | `initialized` | `''`           |

Treat a market as resolved when **`result` is non-empty** (vocabulary-independent,
covers both `determined` and `finalized`). Stages 4 and 5 depend on this.

## Stage 2 — database

`database.py` owns the schema and all reads/writes. Open with WAL mode
(`PRAGMA journal_mode=WAL`) so the listener process writes while agent
processes read.

```sql
universe  (ticker TEXT PRIMARY KEY, added_at INT, retired_at INT)
snapshots (ticker TEXT, captured_at INT, yes_bid REAL, yes_ask REAL,
           bid_size REAL, ask_size REAL, last REAL, volume_24h REAL,
           status TEXT, result TEXT, PRIMARY KEY (ticker, captured_at))
orders    (id INTEGER PRIMARY KEY, agent TEXT, ts INT, ticker TEXT,
           side TEXT, action TEXT, size INT, status TEXT, reject_reason TEXT)
fills     (id INTEGER PRIMARY KEY, order_id INT REFERENCES orders,
           ts INT, ticker TEXT, signed_qty INT, price REAL, fee REAL,
           snapshot_ticker TEXT, snapshot_captured_at INT,
           FOREIGN KEY (snapshot_ticker, snapshot_captured_at)
             REFERENCES snapshots)
positions (agent TEXT, ticker TEXT, signed_qty INT, cost_basis REAL,
           PRIMARY KEY (agent, ticker))
cash      (agent TEXT PRIMARY KEY, balance REAL)
forecasts (agent TEXT, ticker TEXT, ts INT, p_yes REAL,
           carried_forward INT DEFAULT 0)
turns     (agent TEXT, turn_id INT, ts INT, brief TEXT,
           log TEXT)   -- log: JSON list of every tool call + result
```

Key reads: `latest_snapshot(ticker)`, `latest_snapshots(tickers)`,
`active_universe()` (`retired_at IS NULL`).

**Done when:** schema creates cleanly; insert + read round-trip test passes;
two processes (one writing snapshots, one reading) run concurrently without
`database is locked` errors.

## Stage 3 — listener  ← get this running before building anything else

Own process (`python -m prediction_markets.listener`). Market data cannot be
backfilled; every hour it isn't running is history lost.

Loop, every `poll_interval` (120s):

1. Tickers to poll = active universe ∪ tickers with any nonzero position.
  (Before Stage 4 exists, seed from the Stage 1 query so data starts flowing.)
2. Fetch via `get_markets_by_tickers`.
3. For each market, write a snapshot row **only if any tracked field changed**
  vs the latest stored row (most markets are quiet; this cuts rows 10–50x).
4. Monotonic schedule: `next_t += poll_interval; sleep(next_t - now)` — no drift.
5. Whole iteration wrapped in try/except with logging. The loop never dies.
  Append-only + PK `(ticker, captured_at)` makes restarts idempotent.

Also expose `refresh_ticker(ticker)`: fetch one ticker now, write snapshot row,
return it. The broker calls this at fill time — the *only* non-loop write path,
and it still goes through the same snapshot table.

**Done when:** listener runs ≥30 min unattended; row count grows; killing and
restarting it produces no duplicate/gap errors; a forced exception inside one
iteration is logged and the loop continues.

**Measured:** a full sweep of the ~24.7k-market horizon universe takes ~92s,
against a 120s interval — a ~78% duty cycle with no headroom for growth. Cost is
~3.5ms/market server-side and does not improve much with batching. If this needs
to come down, the options are a narrower universe screen or a hybrid fetch (bulk
horizon query for most markets, targeted `tickers=` only for members the bulk
query no longer returns — settled or out-of-window ones, which is exactly the
set settlement detection depends on).

## Stage 4 — universe

Daily sweep (run at a fixed boundary, midnight ET; invoked by runner or cron):

- Query open markets with `close_time` 1–7 days out (`mve_filter=exclude`).
- New qualifying tickers → insert with `added_at = now`.
- Members whose latest snapshot has a non-empty `result` → stamp `retired_at`
  (see the Stage 1 status-vocabulary table; `status` reads `finalized`, never
  `settled`).
- Horizon is checked at ADD time only; members are never retired for drifting
out of the window. Retirement = settlement only.

**Done when:** two consecutive sweeps on live data: first populates, second
adds new listings and retires settled ones; held-position tickers remain
polled by the listener even after retirement.

**Watch the net growth.** Adds are driven by new listings, retirements only by
settlement, and settlement lags close time — so the active set grows until the
two rates balance. First live sweep went 24,716 -> 29,347 active (+11,801
added, -7,170 retired). Some of that was backlog from a seed that never
retired anything, but if net growth persists at that rate the poll budget is
the binding constraint, not the universe query.

## Stage 5 — broker

All order handling; agents never touch Kalshi. Positions are **signed YES
contracts**: buy NO ≡ sell YES exposure. `side=yes/no` + `action=buy/sell`
collapse to a signed quantity at the boundary; ledger stores one integer per
(agent, market).

Order pipeline (`place_order(agent, ticker, side, action, size)`):

1. Validate: ticker in active universe (unless the order *reduces* an existing
  position — grandfathering), `size <= max_order_size` (50), integer size.
2. Cooldown: reject if this agent's last fill on this ticker is `< interval`
  (24h) ago.
3. Price against fresh data: `listener.refresh_ticker(ticker)`.
4. IOC fill: buy at ask / sell at bid — **never mid, never last**.
  Filled qty = `min(size, displayed size at touch)`; remainder cancelled.
   No book on that side → order cancelled, reason recorded.
5. Fee: Kalshi's current per-contract trading fee formula — fetch the current
  schedule from docs.kalshi.com at build time, implement exactly (including
   its rounding rule), cite the URL in a comment. Do not invent a constant.
6. Check cash (cost + fee ≤ balance, after netting); write order + fill rows.
  Fill row stores the FK to the snapshot from step 3.

Also:

- `settle(agent)` — for held tickers whose latest snapshot has a non-empty
`result` (see Stage 1 status table): realize PnL at `result` (yes → $1 per
+qty, no → $1 per −qty), zero the position, credit cash.
  **Third case:** `result` can also be `"scalar"` — a `market_type: binary`
  market that settled to a *partial* value carried in
  `settlement_value_dollars` (e.g. 0.55/0.45 on esports draws; ~2.6% of
  resolutions). It pays `settlement_value` per YES and `1 - settlement_value`
  per NO, so `snapshots` needs a `settlement_value` column. These cannot be
  filtered out by `market_type`. If the value is missing, leave the position
  open rather than paying zero.
- `pnl(agent)` — realized (from settled fills) + unrealized (open positions
marked at mid of latest snapshot).

**Done when:** unit tests cover: fill at touch not mid; partial fill capped at
displayed size; empty-book cancel; cooldown rejection; buy NO then buy YES
nets correctly; settlement pays the right side; fees match two hand-computed
examples from the fee schedule.

## Stage 6 — tools

Agent-facing functions, exactly the spec list. Each call and its full result
is appended to the current turn's `log` (JSON) before returning.

```
get_markets(ticker=None)   -> latest snapshots for universe (or one ticker):
                              yes_bid, yes_ask, spread, volume_24h, close_time
create_order(ticker, side, size)  -> broker.place_order (action inferred: buy)
                                     + explicit sell via side/sign convention
get_past_trades()          -> own fills
get_pnl()                  -> realized vs unrealized
submit_forecast(ticker, p_yes)    -> forecast row (validate 0<=p<=1)
web_search(query)          -> only if config flag on; log query AND results
```

No tool imports kalshi_client. Enforce with a test that greps tool/agent
modules for the import.

**Done when:** a scripted fake agent exercises every tool and the turn log
replays its complete session from the database alone.

## Stage 7 — runner

One turn per `interval` (24h) per agent, sequentially over agents:

1. `broker.settle(agent)`.
2. Build brief: timestamp, cash, positions with current marks + unrealized
  PnL, settlements since last turn, active universe count. Store verbatim.
3. Invoke agent with brief + tools. Agent may hold — doing nothing is legal
  and not an error.
4. Forecast handling: for each market the agent traded or fetched detail on
  this turn, require a `submit_forecast`; missing → copy previous forecast
   with `carried_forward=1`.
5. Write turn row. Crash in one agent's turn: log, continue to next agent.

**Done when:** a 3-turn run with a scripted agent produces complete turn rows,
correct settlement timing, and carried-forward forecasts where expected.

## Stage 8 — baselines + report

Baselines run through the identical runner/broker path, no special casing:

- `hold-cash` — never trades.
- `random` — each turn: uniformly pick N affordable universe markets, random
side, fixed small size (respects cooldown/max_order_size automatically).
- `buy-and-hold` — first turn: spread starting cash across random markets at
ask; never trades again.

Report script: per agent — final equity, realized/unrealized PnL, fees paid,
trade count, Brier score (forecasts vs resolutions), vs each baseline.

**Done when:** all three baselines complete a multi-turn run and the report
renders from the database of that run.

---



## Cross-cutting rules

- Only `listener.py` and `universe.py` import `kalshi_client`.
- All timestamps: Unix seconds, UTC. All prices: dollars (floats), contracts
pay $1.
- Every constant comes from `config.py`; log the full config at run start
into the database.
- No mocking Kalshi in integration tests — the API is public and free; unit
tests for broker logic use synthetic snapshot rows.

