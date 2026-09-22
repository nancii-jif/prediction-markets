> Module names in the original design below predate the package reorganization. See [architecture.md](architecture.md) for current paths and [usage.md](usage.md) for commands.

# Minimal code migration: agent prediction market

Revised 2026-09-15. Authoritative implementation guide; execute and verify the stages in section 8 in order.

This replaces the previous, larger version of this guide. It is authoritative over conflicting older implementation documents for this MVP. Build only the functionality below. Preserve old run data; do not maintain the old runtime as a second supported mode.

## 1. Smallest runnable system

One local Python process contains:

- **Market:** select ten currently tradable Kalshi markets once, using the earlier available expected/latest expiration within the configured hour window; poll their lifecycle and settle without replacements.
- **News:** on-demand Tavily searches through the shared participant tools; the universe's human-order-book placeholder remains empty.
- **Exchange:** a YES-only continuous order book with funded positions.
- **Agents:** eight independent participants, each with its own tool loop and memory.

Inference uses either one shared Qwen3.5-9B endpoint on Modal or OpenAI's hosted API. The single exchange/orchestration process runs locally or, for OpenAI, on a Modal CPU container. Trades happen exclusively on the internal exchange; never submit orders to Kalshi. No rounds.

Use the existing package and SQLite. Do not introduce a supervisor agent, orchestration framework, broker, local HTTP service, separate agent containers, or distributed exchange.

## 2. Implementation structure

| Current module | Responsibility |
| --- | --- |
| `integrations/kalshi.py` | Public requests, pagination, and network retries. |
| `markets/update.py` | Fetch candidates at startup and refresh only the selected cohort's definitions, timestamps, status and payout. |
| `markets/selection.py` | Apply initial category, horizon, volume and uncertainty filters; select distinct events with category round robin. |
| `exchange/bookkeeping.py` | Pure financial calculations described in section 4. |
| `exchange/engine.py` | Atomic order entry, matching, cancellation, reads and settlement. |
| `agents/participant.py`, `prompt.py`, `history.py` | Participant loop, instructions and private conversation restoration. |
| `agents/tools.py` | Register tools and bind agent identity in code. |
| `storage/database.py`, `schema.py`, `runs.py` | Connections, schema, snapshots, frozen settings and continuation lineage. |
| `runtime/runner.py`, `schedule.py` | One event loop for polling and agents; calendar and daily sessions. |
| `config.py`, `requirements.txt`, `launch.sh`, `README.md` | Remove obsolete required options/dependencies and document one launch path. Do not require W&B, old model-family configuration, or round flags. |

Model transport and token counting live in `integrations/`. Trace, export and
plot commands share `analysis/dataset.py`; historical book reconstruction lives
in `analysis/legacy.py`. The unused `universe.py` placeholder has been removed.

The previous auction source tree, configs, documents and run data are preserved
under [archive/](../archive/README.md), outside the active runtime and test
discovery. `auction.py`, `tracking.py` and `display.py` exist only there. Keep
one supported implementation per operation. Top-level forwarding modules retain
compatibility with previous continuous-market commands and imports. The archived
auction format is not migrated into the current exchange. Current schema-1 run
artifacts support continuation; local artifact retention is an operator choice.

## 3. Market and universe

Startup uncertainty filter: require the initial response's `last_price_dollars`
to lie inside `uncertainty_price_range_cents: [lo, hi]`, inclusive, after
converting the bounds from integer YES cents to dollars. Default bounds are
`[10, 90]`; valid bounds satisfy `0 <= lo <= hi <= 100`. Compare decimal prices
without rounding fractional cents; reject missing or malformed prices without
falling back to bids, asks or legacy fields. Apply this before sector rotation
and volume ranking, and never relax it to fill the configured cohort. Freeze
the bounds with run settings. Lifecycle polling does not reapply this filter:
selected markets remain even if their prices leave the range, and excluded
markets are not reconsidered. No source-price field is added to market rows
or agent briefs. Each market starts with an empty order book and no trades;
all liquidity comes from participant orders, with no programmatic bookkeeper
or synthetic opening prices.

At startup, select ten currently tradable binary markets across distinct events, with at least 500 contracts of 24-hour volume. For the resolution-horizon filter, use the earlier of `expected_expiration_time` and `latest_expiration_time`; if only one is provided, use it, and if both are missing, exclude the candidate. Malformed supplied timestamps remain ineligible. Require the chosen timestamp to fall inclusively between startup UTC plus `expected_expiration_min_hours` and startup UTC plus `expected_expiration_max_hours` (currently 12–72 hours). Use 3,600 seconds per hour; reject old `_days` keys rather than reinterpreting them. Require `status == active` and `open_time <= observation UTC < close_time`; missing or malformed opening times are ineligible. Never substitute close time for expiration. Preserve the two original expiration fields even when latest precedes expected; the earlier value is only a selection rule, not a new database column or settlement trigger. Opening time is also a selection-only input. Fill the cohort one sector at a time: queue eligible markets by their primary series category, then take one from each category in `allowed_market_categories` order, repeating from the top until ten are chosen. Rank within a sector by volume, then ticker; skip empty sectors, and give an allowed cross-listing only its primary sector's turn. Save this fixed cohort once; never reselect or refill during the run, even if a market resolves or its dates change. All eight agents see the same cohort and current statuses through `assigned_markets(agent_id)`. Do not implement differentiated assignment policies.

Before ranking, require series categories to be in the frozen `allowed_market_categories` configuration: Elections, Politics, Entertainment (Culture), Commodities, Climate and Weather (Climate), Economics, Mentions, Financials (Finance), and Science and Technology (Tech & Sciences). Exclude Sports, Crypto, and any other category; missing/malformed metadata is ineligible, and every category in a cross-listing must be allowed. At startup only, fetch `/series` and paginated `/events?status=open&with_nested_markets=true` (200 events/page; excludes multivariate events). Join each event to its series via `series_ticker`, and flatten its nested markets with the series categories; do not use potentially stale event categories or infer categories from titles/tickers. These are selection inputs, not new database columns or agent observations. Do not fetch category metadata again or alter existing cohorts during lifecycle polling.

Expose ticker, title, rules, trading status, and these separately labeled UTC timestamps to agents: `close_time` (trading cutoff), `expected_expiration_time` (expected outcome time), and `latest_expiration_time` (latest expiration). Refresh the values in each model brief from the most recent poll. Close time is not a guaranteed resolution or payout deadline; the expected time is an estimate, and settlement requires the final status/payout regardless of these dates. Preserve missing optional timestamps as unknown. See [Kalshi's time-field definitions](https://docs.kalshi.com/getting_started/market_lifecycle#time-fields). Do not pass Kalshi prices/order books or selection statistics to agent prompts or tools.

After the one-time selection, poll only the fixed cohort's unsettled tickers every 300 seconds; stop polling a ticker after final settlement. No recurring candidate scans, vacancy checks, refill queue, WebSocket listener, or separate process. Existing blocking HTTP fetching may use `asyncio.to_thread`, but the worker must not access the SQLite connection; apply returned data in the local event loop. One poll at a time.

- New orders require active status, a future close time, and a successful observation within the last 600 seconds. A successful unchanged poll refreshes that timestamp too.
- At close, cancel resting orders. Retain positions and keep polling until authoritative final settlement.
- During temporary pause/stale data, do not accept or match orders; cancellations remain allowed.
- Final settlement cancels remaining orders and applies the payout exactly once. Mark the market settled and non-tradable, preserve its history, and do not replace it. The number of tradable markets may decline below ten. If every selected market settles, stop the run normally.
- Never invent a payout or retire/liquidate a market because a timeout elapsed. Missing/unsupported final payouts remain pending with a logged error.
- If fewer than ten eligible markets are available at startup, report it; do not silently relax the filter or launch the ten-market test. Do not add later markets automatically.

`universe.py` contains legacy empty news and human-book values. On-demand news is provided separately by `news.py` and the shared `search_news` tool; no news broadcast or human-book tool is registered. Use the same process UTC time for observations across modules and monotonic time for elapsed waits. No simulated clock, broadcasts, subscriptions, or event bus.

## 4. Exchange and purchase-style accounting

Use integer cents and integer contract quantities, with YES prices 1–99. Fees are zero. These are explicit MVP simplifications, not complete Kalshi parity.

### Cash and positions

Store cash balance `F` and signed YES quantity `q` per market. Positive `q` represents YES holdings; negative `q` represents fully purchased NO-equivalent holdings. Starting cash is 100,000 cents per participant.

For a fill at YES price `p`, with signed trade quantity `d` (buy positive, sell negative):

```text
q_new = q + d
F_new = F - p*d + 100*(max(-q, 0) - max(-q_new, 0))
```

This single formula handles opening, closing, and crossing zero:

- Opening a long costs `p` per contract; selling an owned YES credits `p`.
- Opening a short costs `100-p`; closing a short credits `100-p`.
- Do not also lock short collateral or credit short-sale proceeds separately.

At final YES payout `v` in integer cents, from 0 through 100:

```text
F += max(q, 0)*v + max(-q, 0)*(100-v)
q = 0
```

Accept only supported source precision; never silently round prices, quantities, or payouts.

### Outstanding orders still need funding

Reserve funds before accepting orders so multiple orders cannot spend the same balance. Do not simply sum fresh-short costs for sells that close existing holdings.

For each market, using all remaining orders, including the proposed order:

```text
B = sum(buy_limit_price * remaining_buy_quantity)
S = sum((100 - sell_limit_price) * remaining_sell_quantity)
reserve_market = max(B, S - 100*q) - 100*max(-q, 0)

available_cash = F - sum(reserve_market across markets)
```

Accept only if available cash is nonnegative. This fee-free formula covers any partial-fill combination without assuming both sides fill. Recalculate after fills and cancellations. Reservations reduce available cash, not `F`; fills change `F` using the formula above.

For market orders, reserve the full requested quantity at its protection bound: 99 cents for buys, 1 cent for sells. Reject an unfunded request; do not silently resize it. Unfilled market-order quantity is canceled immediately.

### Matching and reads

- Limit orders match immediately when possible; unmatched quantity rests until cancellation, close, or run shutdown.
- Match best price first, then earliest accepted order ID. Execute at the resting order's price; allow partial fills.
- An order may match the same agent's own resting order. Apply the buy leg first; the two legs net to no change in that agent's cash or position, but the trade still prints and sets the mark other accounts are valued at.
- Cancellation applies only to the caller's own unfilled quantity. A filled order cannot be undone.
- Validate action, ticker membership, price/quantity types and bounds, market status, ownership, and funds in code.
- Serialize exchange access with one local lock. Admission, matching, both accounts, remaining quantities, and trades commit in one SQLite transaction, with no network/inference await inside it.
- Settlement likewise updates all accounts and the market's settled marker in one transaction.

Return cash, reserved/available funds, positions, and total marked PnL. Use the last internal trade as the YES mark `m`:

```text
equity = F + sum(max(q, 0)*m + max(-q, 0)*(100-m))
total_pnl = equity - initial_cash
```

No separate realized/unrealized PnL engine or cost-basis accounting is required. Every nonzero position must have an internal trade to supply its mark.

For conservation tests, derive the settlement pool as `100*sum(max(-q, 0))` across all accounts/markets. Agent cash plus this pool must equal initial total funds; signed positions must sum to zero in each market. No separate pool service or ledger is needed.

## 5. Agent loop and tools

A participant holds its ID, model name, prompt, private conversation, and optional private note. One participant trades its whole portfolio, not one model call per market.

Register only:

```text
submit_order(ticker, action, order_type, quantity, price_cents=None)
cancel_order(order_id)
get_orderbook(ticker)                    # five price levels per side
get_recent_trades(ticker)                # last five internal trades
get_own_orders(ticker=None)              # every order placed, newest first
get_account()
search_news(query)                      # Tavily advanced news search, up to five sources
hold_for_now(memory_note=None)
hold_until_next_day(memory_note=None)  # next local session opening
```

The harness attaches agent identity; tools cannot access another agent's account, order history, or memory. Public book/trade outputs omit participant identities. Own-order history returns every order the agent has placed, open and historical, newest first and unpaginated, with price, side, original/filled/remaining quantity, status, cancellation reason, and time. These runs are short enough that the full history fits the input bound; if it ever stops fitting, the context trimmer drops older conversation groups rather than truncating this tool. These are ordinary local functions, not model requests.

`search_news` is model-independent and uses Tavily's Python SDK with `topic="news"`
and `search_depth="advanced"`. Return source titles, URLs, snippets, scores and
publication dates when available. `news.py` also exports sync/async callables
and a JSON function schema for other agent frameworks. The harness alone binds
identity and enforces the configured elapsed time cost. Searches have no daily
or whole-run count cap. The per-session activity counter in `agent_state` is for
analysis only. Dispatched errors still consume time and increment that counter.
Never automatically retry a search. Read credentials from `tavily_api_key_env`,
defaulting to `TAVILY_API_KEY`, and save only its name. Async news requests have
a 30-second deadline, propagate cancellation, and never hold a SQLite
transaction or block another participant. Treat source content as untrusted data.

Each participant shares the same daily 08:00–16:00 `America/New_York` calendar,
including weekends. Tool costs are real elapsed minutes, configured in YAML:
`order_tool_minutes: 2`, `read_tool_minutes: 5`, `news_tool_minutes: 10`, and
`hold_for_now_minutes: 60`. Own-order and account reads use the same five-minute
cost as book and trade-history reads. The account in each brief is automatic.

The independent loop waits for an open session, builds a fresh brief, and makes
one inference request. Each response can dispatch at most one tool. Start its
cost at dispatch, return the result immediately when available, and start the
next inference while the cost runs. Execute that next action only after both
inference and the previous cost finish. Tool network latency consumes the same
interval. Validate funding/status at execution; never retry a state-changing
operation. Malformed or wholly refused responses get the configured retry delay.
A response without tools implies `hold_for_now`.

Order, read, and news tools have no call-count quotas. Their activity counters
remain for analysis and reset on entering a new session, without limiting
dispatch. Every brief reports session timing; it carries no remaining-call budget.

All agent orders are one-sided: buy/sell, market/limit, quantity, and a limit
price when applicable. Remove `submit_quote` from the registered agent tools.
Books start empty and trading remains optional. Agents form their beliefs from
rules, news, and other agents' observed orders/trades.

`hold_for_now` consumes its entire configured elapsed duration, even across a
close or a later opening. `hold_until_next_day` relinquishes the current session
until the next day's opening. Both may replace the private note. When a tool's
cost or execution reaches/passes close, finish it, retain its result as EOD data,
and defer inference until the first open session after completion. Thus a
60-minute hold at 15:30 ends at 16:30, followed by waiting until 08:00. Discard
execution of any inferred action that misses its original session; record an
explicit not-executed result and decide afresh next session. There are no new
overnight model requests. Resting orders and positions persist; polling and
settlement continue. No event-triggered wakeup interrupts a hold.

The production clock remains real UTC time; IANA local dates determine session
boundaries and monotonic time measures elapsed costs. Tests inject fake clocks.
Persist the action deadline and status in `agent_state`; record scheduling,
inference start/completion, and cost completion in transcript timing events.

Bound each model request to 1,024 generated tokens and 120 seconds. Measure the fixed prefix (system prompt plus tool schemas) once with the model tokenizer and allow `working_token_budget` tokens on top of it, so a longer prompt raises the limit instead of displacing history. Serve that prefix from the endpoint's prefix cache, since it is identical for every participant and every turn. Keep input within the resulting limit by dropping oldest complete conversation/tool groups; never cut out mandatory market rules. Fail before any request if the measured prefix, working budget and output budget cannot fit the server context. Keep an optional private note of at most 2,000 characters. Reject truncated/malformed tool output rather than guessing an order.

On transient inference failure, log it and sleep 60 seconds before the next attempt; do not build a retry framework. Invalid tools return structured errors. Permanent configuration/authentication failures stop the run visibly. Never automatically retry a state-changing tool.

One in-flight request per participant permits up to eight concurrently. The example YAMLs use 23 real hours, including closed hours, within the existing Modal runner limit. Stop when duration expires or all selected markets settle. Time limits order, read, and news activity. There is no separate inference-attempt quota.

## 6. Model boundary and hosting

`model_client.py` provides one async Chat Completions function taking messages, tool schemas, model name, provider and endpoint configuration, returning a parsed model response. Inject this function into participants so tests can replace it with a scripted fake. `model_provider` defaults to `vllm` for existing configurations; `openai` uses the same tools, histories, budgets and exchange. Its transport sends `max_completion_tokens` and disables parallel tool calls, omits vLLM-only fields, and supports optional `openai_reasoning_effort` and nullable temperature/top_p. OpenAI model versions belong in `model_name`, with `model_revision: null`. Save returned model identity, usage and fingerprint with the response. Never automatically retry tool execution.

For vLLM, count input using the pinned Hugging Face chat template. OpenAI uses a local tiktoken estimate of messages and schemas plus framing headroom; keep the working-budget trimming policy and validate the configured context guard. The estimate is not exact API usage. OpenAI requires no model weights, Hugging Face download or GPU endpoint warmup. Reasoning tokens count toward the OpenAI output cap.

`deploy/modal_qwen.py` serves Qwen3.5-9B with a compatible, tested vLLM release and tool parser. Start with one A100 40-GB GPU, one replica, prefix caching enabled, and a 20,480-token context; verify this fits the actual workload rather than assuming it does. Enable concurrent serving, disable thinking for this first configuration, and authenticate the endpoint. Record the tested model revision/server version in the launch instructions.

Keep Modal imports in deployment files. Agents remain provider-independent; the runner binds endpoint credentials from environment variables. Do not send the database or other credentials to inference hosting. No autoscaling, multi-GPU planning, load balancer, or serving benchmark suite.

`deploy/modal_openai.py` optionally runs the same orchestration/exchange on one CPU container without a GPU. Upload only the runtime package and validated settings, and supply API keys through a Modal Secret. Keep active SQLite on container-local storage, then copy closed run artifacts to a persistent Volume on completion or handled failure. Continue stopped artifacts in a new child run using their frozen configuration and original market cohort. Abrupt container loss before artifact persistence is not recovered. The OpenAI-only path does not stop an independently deployed Qwen GPU service.

Document how to deploy and stop this single endpoint. Do not provision billable compute as part of writing or reviewing this guide.

## 7. Storage and runner

Fresh experiments claim a new run directory and SQLite file. Continuations copy a stopped parent into a new child directory, preserving the parent and its configuration. Never overwrite an existing run. The historical auction schema is not supported.

Store only:

- Frozen run settings; market definitions, current status/payout, last successful poll, and membership.
- Account balances and signed positions.
- Orders, trades, and a settled marker unique per market.
- Per-agent transcript entries containing model responses and tool results; save its private note, current activity-day start, per-market order-tool counts, shared read-tool count, and scheduled wake time when they change.

Use ordinary integer order/trade IDs for ordering. Query the small order book from SQLite; a second in-memory book/cache is unnecessary. Save the latest market state, not a historical Kalshi benchmark archive.

The runner starts polling and the eight loops after selection succeeds and after the serving endpoint answers a readiness probe with the pinned model. Hosted containers hold requests while booting, so an unready endpoint is indistinguishable from a hang; probe once centrally rather than letting every participant expire its own deadline. Rejected credentials or a different served model stop the run immediately; transport failures and other statuses are retried until the readiness budget expires. On normal stop, stop agent tasks, cancel outstanding orders, commit, and close. Preserve positions and transcripts in the run artifact; mark unresolved positions as such. Do not fabricate a final payout. A crash ends the process. Persisted stopped runs can be explicitly continued in a new child run; process restarts do not automatically replay tools. Carry balances, positions, private history and unfinished elapsed costs, refresh only the original cohort, and settle each market at most once.

Keep only configuration read by this runtime: endpoint/model credentials by environment; market filter/count/poll/freshness settings; participant count/initial cash; request/context limits; session hours and per-tool elapsed time costs; and the run duration. Use one `configs/mvp.yaml`. Defaults may live in code and be captured in the saved settings; do not require irrelevant old keys. Use the existing Python version if the needed local dependencies support it; no unrelated interpreter migration.

One launch command to implement and document:

```bash
python -m prediction_markets.runner --config configs/mvp.yaml
```

`launch.sh` may forward to it. No separate stream process, restart daemon, tmux requirement, reporting service, dry-run CLI, or settlement-only CLI. Ordinary logs of errors, orders, fills, and final account totals are sufficient. `python -m prediction_markets.trace` may read a run database afterwards or alongside it, read-only; it is an inspection aid and never affects a run.

## 8. Implementation order and minimum verification

Complete and verify only one step at a time; do not begin the next step without explicit approval.

1. Establish the foundation: replace the runtime configuration, atomically claim a new run directory, create the fresh schema, freeze the resolved settings, initialize eight accounts and their activity-day state, and enforce one-time storage of an exact ten-market fixed cohort.
2. Implement purchase-style accounting, reservations, and continuous matching; test with scripted orders.
3. Wire one-time selection, fixed-cohort polling, final settlement without replacement, and the empty universe.
4. Add tools and independent participant loops using a fake model, including elapsed tool costs, session boundaries, and holds.
5. Connect the single real endpoint, finish the runner and launch documentation, and perform the bounded live test only when deployment and live execution are authorized.

Implementation status (2026-09-15): Steps 1–4 and Step 5's local client/runner/deployment-script wiring are implemented and verified offline. The actual pinned Qwen tokenizer was checked locally; Modal GPU deployment, server workload validation and the bounded live experiment have **not** been performed. [README.md](README.md) records the candidate serving pins, setup and remaining authorized-live checks. Do not treat local tests as live acceptance.

Required offline tests, with temporary databases:

- Fresh short, cover, long/short flip, both settlement outcomes, and cash conservation. Example: 10-contract short at 40 cents takes cash from $100 to $94; covering at 30 produces $101; settling YES instead leaves $94, NO produces $104.
- Multiple orders cannot spend the same cash; closing existing holdings still works with zero unreserved cash; partial fills and cancellations update reservations correctly.
- Price-time matching, maker price, partial fills, unmatched market-order cancellation, self-matching that leaves cash and position unchanged while still printing, invalid inputs, and order ownership.
- Selection filters the earlier available expected/latest expiration against the configured inclusive hour window, never close time or a days-based interval. Use the sole provided expiration when the other is missing; exclude missing-both or malformed timestamps. Preserve the original time fields and accept latest-before-expected when the earlier value qualifies. Active status alone is insufficient: selection also requires `open_time <= now < close_time`. Agent briefs show the separately labeled time fields.
- Selection applies the nine-category series allowlist before volume ranking; Sports/Crypto, disallowed cross-listings, and unknown categories cannot enter the cohort, even at higher volume.
- The cohort rotates through sectors in configuration order and wraps around once each has had a turn; a high-volume sector cannot claim every slot, exhausted sectors drop out of later passes, a skipped duplicate event does not cost a sector its turn, and an allowed cross-listing occupies only its primary sector.
- Close blocks trading; missing results never create payouts; settlement happens once without replacement or another discovery scan; unchanged polls refresh freshness. The initial cohort stays fixed and polling stops for settled tickers.
- Private memory/account isolation; no Kalshi prices in agent observations; read tool → order tool → wait works; a slow/sleeping agent does not block another.
- Tool results reach inference before their cost finishes; fast inference waits only the remaining cost, and slow inference adds no further cost. Tool network latency counts in the same interval.
- No new tools or inference start outside the session. A pending tool finishes after close; its result persists into a fresh next-session brief. Inference crossing close never executes a stale order.
- Holds use elapsed time, survive closes/openings, and use the configured duration. A 15:30 one-hour hold ends at 16:30, then resumes decisions at 08:00. IANA local dates handle DST and weekend openings.
- Resting orders can fill while their agent holds; one sleeping/thinking/searching agent never blocks others. News searches continue beyond the former count cap while consuming their configured elapsed time cost.


Live acceptance: ten initially selected markets with the specified resolution horizon and eight Qwen participants each complete a valid tool loop; logs contain actual responses, order outcomes, and final balances. Zero trades is possible and must be reported honestly. Offline scripted fills establish matching correctness independently. Report any untested live settlement path.

**Do not add:** crash recovery, event sourcing/replay, durable wake notifications, rolling rate limits, prompt/version registries, distributed components, dashboards/W&B, human benchmark collection, extra models, fee engines, cross-market collateral offsets, or research evaluation. They are not needed to run this MVP.
