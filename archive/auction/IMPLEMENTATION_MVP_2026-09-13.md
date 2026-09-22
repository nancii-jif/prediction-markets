# Minimal code migration: agent prediction market

Revised 2026-09-14. Specification only; application changes are not implemented.

This replaces the previous, larger version of this guide. It is authoritative over conflicting older implementation documents for this MVP. Build only the functionality below. Preserve old run data; do not maintain the old runtime as a second supported mode.

## 1. Smallest runnable system

One local Python process contains:

- **Market:** select ten Kalshi markets once, expected to resolve in 72–168 days; poll their lifecycle and settle without replacements.
- **Universe:** empty news and human-order-book placeholders.
- **Exchange:** a YES-only continuous order book with funded positions.
- **Agents:** eight independent participants, each with its own tool loop and memory.

Only inference runs remotely: one shared Qwen3.5-9B endpoint on Modal. Trades happen exclusively on the internal exchange; never submit orders to Kalshi. No rounds.

Use the existing package and SQLite. Do not introduce a supervisor agent, orchestration framework, broker, local HTTP service, separate agent containers, or distributed exchange.

## 2. File-level migration

| Existing file | Minimum change |
| --- | --- |
| `kalshi_client.py` | Reuse public requests, pagination, and existing network retries. |
| `stream.py` | Fetch candidates once at startup, then poll only selected unsettled tickers for definitions/timestamps/status/payout. Return fetched data before writing it locally; remove the standalone stream process and recurring discovery sweeps. |
| `live_markets.py` | Select ten markets across distinct events once. Remove refills, sector quotas, human-price filtering, and timeout retirement. |
| `ledger.py` | Replace unbounded borrowing and signed-cash settlement with section 4. |
| `auction.py` | Remove from the runtime/import path. Add `exchange.py` for order entry, matching, cancellation, and reads. |
| `agents.py` | Replace `quote()`/`gather_quotes()` with `Participant.run()`, private conversation state, and section 5's loop. Keep participant model and prompt constructor arguments. |
| `tools.py` | Implement only the tools listed below. Bind agent identity in code, not model arguments. |
| `database.py` | Use a fresh minimal schema for the new run; retain useful connection/query helpers. No old-database migration. |
| `runner.py` | Initialize once; run market polling and eight agent tasks in one event loop. Delete round scheduling and auction recovery. |
| `config.py`, `requirements.txt`, `launch.sh`, `README.md` | Remove obsolete required options/dependencies and document one launch path. Do not require W&B, old model-family configuration, or round flags. |

Add only `universe.py`, `model_client.py`, and `deploy/modal_qwen.py` besides `exchange.py`. Keep the harness inside `agents.py`; no separate harness framework or clock class.

Leave historical files/data alone unless an implementation change needs the file. Unused `tracking.py`/`display.py` need not be rewritten or imported. No replacement dashboard.

## 3. Market and universe

At startup, select ten active binary markets across distinct events, with at least 500 contracts of 24-hour volume and `expected_expiration_time` inclusively between startup UTC plus 72 days and startup UTC plus 168 days. These are days, not hours; use 86,400 seconds per day. Require a future `close_time`; skip candidates without a usable expected expiration rather than substituting close time. Rank by volume, then ticker. Save this fixed cohort once; never reselect or refill during the run, even if a market resolves or its dates change. All eight agents see the same cohort and current statuses through `assigned_markets(agent_id)`. Do not implement differentiated assignment policies.

Expose ticker, title, rules, trading status, and these separately labeled UTC timestamps to agents: `close_time` (trading cutoff), `expected_expiration_time` (expected outcome time), and `latest_expiration_time` (latest expiration). Refresh the values in each model brief from the most recent poll. Close time is not a guaranteed resolution or payout deadline; the expected time is an estimate, and settlement requires the final status/payout regardless of these dates. Preserve missing optional timestamps as unknown. See [Kalshi's time-field definitions](https://docs.kalshi.com/getting_started/market_lifecycle#time-fields). Do not pass Kalshi prices/order books or selection statistics to agent prompts or tools.

After the one-time selection, poll only the fixed cohort's unsettled tickers every 120 seconds; stop polling a ticker after final settlement. No recurring candidate scans, vacancy checks, refill queue, WebSocket listener, or separate process. Existing blocking HTTP fetching may use `asyncio.to_thread`, but the worker must not access the SQLite connection; apply returned data in the local event loop. One poll at a time.

- New orders require active status, a future close time, and a successful observation within the last 300 seconds. A successful unchanged poll refreshes that timestamp too.
- At close, cancel resting orders. Retain positions and keep polling until authoritative final settlement.
- During temporary pause/stale data, do not accept or match orders; cancellations remain allowed.
- Final settlement cancels remaining orders and applies the payout exactly once. Mark the market settled and non-tradable, preserve its history, and do not replace it. The number of tradable markets may decline below ten. If every selected market settles, stop the run normally.
- Never invent a payout or retire/liquidate a market because a timeout elapsed. Missing/unsupported final payouts remain pending with a logged error.
- If fewer than ten eligible markets are available at startup, report it; do not silently relax the filter or launch the ten-market test. Do not add later markets automatically.

`universe.py` contains empty news and human-book values only; register no tools for them. Use the same process UTC time for observations across modules and monotonic time for elapsed waits. No simulated clock, broadcasts, subscriptions, or event bus.

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
- If the next match would be against the same agent, cancel the incoming remainder instead.
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
get_own_orders(ticker=None, before_id=None) # newest 20, older pages by ID
get_account()
wait(seconds=60, memory_note=None)
wait_until_next_day(memory_note=None)    # suspend this agent for exactly 8 hours
```

The harness attaches agent identity; tools cannot access another agent's account, order history, or memory. Public book/trade outputs omit participant identities. Own-order history includes price, side, original/filled/remaining quantity, status, and time. These are ordinary local functions, not model requests.

Each participant independently repeats:

1. Build input from its prompt, current market/account information, private note, and recent conversation.
2. Await one model response.
3. Append that response; execute requested tools sequentially and append their results.
4. Call the model again with those results, unless it requested either wait tool.
5. For either wait tool, update the note if supplied and use a cancellable `asyncio.sleep` outside the exchange lock/transaction; then observe fresh state and continue.

A wait ends dispatch of that response; give any subsequent tool calls explicit not-executed results. A response without tool calls implies the default wait. Resting orders can fill while their agent sleeps. There are no event-triggered wakeups, rounds, shared snapshots, or barriers.

Use a 60-second default for `wait` and validate its requested duration to 1–300 seconds. `wait_until_next_day` instead sets this agent's next eligible action time to invocation time plus 28,800 seconds (eight hours); it is not subject to the short-wait limit. Despite its name, it does not mean midnight or a calendar-day boundary. Record the requested wake time in the tool result. Make no inference calls for this agent during the wait; other agents, market polling, fills, and settlements continue. Preserve memory, funds, positions, and resting orders without any daily reset.

For this live-data MVP, advancing an agent's individual clock means scheduling its next activation eight real hours later, not instantly assigning it a future observation timestamp. On waking, use current shared UTC time and fresh exchange state. There is no simulated fast-forward or new clock subsystem. Run shutdown cancels the sleep; the default 30-minute test will therefore stop before an eight-hour waiter resumes. Do not extend the run automatically. This timer-only observation policy is a deliberate minimum, not a calibrated model of human reaction time. Explain both wait tools in the prompt, allow trading and abstention, and do not force two-sided quotes.

Bound each model request to 1,024 generated tokens and 120 seconds. Keep input within 12,000 tokens using the model tokenizer, dropping oldest complete conversation/tool groups; never cut out mandatory market rules. Keep an optional private note of at most 2,000 characters. Reject truncated/malformed tool output rather than guessing an order.

On transient inference failure, log it and sleep 60 seconds before the next attempt; do not build a retry framework. Invalid tools return structured errors. Permanent configuration/authentication failures stop the run visibly. Never automatically retry a state-changing tool.

One in-flight request per participant permits up to eight concurrently. For the first test, cap each participant at 100 inference attempts and the whole run at 30 minutes. Stop when duration expires or all participants exhaust their caps. These are simple run-cost bounds, not rolling quotas; capped agents' orders remain active until run shutdown.

## 6. Model boundary and hosting

`model_client.py` needs one async function taking messages, tool schemas, model name, and endpoint configuration, returning a parsed model response. Use an HTTP client against the serving endpoint. Inject this function into participants so tests can replace it with a scripted fake; do not implement additional provider adapters now.

`deploy/modal_qwen.py` serves Qwen3.5-9B with a compatible, tested vLLM release and tool parser. Start with one A100 40-GB GPU, one replica, and a 16,384-token context; verify this fits the actual workload rather than assuming it does. Enable concurrent serving, disable thinking for this first configuration, and authenticate the endpoint. Record the tested model revision/server version in the launch instructions.

Keep all Modal imports here. Agents only know the endpoint URL, model ID, and credential from environment variables. Do not send the database or other credentials to inference hosting. No autoscaling, multi-GPU planning, load balancer, or serving benchmark suite.

Document how to deploy and stop this single endpoint. Do not provision billable compute as part of writing or reviewing this guide.

## 7. Storage and runner

Use a new run directory and fresh SQLite file; refuse to overwrite or resume an existing run. Reuse SQLite itself, not the old auction schema.

Store only:

- Frozen run settings; market definitions, current status/payout, last successful poll, and membership.
- Account balances and signed positions.
- Orders, trades, and a settled marker unique per market.
- Per-agent transcript entries containing model responses and tool results; save its private note when changed.

Use ordinary integer order/trade IDs for ordering. Query the small order book from SQLite; a second in-memory book/cache is unnecessary. Save the latest market state, not a historical Kalshi benchmark archive.

The runner starts polling and the eight loops after selection succeeds. On normal stop, stop agent tasks, cancel outstanding orders, commit, and close. Preserve positions and transcripts in the run artifact; mark unresolved positions as such. Do not fabricate a final payout. A crash ends the experiment: records remain for inspection, but automated recovery/resume is out of scope.

Keep only configuration read by this runtime: endpoint/model credentials by environment; market filter/count/poll/freshness settings; participant count/initial cash; request/context/wait limits; run duration and per-agent attempt cap. Use one `configs/mvp.yaml`. Defaults may live in code and be captured in the saved settings; do not require irrelevant old keys. Use the existing Python version if the needed local dependencies support it; no unrelated interpreter migration.

One launch command to implement and document:

```bash
python -m prediction_markets.runner --config configs/mvp.yaml
```

`launch.sh` may forward to it. No separate stream process, restart daemon, tmux requirement, reporting service, dry-run CLI, or settlement-only CLI. Ordinary logs of errors, orders, fills, and final account totals are sufficient.

## 8. Implementation order and minimum verification

1. Implement the fresh schema, accounting, and matching; test with scripted orders.
2. Wire one-time selection, fixed-cohort polling, settlement without replacement, and the empty universe.
3. Add tools and independent loops using a fake model.
4. Connect the single real endpoint and perform the bounded live test when deployment/run is authorized.

Required offline tests, with temporary databases:

- Fresh short, cover, long/short flip, both settlement outcomes, and cash conservation. Example: 10-contract short at 40 cents takes cash from $100 to $94; covering at 30 produces $101; settling YES instead leaves $94, NO produces $104.
- Multiple orders cannot spend the same cash; closing existing holdings still works with zero unreserved cash; partial fills and cancellations update reservations correctly.
- Price-time matching, maker price, partial fills, unmatched market-order cancellation, self-trade prevention, invalid inputs, and order ownership.
- Selection uses expected expiration 72–168 days ahead, not close time or an hours-based interval; missing expected times are excluded. Agent briefs show the separately labeled time fields.
- Close blocks trading; missing results never create payouts; settlement happens once without replacement or another discovery scan; unchanged polls refresh freshness. The initial cohort stays fixed and polling stops for settled tickers.
- Private memory/account isolation; no Kalshi prices in agent observations; read tool → order tool → wait works; a slow/sleeping agent does not block another.
- Eight-hour wait sets only the caller's wake deadline, makes no inference calls while sleeping, preserves its state/orders, and is canceled promptly by shutdown. Stub the sleep/time functions in this test; do not actually wait eight hours or add a production simulated clock.

Live acceptance: ten initially selected markets with the specified resolution horizon and eight Qwen participants each complete a valid tool loop; logs contain actual responses, order outcomes, and final balances. Zero trades is possible and must be reported honestly. Offline scripted fills establish matching correctness independently. Report any untested live settlement path.

**Do not add:** crash recovery, event sourcing/replay, durable wake notifications, rolling rate limits, prompt/version registries, distributed components, dashboards/W&B, human benchmark collection, news/search tools, extra models, fee engines, cross-market collateral offsets, or research evaluation. They are not needed to run this MVP.
