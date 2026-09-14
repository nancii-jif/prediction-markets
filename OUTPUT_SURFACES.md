# Output surfaces

Every distinct message this project emits, with one worked example each, for
reformatting. Two audiences that must never be confused:

| Audience          | Written by                                   | Goes to                | May contain Kalshi data |
| ----------------- | -------------------------------------------- | ---------------------- | ----------------------- |
| **Agents** (LLMs) | `agents.build_prompt`, `agents.render_brief` | the model, via litellm | **never**               |
| **Operator**      | `display.*`, via `print()` in `runner.py`    | stdout                 | yes                     |

Log output (`log.*` → stderr and `runs/<name>/*.log`) is a third surface, not
covered here.

Every example below is captured verbatim from a seeded database — no
transcription. Config values behind them: 9 agents (`haiku-1..3`,
`sonnet-1..3`, `opus-1..3`), `MAX_LOT: 50`, `MAX_POSITION: 200`,
`QUOTE_SIDE: "yes"`, `INITIAL_CASH_CENTS: 100000`.

---

# Part A — Agent-facing

Exactly two strings reach a model per call: the system prompt (static, cached)
and the brief (volatile). Nothing else. Both are archived per call in the
`calls` table.

## A1. System prompt

`agents.build_prompt()` → `agents.PROTOCOL` formatted with the contract
definition and `MAX_LOT`. Byte-stable across every call in a run, which is what
the prompt cache holds onto.

```text
You are a market maker on an internal prediction-market exchange.

HOW THE EXCHANGE WORKS
- Trading happens in discrete rounds. In each round you post a two-sided quote
  on each market: a bid (the price at which you will buy) and an ask (the price
  at which you will sell), and the respective lot sizes.
- Each round, all quotes are collected and cleared together in a call auction.
  The clearing price is determined such that the number of contracts can be traded is maximal.
    - If your bid is at or above the clearing price you buy; if your ask is at or below it you sell. Anything that does not execute is cancelled.
    - It could be that no trades can take place i.e., all bids are lower than any ask.
- Each round, if trades take place, the exchange will post the clearing price as well as traded volume and number of unfilled orders at that price; otherwise, the exchange will post the best bid and ask and the total size quoted respectively.
- Prices are whole cents from 1 to 99. Each contract pays 100 cents if the
  event happens and 0 cents if it does not.
- You may hold a negative (short) position. Selling contracts you do not own is
  allowed and requires no collateral, so your cash may go negative.

WHAT YOU ARE TRADING
You are quoting the YES contract. It pays 100 cents if the
market's stated condition is met, and 0 cents otherwise. Your bid is what you
will pay to buy YES; your ask is what you will accept to sell YES.

YOUR TASK
- Every round, you must quote both sides on the market described below. You should bid lower than your ask.
- Your quotes should reflect your view and confidence in the probability of the event happening.
    - If you have no view or are uncertain, quote a wide spread.
- You should rely on your knowledge, rules of this market, and the trading history on the exchange, as well as your position history.

Respond with ONLY a JSON object, no other text:
{"bid_cents": <int>, "bid_size": <int>, "ask_cents": <int>, "ask_size": <int>}

Constraints: 1 <= bid_cents < ask_cents <= 99, and each size is
a whole number of contracts from 1 to 50.
```

**Two swappable blocks inside it.**

`WHAT YOU ARE TRADING` under `QUOTE_SIDE: "no"` (`agents.NO_CONTRACT`):

```text
You are quoting the NO contract. It pays 100 cents if the
market's stated condition is NOT met, and 0 cents if it is met. Your bid is
what you will pay to buy NO; your ask is what you will accept to sell NO.
```

A persona, when set, is prepended as its own paragraph before
`You are a market maker...`:

```text
You are cautious.

You are a market maker on an internal prediction-market exchange.
...
```

## A2. Brief — first round in a market

`agents.render_brief()`. Five labelled blocks, blank-line separated.

```text
MARKET: Bitcoin above $120,000 on Dec 31?
Ticker: KXBTCMAXY-26-120000

RESOLUTION RULES
Resolves YES if the Coinbase BTC/USD price exceeds $120,000.00 at 5pm ET.

Closes in 14h 0m (round 1)

EXCHANGE HISTORY FOR THIS MARKET
No rounds have been held yet. This is the first auction.

YOUR HISTORY IN THIS MARKET
You have not quoted this market before.

YOUR ACCOUNT
  Position in this market: flat
  Cash: 100000c
  Realised PnL: +0c
  Unrealised PnL: +0c
  Position limit: +/-200 contracts; max 50 contracts per side per round

Post your quote now, as JSON only.
```

## A3. Brief — with full history and an open position

```text
MARKET: Will the Democrats win the 2026 House?
Ticker: KXPRESVOTE-26-DEM

RESOLUTION RULES
Resolves YES if the Democratic Party holds 218 or more seats in the U.S. House of Representatives following the 2026 midterm elections, per the Associated Press call.

Closes in 1d 16h (round 4)

EXCHANGE HISTORY FOR THIS MARKET
  round 1: cleared at 47c, volume x18, unfilled 0
  round 2: no trade, best bid 40c x8, best ask 55c x9
  round 3: cleared at 47c, volume x4, unfilled bid x5

YOUR HISTORY IN THIS MARKET
  round 1: 48c x10 at 55c x10, filled +10 at 47c
  round 2: 30c x4 at 70c x6, no fill
  round 3: 46c x6 at 52c x6, no fill

YOUR ACCOUNT
  Position in this market: +10 contracts, average cost 47c
  Cash: 99530c
  Realised PnL: +0c
  Unrealised PnL: +0c
  Position limit: +/-200 contracts; max 50 contracts per side per round

Post your quote now, as JSON only.
```

## A4. Every line variant inside the brief

Each of these is one line the brief can emit. Collected here because they are
what a reformat has to cover.

### Header / metadata

```text
MARKET: Will the Democrats win the 2026 House?
Ticker: KXPRESVOTE-26-DEM
```

```text
RESOLUTION RULES
Resolves YES if the Democratic Party holds 218 or more seats...
```

```text
RESOLUTION RULES
(not provided)
```

Rules are inserted verbatim from Kalshi, unwrapped — often several hundred
characters on one line. This is the longest single line in the brief.

```text
Closes in 1d 16h (round 4)
Closes in 14h 0m (round 1)
Closes in closed (round 9)          <- _format_duration returns "closed" at <= 0
Close time unknown (round 4)        <- market.close_ts is NULL
```

### `EXCHANGE HISTORY FOR THIS MARKET`

```text
No rounds have been held yet. This is the first auction.
```

```text
  round 1: cleared at 47c, volume x18, unfilled bid x5
  round 2: cleared at 58c, volume x30, unfilled ask x8
  round 3: cleared at 50c, volume x20, unfilled 0
  round 4: no trade, best bid 40c x8, best ask 55c x9
  round 5: no trade                          <- no accepted quotes behind it
```

`unfilled <side> x<n>` is the imbalance at the clearing price:
`demand(p) - supply(p)`, computed by `auction.imbalance` from the same accepted
quotes the auction cleared. `bid` means that much buying interest sat at or
above the print and could not be matched; `ask` is the mirror. Size is summed
across the whole side, never per agent.

Degenerate book variants from `agents._format_book`, reachable only if a quote
row is one-sided:

```text
  round 4: no trade, no bid, best ask 55c x9
  round 4: no trade, best bid 40c x8, no ask
```

### `YOUR HISTORY IN THIS MARKET`

```text
You have not quoted this market before.
```

```text
  round 1: 48c x10 at 55c x10, filled +10 at 47c
  round 2: 30c x4 at 70c x6, no fill
  round 3: 48c x9 at 51c x9, filled +4 at 47c      <- pro-rata partial, 9 quoted
  round 4: 40c x6 at 44c x12, filled -12 at 47c    <- negative = sold
  round 5: quote rejected (KXPRESVOTE-26-DEM: a full bid fill would take the position to 215, past the +/-200 cap)
```

The fill quantity is signed rather than worded (`filled +10` / `filled -12`),
matching the sign convention of the position line below it.

The rejection reason is `ledger.validate_quote_set`'s string, pasted in whole.
There are three, and only three — the solvency check is gone, so a set is only
ever rejected for being malformed or breaching the position cap:

```text
  round 5: quote rejected (KXPRESVOTE-26-DEM: prices must satisfy 1 <= bid < ask <= 99, got bid=60 ask=40)
  round 5: quote rejected (KXPRESVOTE-26-DEM: bid_size must be 1..50, got 80)
  round 5: quote rejected (KXPRESVOTE-26-DEM: a full bid fill would take the position to 215, past the +/-200 cap)
```

### `YOUR ACCOUNT`

```text
  Position in this market: flat
  Position in this market: +10 contracts, average cost 47c
  Position in this market: -8 contracts, average cost 47c
  Cash: 99530c
  Realised PnL: +0c
  Unrealised PnL: -240c
  Position limit: +/-200 contracts; max 50 contracts per side per round
  no position limit; max 50 contracts per side per round   <- MAX_POSITION: null
```

Cash and both PnL figures may be negative without limit: shorts are
uncollateralised.

### Closer

```text
Post your quote now, as JSON only.
```

## A5. Retry message

When `parse_quote` rejects a response, the model gets its own reply back plus
one user turn, then one more attempt (`LLM_MAX_RETRIES: 1`).

```text
That response was rejected: response was not valid JSON (Expecting ',' delimiter: line 1 column 34 (char 33)). Reply with only the JSON object.
```

The `{exc}` slot is one of the `QuoteError` strings:

```text
empty response
no JSON object found in the response
response was not valid JSON (<json.JSONDecodeError message>)
missing required field bid_size
bid_cents must be a whole number, got '40'
bid_cents must be a whole number, got 40.5
need 1 <= bid_cents < ask_cents <= 99, got bid=60 ask=40
bid_size must be 1..50, got 0
```

## A6. What the agent sends back

The entire expected output, parsed by `agents.parse_quote`:

```json
{"bid_cents": 44, "bid_size": 10, "ask_cents": 51, "ask_size": 10}
```

A fenced or prose-wrapped object is accepted — the parser takes the first
`{...}` span.

## A7. Frame inversion

Under `QUOTE_SIDE: "no"` every price the agent sees is `100 - p` and the two
sides swap, so the brief above renders as its mirror. The same three rounds:

```text
EXCHANGE HISTORY FOR THIS MARKET
  round 1: cleared at 53c, volume x18, unfilled 0
  round 2: no trade, best bid 45c x9, best ask 60c x8
  round 3: cleared at 53c, volume x4, unfilled ask x5

YOUR HISTORY IN THIS MARKET
  round 1: 45c x10 at 52c x10, filled -10 at 53c

YOUR ACCOUNT
  Position in this market: -10 contracts, average cost 53c
```

Storage stays in YES space throughout; the inversion happens only at this
boundary (`agents._to_frame`, `_qty_to_frame`).

---

# Part B — Terminal (stdout)

Printed by `runner.run_round`, in this order, once per round. Rules are 78
columns (`display.WIDTH`).

A market block is a **price view only**: the quotes posted, where the book
stood, and what it cleared at. Per-agent fills, the tie interval and the
indicative microprice are all in the database and deliberately not printed.

## B1. Round header

```text

──────────────────────────────────────────────────────────────────────────────
  Round 4                                                  2026-09-04 09:20:02
──────────────────────────────────────────────────────────────────────────────
```

## B2. Live markets

```text

  Live markets (3)
    Politics    KXPRESVOTE-26-DEM              Will the Democrats win the 2026 House?         1d16h
    Crypto      KXBTCMAXY-26-120000            Bitcoin above $120,000 on Dec 31?             14h00m
    Economics   KXCPIYOY-26MAR-3.0             CPI year-over-year above 3.0% in March?        1d06h
```

Columns: sector (11, truncated), ticker (30, truncated), title (44, truncated),
close-in (7, right). A market row missing from `markets` degrades to the bare
ticker:

```text
    KXPRESVOTE-26-DEM
```

Real rows run to ~107 columns — past `WIDTH`, and the widest thing the terminal
prints.

## B3. Settlements

Emitted only when something settled or voided; otherwise `display.settlements`
returns `""` and an empty line is printed.

```text

  Settlements (3)
    haiku-1    KXBTCMAXY-26-120000                qty  +12  payout   1,200c
    sonnet-1   KXBTCMAXY-26-120000                qty  -12  payout  -1,200c
    opus-1     KXCPIYOY-26MAR-3.0                 qty   +5  payout     235c  void
```

The trailing `void` marks a ghost market closed at the last internal clear
rather than a real Kalshi payout.

## B4. Market round — cleared

One block per market. Three aligned rows: `Quotes`, `Book`, `Clear`. Quote cells
run two per row.

```text

  KXPRESVOTE-26-DEM
  Will the Democrats win the 2026 House?

    Quotes    haiku-1   48c x10 / 55c x10    haiku-2   47c x8  / 60c x5
              opus-1    41c x6  / 47c x6     sonnet-1  40c x6  / 44c x12
              sonnet-2  45c x8  / 52c x8
              opus-2      rejected — KXPRESVOTE-26-DEM: a full bid fill would take the position to 215, past the +/-200 cap
    Book      best bid 48c x10   best ask 44c x12
    Clear     47c   volume x18   unfilled 0
```

- A quote cell is `agent bid x<size> / ask x<size>` — both sizes, unlike the old
  format.
- `Book` is the same top-of-book the agents see, from `auction.top_of_book`; in
  a crossed round the best bid sits **above** the best ask, which is what made
  the print possible.
- `Clear` carries the price, the executed volume and the imbalance, in the same
  wording as the brief.
- The `rejected` line is unwrapped and routinely exceeds 78 columns.

## B5. Market round — a crossed book with an imbalance

```text
    Quotes    haiku-1   46c x6  / 52c x6     opus-1    44c x4  / 45c x4
              sonnet-1  48c x9  / 51c x9
    Book      best bid 48c x9   best ask 45c x4
    Clear     47c   volume x4   unfilled bid x5
```

## B6. Market round — no trade

```text

  KXPRESVOTE-26-DEM
  Will the Democrats win the 2026 House?

    Quotes    haiku-1   30c x4  / 70c x6     haiku-2   40c x5  / 55c x7
              opus-1    35c x9  / 60c x9     sonnet-1  40c x3  / 55c x2
    Book      best bid 40c x8   best ask 55c x9
    Clear     no trade
```

## B7. Market round — no quotes at all

`Book` is omitted entirely when there is nothing to report.

```text

  KXPRESVOTE-26-DEM
  Will the Democrats win the 2026 House?

    Quotes    (none)
    Clear     no trade
```

## B8. Standings

Once per round, after every market. Sorted by equity, descending.

```text

──────────────────────────────────────────────────────────────────────────────
  Standings after round 3
──────────────────────────────────────────────────────────────────────────────
  agent            cash   open   realised    unreal      equity
  sonnet-3      100,000      0         +0        +0     100,000
  sonnet-2      100,000      0         +0        +0     100,000
  sonnet-1      100,376      1         +0        +0     100,000
  opus-3        100,000      0         +0        +0     100,000
  opus-2        100,000      0         +0        +0     100,000
  opus-1        100,470      1         +0        +0     100,000
  haiku-3       100,000      0         +0        +0     100,000
  haiku-2        99,624      1         +0        +0     100,000
  haiku-1        99,530      1         +0        +0     100,000
```

---

# Regenerating these examples

Nothing here needs an LLM call or Kalshi access. Seed `markets`, `quotes`,
`auctions` and `fills` for one ticker across three rounds, then call
`agents.build_prompt`, `agents.render_brief` and each `display.*` function
directly.
