"""One structured LLM call per (agent, market) per round. No tool loop.

The quote is the entire agent output: there is no separate belief elicitation.
Agents see market metadata, the internal exchange's own history, and their own
ledger state — never a Kalshi price, directly or derived.

The quote frame lives in exactly two places, both in this file: `render_brief`
presents the contract, and `parse_quote` maps the response back. Everything
downstream of the parser is in YES space.
"""

import asyncio
import json
import logging
import re
import time

from . import auction, config, database, ledger

log = logging.getLogger(__name__)

# Static prefix. Kept first and byte-stable across every call so prompt
# caching has something to hold onto — at this call volume the cache is a
# first-order cost lever, not an optimisation.
PROTOCOL = """You are a market maker on an internal prediction-market exchange.

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
{contract_definition}

YOUR TASK
- Every round, you must quote both sides on the market described below. You should bid lower than your ask.
- Your quotes should reflect your view and confidence in the probability of the event happening.
    - If you have no view or are uncertain, quote a wide spread.
- You should rely on your knowledge, rules of this market, and the trading history on the exchange, as well as your position history.

Respond with ONLY a JSON object, no other text:
{{"bid_cents": <int>, "bid_size": <int>, "ask_cents": <int>, "ask_size": <int>}}

Constraints: 1 <= bid_cents < ask_cents <= 99, and each size is
a whole number of contracts from 1 to {max_lot}."""

YES_CONTRACT = """You are quoting the YES contract. It pays 100 cents if the
market's stated condition is met, and 0 cents otherwise. Your bid is what you
will pay to buy YES; your ask is what you will accept to sell YES."""

NO_CONTRACT = """You are quoting the NO contract. It pays 100 cents if the
market's stated condition is NOT met, and 0 cents if it is met. Your bid is
what you will pay to buy NO; your ask is what you will accept to sell NO."""


# Models that answered a temperature-bearing request with a permanent error.
# Discovered at runtime rather than pinned in a list, so a model that gains or
# loses the parameter needs no code change: the Claude 5 family rejects it
# ("`temperature` is deprecated for this model") while Haiku 4.5 accepts it.
# One wasted request per model per process, then cached.
REJECTS_TEMPERATURE: set[str] = set()


class QuoteError(ValueError):
    """The model's response could not be turned into a valid quote."""


class Agent:
    """One participant on the exchange.

    Model, prompt and tools are injected rather than hard-coded so a roster
    can mix families and, later, personas. In v0 every agent shares one prompt
    and no agent has tools.
    """

    def __init__(self, agent_id, model, prompt=None, tools=None,
                 temperature=None, persona=None):
        self.agent_id = agent_id
        self.model = model
        self.persona = persona
        self.tools = list(tools or [])
        self.temperature = temperature
        self.prompt = prompt if prompt is not None else build_prompt(persona)

    def __repr__(self):
        return f"Agent({self.agent_id!r}, model={self.model!r})"

    async def quote(self, ticker: str, round_id: int) -> dict:
        """Produce one validated quote, in YES space, for one market.

        On a parse or validation failure the model gets exactly one more try
        with the error appended. A second failure is recorded as parse_ok = 0
        and the agent simply has no quote in this market this round.
        """
        brief = render_brief(self.agent_id, ticker, round_id)
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": brief},
        ]

        raw = None
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            try:
                raw = await self._call(messages)
                quote = parse_quote(raw, config.QUOTE_SIDE)
            except QuoteError as exc:
                if attempt < config.LLM_MAX_RETRIES:
                    messages += [
                        {"role": "assistant", "content": raw or ""},
                        {"role": "user", "content":
                         f"That response was rejected: {exc}. Reply with only "
                         f"the JSON object."},
                    ]
                    continue
                log.warning("%s %s: unusable response (%s)", self.agent_id, ticker, exc)
                database.record_call(
                    self.agent_id, ticker, round_id, int(time.time()), brief, raw, 0
                )
                return None
            else:
                database.record_call(
                    self.agent_id, ticker, round_id, int(time.time()), brief, raw, 1
                )
                quote["ticker"] = ticker
                quote["agent"] = self.agent_id
                return quote

    async def _call(self, messages) -> str:
        import litellm

        base = dict(
            model=self.model,
            messages=messages,
            timeout=config.LLM_TIMEOUT_S,
        )
        send_temperature = (
            self.temperature is not None and self.model not in REJECTS_TEMPERATURE
        )

        if send_temperature:
            try:
                # num_retries=0 on this attempt only. If the model rejects
                # temperature the error is permanent, and retrying it would
                # burn LLM_API_RETRIES requests before teaching us anything.
                response = await litellm.acompletion(
                    **base, temperature=self.temperature, num_retries=0
                )
                return response.choices[0].message.content
            except Exception as exc:
                if "temperature" in str(exc).lower():
                    REJECTS_TEMPERATURE.add(self.model)
                    log.warning(
                        "%s rejected temperature=%s (%s); omitting the parameter "
                        "for this model from here on",
                        self.model, self.temperature,
                        str(exc).split("\n")[0][:120],
                    )
                    send_temperature = False
                # Anything else is transient — fall through and retry properly,
                # still sending temperature.

        optional = {"temperature": self.temperature} if send_temperature else {}
        response = await litellm.acompletion(
            **base, **optional,
            # Transport-level retries with backoff. Distinct from
            # LLM_MAX_RETRIES, which reprompts a model that returned
            # unparseable output: a 429 or a 5xx is not the model's fault and
            # must not cost the agent its quote for the round.
            num_retries=config.LLM_API_RETRIES,
        )
        return response.choices[0].message.content


def build_prompt(persona: str | None = None) -> str:
    """The static system prompt. Versioned by config.PROMPT_VERSION."""
    definition = YES_CONTRACT if config.QUOTE_SIDE == "yes" else NO_CONTRACT
    prompt = PROTOCOL.format(
        contract_definition=definition, max_lot=config.MAX_LOT
    )
    return f"{persona}\n\n{prompt}" if persona else prompt


def build_roster() -> list[Agent]:
    """Instantiate the configured roster."""
    return [
        Agent(
            agent_id=spec["agent_id"],
            model=spec["model"],
            persona=spec.get("persona"),
            temperature=spec.get("temperature", 0.0),
        )
        for spec in config.AGENTS
    ]


# --- the frame boundary ----------------------------------------------------


def _to_frame(price_cents: int | None) -> int | None:
    """A YES-space price as the agent sees it under the run's frame."""
    if price_cents is None:
        return None
    return price_cents if config.QUOTE_SIDE == "yes" else 100 - price_cents


def _qty_to_frame(signed_qty: int) -> int:
    """A YES-space position as the agent sees it. Long NO is short YES."""
    return signed_qty if config.QUOTE_SIDE == "yes" else -signed_qty


def parse_quote(text: str, quote_side: str = "yes") -> dict:
    """Turn a model response into a validated quote in YES space.

    Under the NO frame the mapping is done here and nowhere else: buying NO at
    p is selling YES at 100-p, so the two sides swap as well as invert.
    """
    if not text:
        raise QuoteError("empty response")

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise QuoteError("no JSON object found in the response")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise QuoteError(f"response was not valid JSON ({exc})") from exc

    fields = ("bid_cents", "bid_size", "ask_cents", "ask_size")
    values = {}
    for field in fields:
        if field not in payload:
            raise QuoteError(f"missing required field {field}")
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise QuoteError(f"{field} must be a whole number, got {value!r}")
        if float(value) != int(value):
            raise QuoteError(f"{field} must be a whole number, got {value!r}")
        values[field] = int(value)

    bid, ask = values["bid_cents"], values["ask_cents"]
    if not (config.PRICE_MIN <= bid < ask <= config.PRICE_MAX):
        raise QuoteError(
            f"need {config.PRICE_MIN} <= bid_cents < ask_cents <= "
            f"{config.PRICE_MAX}, got bid={bid} ask={ask}"
        )
    for field in ("bid_size", "ask_size"):
        if not (1 <= values[field] <= config.MAX_LOT):
            raise QuoteError(
                f"{field} must be 1..{config.MAX_LOT}, got {values[field]}"
            )

    if quote_side == "yes":
        return values
    return {
        "bid_cents": 100 - values["ask_cents"],
        "bid_size": values["ask_size"],
        "ask_cents": 100 - values["bid_cents"],
        "ask_size": values["bid_size"],
    }


# --- the brief -------------------------------------------------------------


def _format_duration(seconds: int) -> str:
    if seconds <= 0:
        return "closed"
    hours, minutes = divmod(int(seconds) // 60, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h"
    return f"{hours}h {minutes}m"


def _format_book(quotes) -> str:
    """The top of book of a round that did not print, in the agent's frame.

    Reported only where there is no clearing price: the best bid and ask are
    then the only thing the round revealed, and without them a run of no-trade
    rounds tells an agent nothing about how far apart the quotes were.
    `auction.top_of_book` has already summed size per level, so no line here is
    attributable to an individual agent.
    """
    top = auction.top_of_book(quotes)
    if top is None:
        return ""

    bid, bid_size = _to_frame(top["bid_cents"]), top["bid_size"]
    ask, ask_size = _to_frame(top["ask_cents"]), top["ask_size"]
    if config.QUOTE_SIDE == "no":
        # The best YES bid is the best offer to sell NO, and vice versa: the
        # sides swap along with their sizes, exactly as in own-quote history.
        bid, ask = ask, bid
        bid_size, ask_size = ask_size, bid_size

    parts = [
        f"best bid {bid}c x{bid_size}" if bid is not None else "no bid",
        f"best ask {ask}c x{ask_size}" if ask is not None else "no ask",
    ]
    return ", " + ", ".join(parts)


def _format_unfilled(quotes, clear_cents: int) -> str:
    """The size that could not trade at the clearing price, in the agent's frame.

    The starved side is already partly known to whoever was rationed by the
    pro-rata allocation, and to nobody else; reporting it makes that common
    knowledge instead of a windfall for the agents who happened to be marginal.
    """
    excess = auction.imbalance(quotes, clear_cents)
    if excess == 0:
        return "unfilled 0"
    # Excess demand for YES is excess supply of NO: the side flips with the frame.
    side = "bid" if (excess > 0) == (config.QUOTE_SIDE == "yes") else "ask"
    return f"unfilled {side} x{abs(excess)}"


def render_brief(agent_id: str, ticker: str, round_id: int, now: int | None = None) -> str:
    """The volatile suffix: everything this agent has inherited about this
    market, and nothing computed beyond it.

    Every value here comes from the `markets` metadata, the internal
    exchange's own tables, or this agent's own ledger. No Kalshi price, no
    derived spread or mid, ever appears — see the isolation test.
    """
    now = now or int(time.time())
    market = database.market(ticker)
    lines = []

    lines.append(f"MARKET: {market['title']}")
    lines.append(f"Ticker: {ticker}")
    lines.append("")
    lines.append("RESOLUTION RULES")
    lines.append(market["rules"] or "(not provided)")
    lines.append("")
    if market["close_ts"]:
        lines.append(
            f"Closes in {_format_duration(market['close_ts'] - now)} "
            f"(round {round_id})"
        )
    else:
        lines.append(f"Close time unknown (round {round_id})")
    lines.append("")

    # Public history: what the exchange itself has printed in this market.
    auctions = database.auctions_for(ticker)
    lines.append("EXCHANGE HISTORY FOR THIS MARKET")
    if not auctions:
        lines.append("No rounds have been held yet. This is the first auction.")
    else:
        books = database.accepted_quotes_by_round(ticker)
        for row in auctions:
            book = books.get(row["round_id"], [])
            if row["clear_cents"] is None:
                lines.append(
                    f"  round {row['round_id']}: no trade{_format_book(book)}"
                )
            else:
                lines.append(
                    f"  round {row['round_id']}: cleared at "
                    f"{_to_frame(row['clear_cents'])}c, volume x{row['cleared_qty']}, "
                    f"{_format_unfilled(book, row['clear_cents'])}"
                )
    lines.append("")

    # Private history: this agent's own quotes and fills. Other agents'
    # individual quotes are never shown.
    quotes = database.quotes_for(agent_id, ticker)
    fills = {f["round_id"]: f for f in database.fills_for(agent_id, ticker)}
    lines.append("YOUR HISTORY IN THIS MARKET")
    if not quotes:
        lines.append("You have not quoted this market before.")
    else:
        for row in quotes:
            if row["status"] != "accepted":
                lines.append(
                    f"  round {row['round_id']}: quote rejected "
                    f"({row['reject_reason']})"
                )
                continue
            bid, ask = _to_frame(row["bid_cents"]), _to_frame(row["ask_cents"])
            bid_size, ask_size = row["bid_size"], row["ask_size"]
            if config.QUOTE_SIDE == "no":
                bid, ask = ask, bid
                bid_size, ask_size = ask_size, bid_size
            line = (f"  round {row['round_id']}: {bid}c x{bid_size} at "
                    f"{ask}c x{ask_size}")
            fill = fills.get(row["round_id"])
            if fill:
                # Signed, not "bought"/"sold": the sign is what the position
                # line is keyed to, and it survives the frame swap unambiguously.
                qty = _qty_to_frame(fill["signed_qty"])
                line += f", filled {qty:+d} at {_to_frame(fill['price_cents'])}c"
            else:
                line += ", no fill"
            lines.append(line)
    lines.append("")

    # Ledger state.
    signed_qty, basis = database.position(agent_id, ticker)
    position = _qty_to_frame(signed_qty)
    account = ledger.pnl(agent_id)
    lines.append("YOUR ACCOUNT")
    if position:
        average = abs(round(basis / signed_qty)) if signed_qty else 0
        average = _to_frame(average) if config.QUOTE_SIDE == "yes" else 100 - average
        lines.append(
            f"  Position in this market: {position:+d} contracts, "
            f"average cost {average}c"
        )
    else:
        lines.append("  Position in this market: flat")
    lines.append(f"  Cash: {account['cash_cents']}c")
    lines.append(f"  Realised PnL: {account['realized_cents']:+d}c")
    lines.append(f"  Unrealised PnL: {account['unrealized_cents']:+d}c")
    cap = (
        "no position limit" if config.MAX_POSITION is None
        else f"Position limit: +/-{config.MAX_POSITION} contracts"
    )
    lines.append(f"  {cap}; max {config.MAX_LOT} contracts per side per round")
    lines.append("")
    lines.append("Post your quote now, as JSON only.")

    return "\n".join(lines)


async def check_families() -> int:
    """One real call per model family, against a synthetic market.

        python -m prediction_markets.agents --check

    Confirms each family is reachable and returns a quote that survives
    parsing and validation. Needs a provider key in the environment; it is the
    one check in this project that cannot be run offline.
    """
    brief = (
        "MARKET: Will a fair coin land heads?\n\n"
        "RESOLUTION RULES\nResolves YES if the coin lands heads.\n\n"
        "Closes in 24h 0m (round 1)\n\n"
        "EXCHANGE HISTORY FOR THIS MARKET\n"
        "No rounds have been held yet. This is the first auction.\n\n"
        "YOUR HISTORY IN THIS MARKET\nYou have not quoted this market before.\n\n"
        "YOUR ACCOUNT\n  Position in this market: flat\n"
        f"  Cash: {config.INITIAL_CASH_CENTS}c\n\n"
        "Post your quote now, as JSON only."
    )

    failures = 0
    for family, model in config.FAMILIES:
        agent = Agent(agent_id=f"check-{family}", model=model)
        try:
            raw = await agent._call([
                {"role": "system", "content": agent.prompt},
                {"role": "user", "content": brief},
            ])
            quote = parse_quote(raw, config.QUOTE_SIDE)
        except Exception as exc:
            failures += 1
            print(f"FAIL  {family:8s} {model}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {family:8s} {model}  ->  {quote}")
    return failures


async def gather_quotes(agents, tickers, round_id) -> list[dict]:
    """Every agent quotes every market, concurrently.

    A per-call failure is logged and dropped; the round continues without that
    agent's quote in that market.
    """
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENCY)

    async def one(agent, ticker):
        async with semaphore:
            try:
                return await agent.quote(ticker, round_id)
            except Exception:
                log.exception("call failed: %s on %s", agent.agent_id, ticker)
                return None

    tasks = [one(agent, ticker) for agent in agents for ticker in tickers]
    return [q for q in await asyncio.gather(*tasks) if q]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="make one real call per model family and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.check:
        raise SystemExit(asyncio.run(check_families()))
    parser.print_help()
