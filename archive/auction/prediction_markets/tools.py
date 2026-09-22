"""Agent-facing tools. v1 headers only.

These exist so the call path has a place to grow into, not because v0 uses
them: nothing here is mentioned in the v0 prompt and nothing here is reachable
from the v0 call path. An agent's entire v0 output is its quote.

Any tool added here must obey the core invariant — no Kalshi price data may
reach an agent-visible surface, so no tool may import kalshi_client or read
the price columns of `snapshots`.
"""


def web_search(query: str):
    raise NotImplementedError("web_search is a v1 tool; not available in v0")


def get_news(ticker: str):
    raise NotImplementedError("get_news is a v1 tool; not available in v0")
