"""Compatibility entry point; implementation lives in prediction_markets.integrations.kalshi."""

import sys
from prediction_markets.integrations import kalshi as _implementation

sys.modules[__name__] = _implementation
