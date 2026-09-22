"""Compatibility entry point; implementation lives in prediction_markets.markets.selection."""

import sys
from prediction_markets.markets import selection as _implementation

sys.modules[__name__] = _implementation
