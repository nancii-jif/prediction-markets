"""Compatibility entry point; implementation lives in prediction_markets.exchange.bookkeeping."""

import sys
from prediction_markets.exchange import bookkeeping as _implementation

sys.modules[__name__] = _implementation
