"""Compatibility entry point; implementation lives in prediction_markets.markets.update."""

import sys
from prediction_markets.markets import update as _implementation

sys.modules[__name__] = _implementation
