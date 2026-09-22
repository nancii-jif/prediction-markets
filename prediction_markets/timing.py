"""Compatibility entry point; implementation lives in prediction_markets.runtime.schedule."""

import sys
from prediction_markets.runtime import schedule as _implementation

sys.modules[__name__] = _implementation
