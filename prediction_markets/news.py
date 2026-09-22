"""Compatibility entry point; implementation lives in prediction_markets.integrations.tavily."""

import sys
from prediction_markets.integrations import tavily as _implementation

sys.modules[__name__] = _implementation
