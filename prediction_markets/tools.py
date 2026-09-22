"""Compatibility entry point; implementation lives in prediction_markets.agents.tools."""

import sys
from prediction_markets.agents import tools as _implementation

sys.modules[__name__] = _implementation
