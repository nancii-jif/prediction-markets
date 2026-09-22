"""Compatibility entry point; implementation lives in prediction_markets.integrations.tokenization."""

import sys
from prediction_markets.integrations import tokenization as _implementation

sys.modules[__name__] = _implementation
