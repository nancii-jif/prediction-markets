"""Compatibility entry point; implementation lives in prediction_markets.integrations.model_client."""

import sys
from prediction_markets.integrations import model_client as _implementation

sys.modules[__name__] = _implementation
