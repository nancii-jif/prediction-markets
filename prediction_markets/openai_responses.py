"""Compatibility entry point; implementation lives in prediction_markets.integrations.openai_responses."""

import sys
from prediction_markets.integrations import openai_responses as _implementation

sys.modules[__name__] = _implementation
