"""Compatibility entry point; implementation lives in prediction_markets.storage.database."""

import sys
from prediction_markets.storage import database as _implementation

sys.modules[__name__] = _implementation
