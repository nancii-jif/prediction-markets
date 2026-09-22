"""Compatibility entry point; implementation lives in prediction_markets.analysis.trace."""

import sys
from prediction_markets.analysis import trace as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
