"""Compatibility entry point; implementation lives in prediction_markets.analysis.prices."""

import sys
from prediction_markets.analysis import prices as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
