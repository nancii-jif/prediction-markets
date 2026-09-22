"""Compatibility entry point; implementation lives in prediction_markets.analysis.export."""

import sys
from prediction_markets.analysis import export as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
