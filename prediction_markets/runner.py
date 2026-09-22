"""Compatibility entry point; implementation lives in prediction_markets.runtime.runner."""

import sys
from prediction_markets.runtime import runner as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
