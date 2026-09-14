"""Load the test run config before anything else.

Config values are read at collection time — parametrize decorators reference
them — so this has to happen at conftest import, before test modules are
imported. pytest guarantees that ordering.
"""

import tempfile
from pathlib import Path

from prediction_markets import config

config.load(config.CONFIG_DIR / "test.yaml", name="test")

# Keep the suite out of runs/: any code path that falls back to config.DB_PATH
# should land in a temporary directory, not in the run artifacts tree.
_TMP = Path(tempfile.mkdtemp(prefix="pm-tests-"))
config._VALUES["RUN_DIR"] = _TMP
config._VALUES["DATA_DIR"] = _TMP
config._VALUES["DB_PATH"] = _TMP / "run.db"
