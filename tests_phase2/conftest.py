from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def reset_tracker_ids():
    """Keep tracker identifiers deterministic across Phase 2 test cases."""
    from sort import KalmanBoxTracker

    KalmanBoxTracker.count = 0
    yield
    KalmanBoxTracker.count = 0
