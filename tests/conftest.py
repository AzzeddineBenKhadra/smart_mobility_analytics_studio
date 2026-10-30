import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobility.detection import detect_columns  # noqa: E402
from mobility.samples import generate_sample_trajectories, generate_sample_traffic  # noqa: E402


@pytest.fixture(scope="session")
def traffic():
    return generate_sample_traffic()


@pytest.fixture(scope="session")
def gps():
    return generate_sample_trajectories()


@pytest.fixture(scope="session")
def traffic_map(traffic):
    m = detect_columns(traffic)
    m["speed_unit"] = "km/h"
    return m


@pytest.fixture(scope="session")
def gps_map(gps):
    m = detect_columns(gps)
    m["speed_unit"] = "km/h"
    return m
