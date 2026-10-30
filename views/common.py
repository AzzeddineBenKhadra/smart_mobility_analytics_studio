"""Cached accessors shared by several pages."""

from mobility.insights import add_congestion, prepare_panel
from mobility.state import cached


def mapping_key(m, *extra):
    keys = ("timestamp", "speed", "volume", "road_id", "latitude", "longitude",
            "speed_unit") + extra
    return tuple(m.get(k) for k in keys)


def get_panel(df, m, free_flow_q=0.85):
    """Tidy panel (+ congestion index when speed is mapped). None if unusable."""
    key = mapping_key(m)
    base = cached("panel_base", lambda: prepare_panel(df, m), key)
    if base is None:
        return None
    return cached("panel_ci", lambda: add_congestion(base, free_flow_q), key, free_flow_q)


def missing_fields_message(m, required):
    names = {
        "timestamp": "timestamp", "speed": "speed", "volume": "traffic volume",
        "latitude": "latitude", "longitude": "longitude", "vehicle_id": "vehicle ID",
        "road_id": "road / segment ID",
    }
    missing = [names[r] for r in required if not m.get(r)]
    return ", ".join(missing)
