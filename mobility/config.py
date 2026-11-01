"""Constants, field aliases and units.

Field detection rules are *config-driven*: the defaults below can be extended
without touching code by editing ``config/field_aliases.json`` (values are
merged with the defaults, never replace them).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALIASES_FILE = ROOT / "config" / "field_aliases.json"

APP_NAME = "Smart Mobility Analytics Studio"

# ---- limits -------------------------------------------------------------
MAX_HISTORY = 10            # undo snapshots kept
MAX_HISTORY_MB = 400        # approximate memory cap for undo snapshots
MAX_PLOT_POINTS = 50_000    # points sent to the browser
MAX_TRAIN_ROWS = 300_000    # rows used to fit a model
MAX_EXCEL_ROWS = 1_048_575
DUCKDB_THRESHOLD_MB = 50    # CSVs larger than this use DuckDB when installed

# ---- units --------------------------------------------------------------
# factor that converts the unit to km/h
SPEED_UNITS = {"km/h": 1.0, "mph": 1.609344, "m/s": 3.6, "knots": 1.852}

WEEKEND_PRESETS = {
    "Saturday–Sunday": (5, 6),
    "Friday–Saturday": (4, 5),
    "Sunday only": (6,),
}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ---- congestion levels (congestion index = 1 - speed / free-flow speed) ----
CONGESTION_LEVELS = [
    ("Free flow", 0.0, 0.15),
    ("Light", 0.15, 0.30),
    ("Moderate", 0.30, 0.50),
    ("Heavy", 0.50, 1.01),
]
HEAVY_THRESHOLD = 0.50

# ---- fields -------------------------------------------------------------
FIELD_LABELS = {
    "timestamp": "Timestamp / date",
    "vehicle_id": "Vehicle / device ID",
    "origin_lat": "Trip origin latitude",
    "origin_lon": "Trip origin longitude",
    "dest_lat": "Trip destination latitude",
    "dest_lon": "Trip destination longitude",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "speed": "Speed",
    "volume": "Traffic volume / count",
    "road_id": "Road / segment ID",
}

# Order matters: more specific fields first, so that e.g. ``pickup_latitude``
# is claimed by origin_lat before the generic latitude rule sees it.
DEFAULT_FIELD_RULES = {
    "timestamp": {
        "exact": ["timestamp", "datetime", "dateandtime", "eventtime",
                  "recordedat", "measurementtime", "datetimeutc",
                  "timestamputc", "date", "time"],
        "contains": ["timestamp", "datetime"],
        "tokens": ["date", "time"],
        "exclude": [],
    },
    "vehicle_id": {
        "exact": ["vehicleid", "vehicle", "deviceid", "device", "trackid",
                  "taxiid", "driverid", "carid", "objectid", "uid"],
        "contains": ["vehicleid", "deviceid", "trackid", "taxiid", "carid"],
        "tokens": [],
        "exclude": [],
    },
    "origin_lat": {
        "exact": ["originlat", "originlatitude", "startlat", "startlatitude",
                  "fromlat", "pickuplat", "pickuplatitude"],
        "contains": ["originlat", "startlat", "pickuplat", "fromlat"],
        "tokens": [],
        "exclude": [],
    },
    "origin_lon": {
        "exact": ["originlon", "originlng", "originlongitude", "startlon",
                  "startlng", "startlongitude", "fromlon", "pickuplon",
                  "pickuplongitude"],
        "contains": ["originlon", "originlng", "startlon", "startlng",
                     "pickuplon", "pickuplng", "fromlon"],
        "tokens": [],
        "exclude": [],
    },
    "dest_lat": {
        "exact": ["destlat", "destlatitude", "destinationlat",
                  "destinationlatitude", "endlat", "endlatitude", "tolat",
                  "dropofflat", "dropofflatitude"],
        "contains": ["destlat", "destinationlat", "endlat", "dropofflat"],
        "tokens": [],
        "exclude": [],
    },
    "dest_lon": {
        "exact": ["destlon", "destlng", "destlongitude", "destinationlon",
                  "destinationlongitude", "endlon", "endlng", "endlongitude",
                  "tolon", "dropofflon", "dropofflongitude"],
        "contains": ["destlon", "destlng", "destinationlon", "endlon",
                     "endlng", "dropofflon", "dropofflng"],
        "tokens": [],
        "exclude": [],
    },
    "latitude": {
        "exact": ["latitude", "lat", "gpslatitude", "ycoordinate", "coordy",
                  "pointlatitude", "ycoord"],
        "contains": ["latitude"],
        "tokens": ["lat", "latitude"],
        "exclude": [],
    },
    "longitude": {
        "exact": ["longitude", "lon", "lng", "long", "gpslongitude",
                  "xcoordinate", "coordx", "pointlongitude", "xcoord"],
        "contains": ["longitude"],
        "tokens": ["lon", "lng", "long", "longitude"],
        "exclude": [],
    },
    "speed": {
        "exact": ["speed", "speedkmh", "speedkph", "speedmph", "vehiclespeed",
                  "avgspeed", "averagespeed", "velocity"],
        "contains": ["speed", "velocity"],
        "tokens": ["speed", "velocity"],
        "exclude": ["limit"],
    },
    "volume": {
        "exact": ["trafficvolume", "volume", "flow", "trafficflow",
                  "vehiclecount", "vehicles", "vehiclevolume", "flowrate",
                  "countvehicles", "trafficcount"],
        "contains": ["trafficvolume", "vehiclecount", "trafficcount",
                     "trafficflow", "vehiclevolume", "flowrate"],
        "tokens": ["volume", "vehicles", "flow"],
        "exclude": [],
    },
    "road_id": {
        "exact": ["roadid", "roadidentifier", "roadcode", "segmentid",
                  "linkid", "streetid", "highwayid", "roadname",
                  "streetname", "segment", "road", "link", "sensorid",
                  "detectorid", "stationid"],
        "contains": ["roadid", "segmentid", "linkid", "roadname",
                     "streetname", "streetid", "sensorid", "detectorid"],
        "tokens": [],
        "exclude": [],
    },
}

ID_TOKENS = {"id", "uid", "uuid", "idx", "index", "key", "code"}


def load_field_rules(path=ALIASES_FILE):
    """Defaults merged with the optional JSON file (union of the lists)."""
    rules = {
        field: {k: list(v) for k, v in rule.items()}
        for field, rule in DEFAULT_FIELD_RULES.items()
    }

    try:
        extra = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return rules

    for field, rule in extra.items():
        if field.startswith("_") or field not in rules or not isinstance(rule, dict):
            continue
        for key in ("exact", "contains", "tokens", "exclude"):
            for value in rule.get(key, []) or []:
                value = str(value).lower()
                if value not in rules[field][key]:
                    rules[field][key].append(value)
    return rules
