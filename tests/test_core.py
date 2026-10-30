import io
import numpy as np
import pandas as pd

from mobility import dataio, geo, insights as I, quality, report, samples, trips
from mobility.detection import detect_columns
from mobility.utils import haversine_m, parse_datetime


def test_samples_are_reproducible():
    a = samples.generate_sample_traffic(seed=1, days=3)
    b = samples.generate_sample_traffic(seed=1, days=3)
    pd.testing.assert_frame_equal(a, b)


def test_detection_traffic(traffic_map):
    for f in ("timestamp", "speed", "volume", "latitude", "longitude", "road_id"):
        assert traffic_map.get(f), f


def test_detection_gps(gps_map):
    assert gps_map["vehicle_id"] == "vehicle_id"


def test_detection_taxi_style():
    df = pd.DataFrame({
        "pickup_datetime": ["2024-01-01 10:00:00"], "dropoff_datetime": ["2024-01-01 10:20:00"],
        "pickup_latitude": [40.7], "pickup_longitude": [-74.0],
        "dropoff_latitude": [40.8], "dropoff_longitude": [-73.9]})
    m = detect_columns(df)
    assert m["origin_lat"] == "pickup_latitude" and m["dest_lat"] == "dropoff_latitude"


def test_parse_datetime_epoch_and_strings():
    assert parse_datetime(pd.Series([1700000000, 1700000060])).notna().all()
    assert parse_datetime(pd.Series(["2024-01-01 10:00", "bad"])).notna().sum() == 1


def test_haversine_known_distance():
    d = haversine_m(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
    assert abs(float(d[0]) - 111_195) < 300


def test_quality_finds_dirt(traffic, traffic_map):
    f = quality.profile_data(traffic, traffic_map)
    assert 0 <= f["quality_score"] < 100
    assert f["impossible_speed"] > 0


def test_cleaning_reduces_problems(traffic, traffic_map):
    before = quality.profile_data(traffic, traffic_map)
    cleaned, _ = quality.run_smart_cleaning(
        traffic, {k: True for k in ("duplicates", "coordinates", "impossible", "missing")}, traffic_map)[:2]
    after = quality.profile_data(cleaned, traffic_map)
    assert after["quality_score"] >= before["quality_score"]
    assert len(cleaned) <= len(traffic)


def test_panel_and_findings(traffic, traffic_map):
    panel = I.add_congestion(I.prepare_panel(traffic, traffic_map))
    assert panel["ci"].dropna().between(0, 1).all()
    assert len(I.generate_findings(panel, (5, 6))) >= 3


def test_incident_detection_finds_injected(traffic, traffic_map):
    panel = I.add_congestion(I.prepare_panel(traffic, traffic_map))
    flagged, _ = I.detect_anomalies(panel, "speed", "robust_z", 3.5, 0.01, "low")
    inc = I.group_incidents(flagged, I.infer_interval_minutes(panel), min_points=3)
    assert len(inc) >= 1


def test_trips_pipeline(gps, gps_map):
    out = trips.analyse_trajectories(gps, gps_map)
    t = out["trips"] if isinstance(out, dict) else out[0]
    assert len(t) > 50


def test_geo_grid_and_blend(traffic, traffic_map):
    cells = geo.aggregate_grid(traffic.dropna(subset=["latitude", "longitude"]),
                               traffic_map["latitude"], traffic_map["longitude"], 1000)
    assert len(cells) >= 1 and cells["count"].sum() > 0
    assert geo.blend(0, 0, 1).startswith("#")


def test_point_in_polygon():
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"n": "A"},
          "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}}]}
    parsed = geo.parse_geojson(gj)
    idx = geo.assign_points_to_polygons(np.array([0.5, 5.0]), np.array([0.5, 5.0]), parsed)
    assert idx[0] == 0 and idx[1] == -1


def test_dataio_roundtrip(traffic):
    data = traffic.head(100).to_csv(index=False).encode()
    assert len(dataio.read_dataset(data, "x.csv")) == 100


def test_report_escapes_html():
    html = report.build_report_html("<b>T</b>", "", [{"title": "S", "paragraphs": ["<script>x</script>"]}])
    assert "<script>x</script>" not in html


def test_nullable_and_arrow_style_dtypes(traffic, traffic_map, gps, gps_map):
    """Nullable dtypes (what Arrow/Parquet/DuckDB data looks like) must not break insights or trips."""
    t = traffic.copy()
    t["segment_id"] = t["segment_id"].astype("string")
    t["speed_kmh"] = t["speed_kmh"].astype("Float64")
    panel = I.add_congestion(I.prepare_panel(t, traffic_map))
    flagged, _ = I.detect_anomalies(panel, "speed", "robust_z", 3.5, 0.01, "low")
    I.group_incidents(flagged, 15, min_points=2)

    g = gps.copy()
    g["vehicle_id"] = g["vehicle_id"].astype("string")
    g["speed_kmh"] = g["speed_kmh"].astype("Float64")
    out = trips.analyse_trajectories(g, gps_map)
    assert len(out["trips"] if isinstance(out, dict) else out[0]) > 50


def test_normalize_dtypes_keeps_plain_frames():
    from mobility.dataio import normalize_dtypes
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    pd.testing.assert_frame_equal(normalize_dtypes(df), df)
