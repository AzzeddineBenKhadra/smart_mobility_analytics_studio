"""Trajectory analytics: motion, trips, stops and origin–destination flows.

Method (deliberately simple and explainable)
--------------------------------------------
1. Points are sorted per vehicle; distance/speed between consecutive points is
   computed with the haversine formula. Single-point GPS spikes are removed.
2. A point is *stationary* when its speed is below ``stop_speed_kmh``. A run of
   stationary points lasting at least ``min_stop_s`` is a **stop**.
3. A **trip** is a run of non-stop points. Trips are also split when the time
   gap between points exceeds ``gap_s`` or when the position jumps faster than
   ``max_speed_kmh`` (e.g. the device was switched off and moved).
4. Trips shorter than ``min_trip_m`` / ``min_trip_points`` are discarded.

Limits: with very dense sampling (≈1 Hz) GPS jitter can look like movement, so
raise ``stop_speed_kmh``; with very sparse sampling short stops are invisible.
"""

import numpy as np
import pandas as pd

from .geo import assign_grid, cell_bounds
from .utils import haversine_m, parse_datetime


def _group_ids(change):
    """Run ids from a 'value changed' flag; safe for pyarrow-backed booleans."""
    return pd.Series(np.cumsum(np.asarray(change.fillna(True), dtype=bool)), index=change.index)


def build_trajectory(df, m):
    """Tidy trajectory frame: vehicle, ts, lat, lon (+ reported speed)."""
    for key in ("timestamp", "latitude", "longitude"):
        if not m.get(key):
            raise ValueError(
                "Trajectory analysis needs timestamp, latitude and longitude "
                "mapped (see Column mapping in the sidebar).")

    traj = pd.DataFrame(index=df.index)
    if m.get("vehicle_id"):
        traj["vehicle"] = df[m["vehicle_id"]].astype("string").fillna("(unknown)").astype(object)
    else:
        traj["vehicle"] = pd.Series(["vehicle_1"] * len(df), index=df.index, dtype=object)
    traj["ts"] = parse_datetime(df[m["timestamp"]])
    traj["lat"] = pd.to_numeric(df[m["latitude"]], errors="coerce")
    traj["lon"] = pd.to_numeric(df[m["longitude"]], errors="coerce")
    if m.get("speed"):
        traj["reported_speed"] = pd.to_numeric(df[m["speed"]], errors="coerce")

    traj = traj.dropna(subset=["ts", "lat", "lon"])
    traj = traj[traj["lat"].between(-90, 90) & traj["lon"].between(-180, 180)]
    traj = traj[~((traj["lat"] == 0) & (traj["lon"] == 0))]
    traj = traj.drop_duplicates(subset=["vehicle", "ts"], keep="first")
    return traj.sort_values(["vehicle", "ts"], kind="stable").reset_index(drop=True)


def compute_motion(traj):
    """Add dt_s, dist_m and speed_kmh (NaN for each vehicle's first point)."""
    out = traj.sort_values(["vehicle", "ts"], kind="stable").reset_index(drop=True)
    grouped = out.groupby("vehicle", sort=False)

    out["dt_s"] = grouped["ts"].diff().dt.total_seconds()
    prev_lat, prev_lon = grouped["lat"].shift(), grouped["lon"].shift()
    out["dist_m"] = haversine_m(prev_lat, prev_lon, out["lat"], out["lon"])

    with np.errstate(divide="ignore", invalid="ignore"):
        speed = out["dist_m"] / out["dt_s"] * 3.6
    out["speed_kmh"] = speed.where(out["dt_s"] > 0)
    return out


def clean_trajectory(traj, max_speed_kmh=200.0, max_passes=3):
    """Remove single-point GPS spikes. Returns (clean_traj, n_removed)."""
    out = compute_motion(traj)
    removed = 0

    for _ in range(max_passes):
        jump_in = out["speed_kmh"] > max_speed_kmh
        jump_out = jump_in.groupby(out["vehicle"], sort=False).shift(-1)
        spike = jump_in & jump_out.fillna(False).astype(bool)
        if not spike.any():
            break
        removed += int(spike.sum())
        out = compute_motion(out.loc[~spike, ["vehicle", "ts", "lat", "lon"]
                                     + (["reported_speed"] if "reported_speed" in out else [])])

    return out, removed


def segment_trips(traj, stop_speed_kmh=3.0, min_stop_s=300, gap_s=1800,
                  min_trip_m=300, min_trip_points=3, max_speed_kmh=200.0):
    """Split trajectories into trips and stops.

    ``traj`` must already contain motion columns (see ``compute_motion`` or
    ``clean_trajectory``). Returns a dict with ``traj`` (trip_id / stop_id
    added), ``trips`` and ``stops`` DataFrames.
    """
    t = traj.reset_index(drop=True).copy()
    if t.empty:
        raise ValueError("There are no valid GPS points to analyse.")

    new_vehicle = (t["vehicle"] != t["vehicle"].shift()).fillna(True).astype(bool)
    brk = new_vehicle | (t["dt_s"] > gap_s) | (t["speed_kmh"] > max_speed_kmh)

    # The first point after a break has no meaningful speed of its own, so it
    # borrows the next point's speed (when that point belongs to the same run).
    next_ok = ~brk.shift(-1, fill_value=True)
    next_speed = t["speed_kmh"].shift(-1).where(next_ok)
    speed = t["speed_kmh"].where(~brk, next_speed).fillna(0.0)

    stationary = speed < stop_speed_kmh
    run = _group_ids(brk | (stationary != stationary.shift()))
    run_ts = t["ts"].groupby(run)
    run_seconds = (run_ts.transform("max") - run_ts.transform("min")).dt.total_seconds()
    in_stop = stationary & (run_seconds >= min_stop_s)

    seg = _group_ids(brk | (in_stop != in_stop.shift()))
    first_in_seg = seg != seg.shift()

    work = t.assign(
        seg=seg,
        in_stop=in_stop,
        dist_in=t["dist_m"].where(~brk & ~first_in_seg, 0.0).fillna(0.0),
        speed_ok=t["speed_kmh"].where(~brk),
    )

    # ---- trips ------------------------------------------------------------
    moving = work[~work["in_stop"]]
    trips = moving.groupby("seg").agg(
        vehicle=("vehicle", "first"),
        start_ts=("ts", "min"),
        end_ts=("ts", "max"),
        points=("ts", "size"),
        distance_m=("dist_in", "sum"),
        max_speed_kmh=("speed_ok", "max"),
        o_lat=("lat", "first"), o_lon=("lon", "first"),
        d_lat=("lat", "last"), d_lon=("lon", "last"),
    )
    trips["duration_min"] = (trips["end_ts"] - trips["start_ts"]).dt.total_seconds() / 60
    trips = trips[
        (trips["points"] >= min_trip_points)
        & (trips["distance_m"] >= min_trip_m)
        & (trips["duration_min"] > 0)
    ].copy()
    trips["distance_km"] = trips["distance_m"] / 1000
    trips["avg_speed_kmh"] = trips["distance_km"] / (trips["duration_min"] / 60)
    trips = trips.drop(columns="distance_m").sort_values(["vehicle", "start_ts"])
    seg_to_trip = {seg_id: n for n, seg_id in enumerate(trips.index, start=1)}
    trips.insert(0, "trip_id", range(1, len(trips) + 1))
    trips = trips.reset_index(drop=True)

    # ---- stops ------------------------------------------------------------
    stopped = work[work["in_stop"]]
    stops = stopped.groupby("seg").agg(
        vehicle=("vehicle", "first"),
        start_ts=("ts", "min"),
        end_ts=("ts", "max"),
        points=("ts", "size"),
        lat=("lat", "median"),
        lon=("lon", "median"),
    )
    stops["duration_min"] = (stops["end_ts"] - stops["start_ts"]).dt.total_seconds() / 60
    stops = stops.sort_values(["vehicle", "start_ts"])
    seg_to_stop = {seg_id: n for n, seg_id in enumerate(stops.index, start=1)}
    stops.insert(0, "stop_id", range(1, len(stops) + 1))
    stops = stops.reset_index(drop=True)

    out = t.copy()
    out["trip_id"] = work["seg"].map(seg_to_trip).where(~work["in_stop"]).fillna(-1).astype(int)
    out["stop_id"] = work["seg"].map(seg_to_stop).where(work["in_stop"]).fillna(-1).astype(int)

    return {"traj": out, "trips": trips, "stops": stops}


def analyse_trajectories(df, m, stop_speed_kmh=3.0, min_stop_min=5.0,
                         gap_min=30.0, min_trip_m=300.0, max_speed_kmh=200.0):
    """Convenience pipeline: tidy → clean → segment. Returns result dict."""
    traj = build_trajectory(df, m)
    clean, removed = clean_trajectory(traj, max_speed_kmh)
    result = segment_trips(
        clean, stop_speed_kmh=stop_speed_kmh, min_stop_s=min_stop_min * 60,
        gap_s=gap_min * 60, min_trip_m=min_trip_m, max_speed_kmh=max_speed_kmh)
    result["gps_spikes_removed"] = removed
    result["points_used"] = int(len(clean))
    result["vehicles"] = int(clean["vehicle"].nunique())
    return result


def trips_from_od_columns(df, m):
    """Build a trips table from datasets that already have origin/dest columns."""
    needed = ("origin_lat", "origin_lon", "dest_lat", "dest_lon")
    if not all(m.get(k) for k in needed):
        raise ValueError("Map origin and destination latitude/longitude columns first.")

    trips = pd.DataFrame(index=df.index)
    for key, name in zip(needed, ("o_lat", "o_lon", "d_lat", "d_lon")):
        trips[name] = pd.to_numeric(df[m[key]], errors="coerce")
    if m.get("timestamp"):
        trips["start_ts"] = parse_datetime(df[m["timestamp"]])
    if m.get("vehicle_id"):
        trips["vehicle"] = df[m["vehicle_id"]].astype("string")

    trips = trips.dropna(subset=list(("o_lat", "o_lon", "d_lat", "d_lon")))
    ok = (trips["o_lat"].between(-90, 90) & trips["d_lat"].between(-90, 90)
          & trips["o_lon"].between(-180, 180) & trips["d_lon"].between(-180, 180)
          & ~((trips["o_lat"] == 0) & (trips["o_lon"] == 0)))
    trips = trips[ok].reset_index(drop=True)
    trips["distance_km"] = haversine_m(
        trips["o_lat"], trips["o_lon"], trips["d_lat"], trips["d_lon"]) / 1000
    trips.insert(0, "trip_id", range(1, len(trips) + 1))
    return trips


# ---------------------------------------------------------------------------
# origin–destination
# ---------------------------------------------------------------------------

def od_flows(trips, cell_m=1000):
    """Aggregate trips into zone-to-zone flows on a square grid.

    Returns a DataFrame with origin/destination zone labels (Z1 = busiest),
    zone centres, number of trips and average distance.
    """
    if trips.empty:
        return pd.DataFrame()

    ref_lat = float(np.nanmedian(np.concatenate([trips["o_lat"], trips["d_lat"]])))
    oy, ox, steps = assign_grid(trips["o_lat"], trips["o_lon"], cell_m, ref_lat)
    dy, dx, _ = assign_grid(trips["d_lat"], trips["d_lon"], cell_m, ref_lat)

    data = pd.DataFrame({"oy": oy, "ox": ox, "dy": dy, "dx": dx})
    if "distance_km" in trips:
        data["distance_km"] = trips["distance_km"].to_numpy()
    else:
        data["distance_km"] = np.nan

    flows = data.groupby(["oy", "ox", "dy", "dx"]).agg(
        trips=("oy", "size"), avg_distance_km=("distance_km", "mean")).reset_index()

    # rank zones by total activity so labels are meaningful (Z1 = busiest)
    activity = pd.concat([
        flows.groupby(["oy", "ox"])["trips"].sum(),
        flows.groupby(["dy", "dx"])["trips"].sum().rename_axis(["oy", "ox"]),
    ]).groupby(level=[0, 1]).sum().sort_values(ascending=False)
    labels = {key: f"Z{rank}" for rank, key in enumerate(activity.index, start=1)}

    flows["origin"] = [labels[(a, b)] for a, b in zip(flows["oy"], flows["ox"])]
    flows["destination"] = [labels[(a, b)] for a, b in zip(flows["dy"], flows["dx"])]

    o_lat0, o_lat1, o_lon0, o_lon1 = cell_bounds(flows["oy"], flows["ox"], steps)
    d_lat0, d_lat1, d_lon0, d_lon1 = cell_bounds(flows["dy"], flows["dx"], steps)
    flows["o_lat"], flows["o_lon"] = (o_lat0 + o_lat1) / 2, (o_lon0 + o_lon1) / 2
    flows["d_lat"], flows["d_lon"] = (d_lat0 + d_lat1) / 2, (d_lon0 + d_lon1) / 2
    flows["intrazonal"] = flows["origin"] == flows["destination"]

    flows = flows.drop(columns=["oy", "ox", "dy", "dx"]).sort_values(
        "trips", ascending=False).reset_index(drop=True)
    return flows[["origin", "destination", "trips", "avg_distance_km", "intrazonal",
                  "o_lat", "o_lon", "d_lat", "d_lon"]]


def od_matrix(flows, top_n=10):
    """Origin × destination matrix for the ``top_n`` busiest zones."""
    if flows.empty:
        return pd.DataFrame()

    activity = pd.concat([
        flows.groupby("origin")["trips"].sum(),
        flows.groupby("destination")["trips"].sum(),
    ]).groupby(level=0).sum().sort_values(ascending=False)
    zones = list(activity.head(top_n).index)

    subset = flows[flows["origin"].isin(zones) & flows["destination"].isin(zones)]
    matrix = subset.pivot_table(index="origin", columns="destination",
                                values="trips", aggfunc="sum", fill_value=0)
    return matrix.reindex(index=zones, columns=zones, fill_value=0)


def trip_summary(trips, stops=None):
    """Headline numbers for the trips page."""
    summary = {
        "trips": int(len(trips)),
        "vehicles": int(trips["vehicle"].nunique()) if "vehicle" in trips else None,
        "total_km": float(trips["distance_km"].sum()) if "distance_km" in trips else None,
        "median_km": float(trips["distance_km"].median()) if len(trips) else None,
    }
    if "duration_min" in trips:
        summary["median_min"] = float(trips["duration_min"].median()) if len(trips) else None
        summary["avg_speed_kmh"] = (float(trips["avg_speed_kmh"].mean())
                                    if len(trips) else None)
    if stops is not None:
        summary["stops"] = int(len(stops))
        summary["median_stop_min"] = (float(stops["duration_min"].median())
                                      if len(stops) else None)
    return summary
