"""Trips & OD page: GPS trip/stop detection and origin–destination flows."""

import pandas as pd
import plotly.express as px
import streamlit as st

from mobility import mapping as maps
from mobility import trips as T
from mobility.state import cached
from mobility.ui import metric_row, safe_section, show_df, show_map, show_plot
from .common import missing_fields_message

LARGE_ROWS = 300_000


def _show_od(trips, key):
    st.caption("Zones are a square grid. Z1 is the busiest zone. Lines are straight "
               "'desire lines' between zone centres, not routes.")
    c1, c2, c3 = st.columns(3)
    cell = c1.slider("Zone size (metres)", 500, 10000, 2000, step=500, key=f"{key}_cell")
    top_n = c2.slider("Flows drawn", 10, 200, 40, step=10, key=f"{key}_top")
    min_trips = c3.slider("Minimum trips per flow", 1, 20, 1, key=f"{key}_min")

    flows = cached("od_flows", lambda: T.od_flows(trips, cell), key, cell)
    if flows.empty:
        st.info("No trips to aggregate.")
        return

    inter = flows[~flows["intrazonal"]]
    metric_row([("Zone pairs with trips", f"{len(flows):,}"),
                ("Trips between zones", f"{int(inter['trips'].sum()):,}"),
                ("Trips inside one zone", f"{int(flows.loc[flows['intrazonal'], 'trips'].sum()):,}")])

    with safe_section("Drawing the flow map"):
        show_map(maps.od_map(flows, top_n, min_trips), key=f"{key}_map")

    st.subheader("Top flows")
    show_df(flows.head(30).drop(columns=["o_lat", "o_lon", "d_lat", "d_lon"]).rename(
        columns={"origin": "Origin", "destination": "Destination", "trips": "Trips",
                 "avg_distance_km": "Avg distance (km)", "intrazonal": "Same zone"}),
        hide_index=True)

    matrix = T.od_matrix(flows, 10)
    if not matrix.empty:
        fig = px.imshow(matrix, text_auto=True, aspect="auto",
                        color_continuous_scale="Blues",
                        labels={"x": "Destination", "y": "Origin", "color": "Trips"},
                        title="Origin–destination matrix (10 busiest zones)")
        show_plot(fig, key=f"{key}_matrix")


def _trajectory_mode(df, m):
    st.markdown(
        "Detects **trips** and **stops** in GPS traces: a stop is a stretch with speed "
        "below a threshold lasting at least the minimum duration; trips are the movement "
        "between stops.")
    if not m.get("vehicle_id"):
        st.warning("No vehicle / device ID is mapped, so all points are treated as one "
                   "vehicle. Map an ID column for meaningful trips.")

    with st.expander("Detection settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        stop_speed = c1.slider("Stationary below (km/h)", 1.0, 10.0, 3.0, 0.5,
                               help="Raise for dense (≈1 s) GPS data with jitter.")
        min_stop = c2.slider("Minimum stop (minutes)", 1.0, 60.0, 5.0, 1.0)
        gap = c3.slider("Split trip at gaps over (minutes)", 5.0, 240.0, 30.0, 5.0)
        c4, c5 = st.columns(2)
        min_trip_m = c4.slider("Minimum trip length (m)", 50.0, 2000.0, 300.0, 50.0)
        max_speed = c5.slider("GPS glitch above (km/h)", 100.0, 400.0, 200.0, 10.0)

    key = (m.get("vehicle_id"), m.get("timestamp"), m.get("latitude"), m.get("longitude"),
           stop_speed, min_stop, gap, min_trip_m, max_speed)

    store = st.session_state["_cache"]
    already = ("trip_analysis", (key,)) in store
    if len(df) > LARGE_ROWS and not already:
        st.info(f"{len(df):,} points: the analysis may take a while.")
        if not st.button("Run trip analysis", type="primary"):
            return

    with st.spinner("Detecting trips and stops..."):
        result = cached("trip_analysis", lambda: T.analyse_trajectories(
            df, m, stop_speed, min_stop, gap, min_trip_m, max_speed), key)

    trips, stops, traj = result["trips"], result["stops"], result["traj"]
    summary = T.trip_summary(trips, stops)

    metric_row([("Vehicles", f"{result['vehicles']:,}"),
                ("Trips", f"{summary['trips']:,}"),
                ("Stops", f"{summary['stops']:,}"),
                ("Total distance", f"{summary['total_km']:,.0f} km")])
    metric_row([("Median trip", f"{summary['median_km'] or 0:.1f} km"),
                ("Median duration", f"{summary['median_min'] or 0:.0f} min"),
                ("Mean trip speed", f"{summary['avg_speed_kmh'] or 0:.0f} km/h"),
                ("GPS spikes removed", f"{result['gps_spikes_removed']:,}")])

    if trips.empty:
        st.warning("No trips were found. Try lowering the minimum stop or trip length.")
        return

    tabs = st.tabs(["Summary", "Trip map", "Stops", "Origin–destination", "Tables"])

    with tabs[0], safe_section("Trip summary"):
        c1, c2 = st.columns(2)
        with c1:
            show_plot(px.histogram(trips, x="distance_km", nbins=40,
                                   labels={"distance_km": "Trip distance (km)"},
                                   title="Trip distance"), key="tr_dist")
        with c2:
            show_plot(px.histogram(trips, x="duration_min", nbins=40,
                                   labels={"duration_min": "Trip duration (min)"},
                                   title="Trip duration"), key="tr_dur")
        by_hour = (trips.groupby(trips["start_ts"].dt.hour).size()
                   .rename("Trips").rename_axis("Hour").reset_index())
        show_plot(px.bar(by_hour, x="Hour", y="Trips", title="Trip starts by hour of day"),
                  key="tr_hour")

    with tabs[1], safe_section("Trip map"):
        st.caption("A random sample of trips, coloured by average speed (red = slow).")
        limit = st.slider("Trips drawn", 20, 400, 150, 10, key="tr_limit")
        show_map(maps.trips_map(traj, trips, limit), key="tr_map")

    with tabs[2], safe_section("Stops"):
        if stops.empty:
            st.info("No stops detected.")
        else:
            show_plot(px.histogram(stops, x="duration_min", nbins=40,
                                   labels={"duration_min": "Stop duration (min)"},
                                   title="How long do vehicles stop?"), key="st_dur")
            show_map(maps.stops_map(stops), key="st_map")

    with tabs[3], safe_section("Origin–destination"):
        _show_od(trips, "tr_od")

    with tabs[4]:
        st.subheader("Trips")
        show_df(trips)
        st.download_button("Download trips (CSV)", trips.to_csv(index=False).encode("utf-8"),
                           "trips.csv", "text/csv")
        st.subheader("Stops")
        show_df(stops)
        st.download_button("Download stops (CSV)", stops.to_csv(index=False).encode("utf-8"),
                           "stops.csv", "text/csv")


def _od_table_mode(df, m):
    trips = cached("od_trips", lambda: T.trips_from_od_columns(df, m),
                   (m.get("origin_lat"), m.get("origin_lon"), m.get("dest_lat"),
                    m.get("dest_lon"), m.get("timestamp")))
    if trips.empty:
        st.warning("No rows with valid origin and destination coordinates.")
        return

    metric_row([("Trips", f"{len(trips):,}"),
                ("Median straight-line distance", f"{trips['distance_km'].median():.1f} km")])
    show_plot(px.histogram(trips, x="distance_km", nbins=50,
                           labels={"distance_km": "Straight-line distance (km)"},
                           title="Trip distance"), key="od_dist")
    if "start_ts" in trips and trips["start_ts"].notna().any():
        by_hour = (trips.groupby(trips["start_ts"].dt.hour).size()
                   .rename("Trips").rename_axis("Hour").reset_index())
        show_plot(px.bar(by_hour, x="Hour", y="Trips", title="Trip starts by hour of day"),
                  key="od_hour")
    _show_od(trips, "od_tbl")


def render(df, m):
    st.header("Trips & Origin–Destination")

    has_traj = all(m.get(k) for k in ("timestamp", "latitude", "longitude"))
    has_od = all(m.get(k) for k in ("origin_lat", "origin_lon", "dest_lat", "dest_lon"))

    if not (has_traj or has_od):
        st.info(
            "This page needs either GPS points (timestamp + latitude + longitude, ideally "
            f"a vehicle ID — missing: {missing_fields_message(m, ['timestamp', 'latitude', 'longitude'])}) "
            "or a trip table with origin and destination coordinates. "
            "Set them under **Column mapping** in the sidebar, or load the sample GPS data.")
        return

    modes = []
    if has_traj:
        modes.append("GPS trajectories")
    if has_od:
        modes.append("Trip table (origin / destination columns)")
    mode = modes[0] if len(modes) == 1 else st.radio("Data type", modes, horizontal=True)

    with safe_section("Trip analysis"):
        if mode == "GPS trajectories":
            _trajectory_mode(df, m)
        else:
            _od_table_mode(df, m)
