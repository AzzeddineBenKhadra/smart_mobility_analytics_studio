"""Maps page: points, congestion, animation, grid zones, district polygons."""

import pandas as pd
import streamlit as st

from mobility import geo
from mobility import insights as I
from mobility import mapping as maps
from mobility.state import cached
from mobility.ui import metric_row, safe_section, show_df, show_map
from mobility.utils import numeric_columns
from .common import get_panel, mapping_key


def _points_tab(df, m):
    lat_col, lon_col = m["latitude"], m["longitude"]
    c1, c2, c3 = st.columns(3)
    mode = c1.radio("Display", ["Points", "Clusters", "Heatmap"], horizontal=True)
    max_points = c2.slider("Max points drawn", 500, 20000, 2000, step=500)

    color_col = weight_col = None
    if mode == "Points":
        choice = c3.selectbox("Colour by (optional)", ["None"] + list(df.columns))
        color_col = None if choice == "None" else choice
    elif mode == "Heatmap":
        choice = c3.selectbox("Weight by (optional)", ["None"] + numeric_columns(df))
        weight_col = None if choice == "None" else choice

    fmap, legend_html, shown, total = maps.points_map(
        df, lat_col, lon_col, mode, color_col, weight_col, max_points)
    show_map(fmap, key="map_points")
    if legend_html:
        st.markdown(legend_html, unsafe_allow_html=True)
    st.caption(f"Showing {shown:,} of {total:,} valid records"
               + (" (random sample — raise the slider to draw more)." if total > shown else "."))


def _segments_tab(df, m):
    panel = get_panel(df, m)
    if panel is None or "ci" not in panel.columns:
        st.info("Needs timestamp and speed mapped (for the congestion index).")
        return
    if "lat" not in panel.columns:
        st.info("Map latitude and longitude to place segments on the map.")
        return
    ranking = cached("ranking_map", lambda: I.segment_ranking(panel), mapping_key(m))
    show_map(maps.segment_map(ranking), key="map_segments")
    st.caption("Colour = average congestion index (green free, red congested); "
               "circle size = average volume. Click a segment for details.")


def _animation_tab(df, m):
    panel = get_panel(df, m)
    if panel is None or "ci" not in panel.columns or "lat" not in panel.columns:
        st.info("Needs timestamp, speed, latitude and longitude mapped.")
        return
    show_map(maps.congestion_time_map(panel), key="map_anim")
    st.caption("Press play to watch the average congestion of each segment through the "
               "24 hours of an average day.")


def _grid_tab(df, m):
    lat_col, lon_col = m["latitude"], m["longitude"]
    c1, c2, c3 = st.columns(3)
    cell = c1.slider("Zone size (metres)", 100, 10000, 1000, step=100, key="grid_cell")
    numbers = numeric_columns(df)
    value_choice = c2.selectbox("Value", ["Record count"] + numbers, key="grid_value")
    agg = "count"
    value_col = None
    if value_choice != "Record count":
        value_col = value_choice
        agg = c3.selectbox("Aggregate", ["mean", "sum", "median", "max"], key="grid_agg")

    cells = cached("grid", lambda: geo.aggregate_grid(df, lat_col, lon_col, cell, value_col, agg),
                   (lat_col, lon_col), cell, value_col, agg)
    if cells.empty:
        st.warning("No valid coordinates.")
        return
    if len(cells) > 6000:
        st.warning(f"{len(cells):,} zones would be drawn — increase the zone size.")
        return

    label = "Records" if value_col is None else f"{agg} of {value_col}"
    metric_row([("Zones", f"{len(cells):,}"), ("Records", f"{int(cells['count'].sum()):,}")])
    show_map(maps.grid_map(cells, label), key="map_grid")
    with st.expander("Zone table"):
        show_df(cells.sort_values("value", ascending=False).head(200)[
            ["cell", "count", "value", "center_lat", "center_lon"]], hide_index=True)


def _districts_tab(df, m):
    lat_col, lon_col = m["latitude"], m["longitude"]
    st.caption("Upload a GeoJSON file with polygon boundaries (districts, traffic "
               "analysis zones, ...). Points are assigned to the polygon that contains them.")
    upload = st.file_uploader("District boundaries (GeoJSON)", type=["geojson", "json"],
                              key="geojson_upload")
    if upload is None:
        st.info("No boundaries uploaded yet.")
        return

    geojson = geo.parse_geojson(upload.getvalue())
    names = geo.property_names(geojson)
    st.success(f"Loaded {len(geojson['features']):,} polygon feature(s).")

    c1, c2, c3 = st.columns(3)
    name_prop = c1.selectbox("Zone name property", ["(none)"] + names, key="geo_name")
    numbers = numeric_columns(df)
    value_choice = c2.selectbox("Value", ["Record count"] + numbers, key="geo_value")
    agg, value_col = "count", None
    if value_choice != "Record count":
        value_col = value_choice
        agg = c3.selectbox("Aggregate", ["mean", "sum", "median", "max"], key="geo_agg")

    with st.spinner("Assigning points to zones..."):
        table = geo.aggregate_by_polygon(
            df, lat_col, lon_col, geojson,
            None if name_prop == "(none)" else name_prop, value_col, agg)

    outside = len(df) - int(table["count"].sum())
    metric_row([("Zones", f"{len(table):,}"),
                ("Zones with data", f"{int((table['count'] > 0).sum()):,}"),
                ("Points outside all zones", f"{outside:,}")])
    label = "Records" if value_col is None else f"{agg} of {value_col}"
    show_map(maps.choropleth_map(geojson, table, label), key="map_choro")
    show_df(table.sort_values("value", ascending=False), hide_index=True)


def render(df, m):
    st.header("Maps")

    if not (m.get("latitude") and m.get("longitude")):
        st.info("Map both latitude and longitude in the sidebar (**Column mapping**) to "
                "display maps — or load the sample data.")
        return

    def count_valid():
        lat = pd.to_numeric(df[m["latitude"]], errors="coerce")
        lon = pd.to_numeric(df[m["longitude"]], errors="coerce")
        return int((lat.between(-90, 90) & lon.between(-180, 180)).sum())

    valid = cached("valid_coords", count_valid, m["latitude"], m["longitude"])
    st.metric("Rows with valid coordinates", f"{valid:,}")
    if valid == 0:
        st.warning("No valid coordinate pairs were found.")
        return

    tabs = st.tabs(["Points & heatmap", "Segment congestion", "Congestion by hour",
                    "Grid zones", "District polygons"])
    with tabs[0], safe_section("Point map"):
        _points_tab(df, m)
    with tabs[1], safe_section("Segment map"):
        _segments_tab(df, m)
    with tabs[2], safe_section("Congestion animation"):
        _animation_tab(df, m)
    with tabs[3], safe_section("Grid zones"):
        _grid_tab(df, m)
    with tabs[4], safe_section("District polygons"):
        _districts_tab(df, m)
