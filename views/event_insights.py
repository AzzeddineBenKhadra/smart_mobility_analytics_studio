"""Mobility Insights for event data (e.g. incident reports: one row = one event)."""

import pandas as pd
import plotly.express as px
import streamlit as st

from mobility import insights as I
from mobility.config import DAY_NAMES
from mobility.geo import aggregate_grid
from mobility.state import cached
from mobility.ui import metric_row, safe_section, show_df, show_plot
from mobility.utils import is_text_col
from .common import mapping_key


def _category_columns(df):
    cols = []
    for c in df.columns:
        if is_text_col(df[c]):
            n = df[c].nunique(dropna=True)
            if 2 <= n <= 1000:
                cols.append(c)
    return cols


def render(df, m):
    st.header("Mobility Insights — event data")
    st.info("No speed or volume column is mapped, so each row is treated as one **event** "
            "(for example an incident report). The charts show **when and where events are "
            "reported**, not measured congestion.")

    weekend_days = tuple(m.get("weekend_days", (5, 6)))
    with st.expander("Settings", expanded=False):
        window = st.selectbox("Period to analyse",
                              ["Last 12 months", "Last 24 months", "All history"],
                              index=0, key="ev_window")
    max_days = {"Last 12 months": 365, "Last 24 months": 730, "All history": 0}[window]

    with st.spinner("Preparing events..."):
        panel = cached("event_insight_panel",
                       lambda: I.prepare_event_panel(df, m, by_zone=False, max_days=max_days),
                       mapping_key(m), max_days)
    if panel is None or panel.empty:
        st.warning("No usable rows: the timestamp column could not be parsed.")
        return

    counts = I.event_counts_by_hour(panel, weekend_days)
    days = counts.index.normalize().nunique()
    metric_row([("Events", f"{len(panel):,}"), ("Days covered", f"{days:,}"),
                ("Average per day", f"{len(panel) / max(days, 1):.0f}"),
                ("Period", f"{panel['ts'].min():%Y-%m-%d} → {panel['ts'].max():%Y-%m-%d}")])

    tabs = st.tabs(["Key findings", "Time patterns", "Hotspots", "Categories", "Trend"])

    with tabs[0], safe_section("Key findings"):
        for line in I.generate_event_findings(panel, weekend_days):
            st.markdown(f"- {line}")

    with tabs[1], safe_section("Time patterns"):
        prof = counts.groupby(["day_type", "hour"])["n"].mean().reset_index()
        fig = px.line(prof, x="hour", y="n", color="day_type", markers=True,
                      labels={"hour": "Hour of day", "n": "Average events per hour",
                              "day_type": ""},
                      title="Events by hour of day")
        show_plot(fig, key="ev_hour")
        heat = I.event_heat_table(counts)
        fig = px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                        labels={"x": "Hour", "y": "", "color": "events / hour"},
                        title="Average events per hour: weekday × hour")
        show_plot(fig, key="ev_heat")
        st.caption("Empty hours count as zero. Missing periods in the source data "
                   "(outages) pull the averages down.")

    with tabs[2], safe_section("Hotspots"):
        if not (m.get("latitude") and m.get("longitude")) or "lat" not in panel.columns:
            st.info("Hotspots need latitude and longitude mapped in the sidebar.")
        else:
            cell_m = st.selectbox("Zone size (m)", [500, 1000, 2000], index=1, key="ev_cell")
            geo = panel.dropna(subset=["lat", "lon"])
            cells = aggregate_grid(geo, "lat", "lon", cell_m)
            cells = cells.sort_values("count", ascending=False).head(15).copy()
            cells["share_pct"] = (cells["count"] / len(geo) * 100).round(2)
            st.caption(f"Top 15 of the zones ({cell_m} m squares). Swapped latitude/longitude "
                       "values are repaired automatically here; far-away points are ignored.")
            show_df(cells[["cell", "count", "share_pct", "center_lat", "center_lon"]],
                    hide_index=True)
            fig = px.bar(cells, x="cell", y="count", title="Busiest zones")
            show_plot(fig, key="ev_cells")
            st.caption("Paste a zone's centre into Google Maps to see which area it is, "
                       "or use the app's **Maps** page.")

    with tabs[3], safe_section("Categories"):
        cat_cols = _category_columns(df)
        if not cat_cols:
            st.info("No text column with a manageable number of categories was found.")
        else:
            col = st.selectbox("Category column", cat_cols, key="ev_cat")
            st.caption("Top categories across the whole file.")
            top = df[col].value_counts().head(15).reset_index()
            top.columns = ["category", "count"]
            fig = px.bar(top.iloc[::-1], x="count", y="category", orientation="h",
                         title=f"Top categories in {col}")
            show_plot(fig, key="ev_topcat")

    with tabs[4], safe_section("Trend"):
        monthly = panel.groupby(panel["ts"].dt.to_period("M")).size()
        monthly.index = monthly.index.to_timestamp()
        fig = px.bar(monthly.reset_index(name="events"), x="ts", y="events",
                     labels={"ts": "Month"}, title="Events per month")
        show_plot(fig, key="ev_month")
        st.caption("Sudden jumps or gaps usually mean a change in how the data was "
                   "recorded, not a real change in traffic. Check before drawing conclusions.")
