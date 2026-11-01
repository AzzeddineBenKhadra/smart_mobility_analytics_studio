"""Mobility Insights page: findings, time patterns, segments, speed-flow, incidents."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mobility import insights as I
from mobility.config import DAY_NAMES, MAX_PLOT_POINTS
from mobility.state import cached
from mobility.ui import metric_row, safe_section, show_df, show_plot
from mobility.utils import numeric_columns, sample_for_plot
from .common import get_panel, mapping_key, missing_fields_message

METRIC_LABELS = {"ci": "Congestion index", "speed": "Speed (km/h)", "volume": "Volume per record"}


def _metric_options(panel):
    return [k for k in ("ci", "speed", "volume") if k in panel.columns]


def _scale_for(metric):
    # congestion: high is bad (red); speed/volume: high is good/high flow
    return "RdYlGn_r" if metric == "ci" else ("RdYlGn" if metric == "speed" else "Viridis")


def _findings_tab(panel, m, weekend_days):
    n_incidents = None
    if "speed" in panel.columns:
        try:
            flagged, _ = cached(
                "incidents_default",
                lambda: I.detect_anomalies(panel, "speed", "seasonal", 4.0, direction="drops"),
                mapping_key(m), tuple(weekend_days))
            interval = cached("interval", lambda: I.infer_interval_minutes(panel),
                              mapping_key(m))
            n_incidents = len(I.group_incidents(flagged, interval, min_points=2))
        except ValueError:
            n_incidents = None

    findings = I.generate_findings(panel, weekend_days, n_incidents)
    st.subheader("Key findings")
    st.markdown("\n".join(f"- {f}" for f in findings))
    st.caption("Findings are generated directly from the data with fixed rules; "
               "check them against the charts before quoting them.")
    st.download_button("Download findings (.txt)", "\n".join(f"- {f}" for f in findings),
                       file_name="mobility_findings.txt", mime="text/plain")

    levels = I.level_distribution(panel)
    if not levels.empty:
        st.subheader("How much of the time is the network congested?")
        fig = px.bar(levels, x="Level", y="Share %", text="Share %",
                     color="Level", color_discrete_map={
                         "Free flow": "#1a9850", "Light": "#a6d96a",
                         "Moderate": "#fdae61", "Heavy": "#d73027"})
        fig.update_layout(showlegend=False)
        show_plot(fig, key="in_levels")
        st.caption("Congestion index = 1 − speed / free-flow speed. "
                   "Levels: free < 0.15 ≤ light < 0.30 ≤ moderate < 0.50 ≤ heavy.")


def _time_tab(panel, m, weekend_days):
    options = _metric_options(panel)
    metric = st.selectbox("Metric", options, format_func=METRIC_LABELS.get, key="in_metric")

    profile = I.time_profile(panel, metric, weekend_days)
    fig = px.line(profile, x="hour", y="mean", color="day_type", markers=True,
                  labels={"hour": "Hour of day", "mean": METRIC_LABELS[metric],
                          "day_type": ""},
                  title=f"{METRIC_LABELS[metric]} by hour — weekday vs weekend")
    fig.update_xaxes(dtick=2)
    show_plot(fig, key="in_profile")

    heat = I.heat_table(panel, metric)
    fig = px.imshow(heat, aspect="auto", color_continuous_scale=_scale_for(metric),
                    labels={"x": "Hour of day", "y": "", "color": METRIC_LABELS[metric]},
                    title=f"{METRIC_LABELS[metric]} — weekday × hour")
    show_plot(fig, key="in_heat")

    st.subheader("Trend over time")
    work = panel[["ts", "road", metric]].dropna()
    days = (work["ts"].max() - work["ts"].min()).total_seconds() / 86400 if len(work) else 0
    freqs = {"15 minutes": "15min", "Hourly": "1h", "6 hours": "6h",
             "Daily": "1D", "Weekly": "1W"}
    default = ("15 minutes" if days <= 3 else "Hourly" if days <= 30
               else "Daily" if days <= 365 else "Weekly")
    label = st.selectbox("Aggregation interval", list(freqs),
                         index=list(freqs).index(default), key="in_freq")
    by_road = panel["road"].nunique() > 1 and st.checkbox(
        "Split by road (5 busiest)", value=False, key="in_by_road")

    if by_road:
        top = work["road"].value_counts().head(5).index
        trend = (work[work["road"].isin(top)]
                 .groupby(["road", pd.Grouper(key="ts", freq=freqs[label])])[metric]
                 .mean().dropna().reset_index())
        fig = px.line(trend, x="ts", y=metric, color="road")
    else:
        trend = (work.set_index("ts")[metric].sort_index().resample(freqs[label])
                 .mean().dropna().reset_index())
        fig = px.line(trend, x="ts", y=metric)
    fig.update_layout(title=f"{label} average of {METRIC_LABELS[metric]}",
                      xaxis_title="Time", yaxis_title=METRIC_LABELS[metric])
    if trend.empty:
        st.info("No values available for this trend.")
    else:
        show_plot(fig, key="in_trend")


def _segments_tab(panel):
    if "ci" not in panel.columns:
        st.info("Segment ranking needs a mapped speed column.")
        return
    ranking = I.segment_ranking(panel)
    if ranking.empty:
        st.info("No segments with speed data.")
        return

    st.subheader("Most congested segments")
    display = ranking.rename(columns={
        "road": "Segment", "records": "Records", "mean_speed": "Mean speed (km/h)",
        "free_flow": "Free-flow (km/h)", "mean_ci": "Mean congestion index",
        "pct_heavy": "% time heavy", "planning_time_index": "Planning time index",
        "mean_volume": "Mean volume", "worst_hour": "Worst hour"})
    show_df(display.drop(columns=[c for c in ("lat", "lon") if c in display.columns]),
            hide_index=True)

    top = ranking.head(25).sort_values("mean_ci")
    fig = px.bar(top, x="mean_ci", y="road", orientation="h", color="mean_ci",
                 color_continuous_scale="RdYlGn_r",
                 labels={"mean_ci": "Mean congestion index", "road": ""},
                 title="Congestion index by segment")
    fig.update_layout(coloraxis_showscale=False)
    show_plot(fig, key="in_rank")
    st.caption("Planning time index = free-flow speed ÷ 5th-percentile speed: how much "
               "more time to budget to arrive on time 95% of the time.")

    st.subheader("Worst time slots")
    show_df(I.worst_time_slots(panel, 10).rename(columns={
        "mean_ci": "Mean congestion index", "records": "Records"}), hide_index=True)

    if len(ranking) > 1:
        st.subheader("Inspect one segment")
        road = st.selectbox("Segment", ranking["road"].tolist(), key="in_road")
        sub = panel[panel["road"] == road]
        prof = I.time_profile(sub, "ci")
        fig = px.line(prof, x="hour", y="mean", color="day_type", markers=True,
                      labels={"hour": "Hour of day", "mean": "Congestion index", "day_type": ""},
                      title=f"{road}: congestion by hour")
        show_plot(fig, key="in_road_profile")


def _speed_flow_tab(panel):
    if not {"speed", "volume"}.issubset(panel.columns):
        st.info("The speed–flow relationship needs both speed and volume mapped.")
        return

    data = panel.dropna(subset=["speed", "volume"])
    data, sampled = sample_for_plot(data[["road", "speed", "volume"]], 20_000)
    colour = "road" if data["road"].nunique() <= 12 else None
    fig = px.scatter(data, x="volume", y="speed", color=colour, opacity=0.45,
                     labels={"volume": "Volume per record", "speed": "Speed (km/h)"},
                     title="Speed–flow diagram")
    show_plot(fig, key="in_flow")
    st.caption("Speed stays high until flow approaches capacity, then drops: the curve's "
               "bend shows where a road starts to congest." +
               (" Random sample of 20,000 points shown." if sampled else ""))

    capacity = I.capacity_estimate(panel)
    if not capacity.empty:
        st.subheader("Empirical capacity by segment")
        show_df(capacity.rename(columns={
            "road": "Segment", "capacity_q95": "Capacity (95th pct volume)",
            "speed_near_capacity": "Speed near capacity (km/h)",
            "free_flow": "Free-flow (km/h)"}), hide_index=True)


def _incidents_tab(panel, m, weekend_days):
    metrics = [k for k in ("speed", "volume") if k in panel.columns]
    if not metrics:
        st.info("Incident detection needs a mapped speed or volume column.")
        return

    c1, c2, c3 = st.columns(3)
    metric = c1.selectbox("Metric", metrics, format_func=METRIC_LABELS.get, key="inc_metric")
    method = c2.selectbox("Method", ["Seasonal robust z-score", "Isolation Forest"],
                          key="inc_method")
    direction = c3.selectbox("Direction", ["drops", "spikes", "both"], key="inc_dir")

    method_key = "seasonal" if method.startswith("Seasonal") else "isolation"
    z_thr, contamination = 4.0, 0.01
    if method_key == "seasonal":
        z_thr = st.slider("Sensitivity (|z| threshold — lower finds more)", 2.0, 8.0, 4.0, 0.5,
                          key="inc_z")
    else:
        contamination = st.slider("Expected share of anomalies", 0.001, 0.05, 0.01, 0.001,
                                  format="%.3f", key="inc_cont")
    min_points = st.slider("Minimum consecutive observations per incident", 1, 6, 2,
                           key="inc_min")

    st.caption(
        "Seasonal method: compares each observation with the typical value for the same "
        "road, weekday and hour, so recurring rush-hour slowdowns are not flagged. "
        "Isolation Forest looks at speed and volume together." )

    with safe_section("Incident detection"):
        key = (metric, method_key, z_thr, contamination, direction)
        flagged, full = cached(
            "anomalies",
            lambda: I.detect_anomalies(panel, metric, method_key, z_thr, contamination,
                                       direction),
            mapping_key(m), key)
        interval = cached("interval", lambda: I.infer_interval_minutes(panel), mapping_key(m))
        incidents = I.group_incidents(flagged, interval, min_points=min_points)

        metric_row([("Flagged observations", f"{len(flagged):,}"),
                    ("Incidents", f"{len(incidents):,}"),
                    ("Observation interval", f"{interval:g} min")])
        if incidents.empty:
            st.success("No incidents with the current settings.")
            return

        show_df(incidents.rename(columns={
            "road": "Segment", "start": "Start", "end": "End", "duration_min": "Minutes",
            "points": "Obs.", "worst_z": "Worst z", "lowest_value": "Lowest value",
            "expected": "Expected"}), hide_index=True)

        options = [f"{r.road} — {r.start:%Y-%m-%d %H:%M}" for r in incidents.itertuples()]
        chosen = st.selectbox("Zoom to incident", options, key="inc_zoom")
        row = incidents.iloc[options.index(chosen)]
        lo = row["start"] - pd.Timedelta(hours=12)
        hi = row["end"] + pd.Timedelta(hours=12)
        window = full[(full["road"] == row["road"]) & (full["ts"].between(lo, hi))]
        hits = window[window["flag"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=window["ts"], y=window[metric], mode="lines",
                                 name="Observed"))
        fig.add_trace(go.Scatter(x=window["ts"], y=window["expected"], mode="lines",
                                 name="Expected", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=hits["ts"], y=hits[metric], mode="markers",
                                 name="Flagged", marker=dict(size=9, color="#d73027")))
        fig.update_layout(title=f"{row['road']} around {row['start']:%d %b %H:%M}",
                          xaxis_title="Time", yaxis_title=METRIC_LABELS[metric])
        show_plot(fig, key="inc_chart")


def _distributions_tab(df, panel, m):
    numbers = numeric_columns(df)
    for col, title in ((m.get("speed"), "Speed"), (m.get("volume"), "Traffic volume")):
        if not col or col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        st.subheader(title)
        cols = st.columns(5)
        cols[0].metric("Average", f"{values.mean():,.2f}")
        cols[1].metric("Median", f"{values.median():,.2f}")
        cols[2].metric("85th pct", f"{values.quantile(0.85):,.2f}")
        cols[3].metric("Maximum", f"{values.max():,.2f}")
        cols[4].metric("Minimum", f"{values.min():,.2f}")
        plot_data, _ = sample_for_plot(values.to_frame(col), MAX_PLOT_POINTS)
        show_plot(px.histogram(plot_data, x=col, title=f"{title} distribution"),
                  key=f"in_dist_{title}")

    st.subheader("Numeric correlations")
    from mobility.detection import find_identifier_columns
    ids = find_identifier_columns(df, exclude={m.get("speed"), m.get("volume")})
    usable = [c for c in numbers if c not in ids]
    if len(usable) < 2:
        st.info("At least two numeric (non-ID) columns are needed.")
        return
    chosen = st.multiselect("Columns", usable, default=usable[:12], key="in_corr_cols")
    if len(chosen) >= 2:
        corr = df[chosen].corr(numeric_only=True)
        show_plot(px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="Correlation matrix"), key="in_corr")


def render(df, m):
    st.header("Mobility Insights")

    if not m.get("timestamp") or not (m.get("speed") or m.get("volume")):
        needs = missing_fields_message(m, ["timestamp"]) or "speed or volume"
        st.info(f"This page needs a timestamp and a speed and/or volume column. "
                f"Missing: {needs}. Set them in the sidebar under **Column mapping**.")
        return

    with st.expander("Settings"):
        ffq = st.slider("Free-flow speed = percentile of observed speeds", 70, 99, 85,
                        help="Used for the congestion index. 85 is a common choice.") / 100

    weekend_days = m.get("weekend_days", (5, 6))
    with st.spinner("Preparing time series..."):
        panel = get_panel(df, m, ffq)
    if panel is None:
        st.warning("No usable rows: the timestamp column could not be parsed.")
        return

    summary = I.network_summary(panel)
    items = [("Observations", f"{summary['records']:,}"),
             ("Segments", f"{summary['roads']:,}"),
             ("Period", f"{(summary['end'] - summary['start']).days + 1} days")]
    if "mean_speed" in summary:
        items.append(("Mean speed", f"{summary['mean_speed']:.0f} km/h"))
    if "pct_heavy" in summary:
        items.append(("Heavy congestion", f"{summary['pct_heavy']:.1f}% of time"))
    metric_row(items)

    tabs = st.tabs(["Key findings", "Time patterns", "Segments", "Speed–flow",
                    "Incidents", "Distributions"])
    with tabs[0], safe_section("Key findings"):
        _findings_tab(panel, m, weekend_days)
    with tabs[1], safe_section("Time patterns"):
        _time_tab(panel, m, weekend_days)
    with tabs[2], safe_section("Segment analysis"):
        _segments_tab(panel)
    with tabs[3], safe_section("Speed–flow analysis"):
        _speed_flow_tab(panel)
    with tabs[4]:
        _incidents_tab(panel, m, weekend_days)
    with tabs[5], safe_section("Distributions"):
        _distributions_tab(df, panel, m)
