"""Overview page."""

import pandas as pd
import plotly.express as px
import streamlit as st

from mobility.config import FIELD_LABELS
from mobility.quality import column_profile
from mobility.state import cached
from mobility.ui import metric_row, show_df, show_plot


def _availability(m):
    has = lambda *keys: all(m.get(k) for k in keys)  # noqa: E731
    rows = [
        ("Mobility Insights (peaks, congestion, incidents)",
         has("timestamp") and (m.get("speed") or m.get("volume")),
         "timestamp + speed and/or volume"),
        ("Trips & OD — GPS trajectories",
         has("timestamp", "latitude", "longitude"),
         "timestamp + latitude + longitude (+ vehicle ID)"),
        ("Trips & OD — trip table",
         has("origin_lat", "origin_lon", "dest_lat", "dest_lon"),
         "origin and destination latitude/longitude"),
        ("Maps", has("latitude", "longitude"), "latitude + longitude"),
        ("Forecasting",
         has("timestamp") and (m.get("speed") or m.get("volume")),
         "timestamp + speed or volume"),
    ]
    return pd.DataFrame({
        "Analysis": [r[0] for r in rows],
        "Available": ["✅ yes" if r[1] else "— not with current mapping" for r in rows],
        "Needs": [r[2] for r in rows],
    })


def render(df, m):
    st.header("Dashboard Overview")

    missing_total = cached("missing_total", lambda: int(df.isna().sum().sum()))
    duplicates = cached("duplicates", lambda: int(df.duplicated().sum()))

    metric_row([
        ("Total records", f"{len(df):,}"),
        ("Total columns", f"{len(df.columns):,}"),
        ("Missing cells", f"{missing_total:,}"),
        ("Duplicate rows", f"{duplicates:,}"),
    ])

    st.subheader("What can I do with this data?")
    show_df(_availability(m), hide_index=True)

    st.subheader("Mapped transportation fields")
    show_df(pd.DataFrame({
        "Field": list(FIELD_LABELS.values()),
        "Column": [m.get(f) or "Not mapped" for f in FIELD_LABELS],
    }), hide_index=True)
    st.caption(
        f"Speed unit: **{m.get('speed_unit', 'km/h')}**. "
        "Wrong or missing fields? Change them in the sidebar under **Column mapping**.")

    st.subheader("Dataset preview")
    show_df(df.head(20))

    profile = cached("column_profile", lambda: column_profile(df))
    st.subheader("Columns")
    show_df(profile, hide_index=True)

    st.subheader("Missing values by column")
    missing = profile[profile["Missing"] > 0]
    if missing.empty:
        st.success("No missing values detected.")
    else:
        show_plot(px.bar(
            missing, x="Column", y="Missing %", hover_data=["Missing"],
            title="Share of missing values by column (%)"), key="ov_missing")
