"""Sidebar widgets: column mapping, units and weekend definition."""

import streamlit as st

from mobility.config import FIELD_LABELS, SPEED_UNITS, WEEKEND_PRESETS
from mobility.detection import detect_columns, detect_speed_unit
from mobility.state import cached


def column_mapping_ui(df):
    """Auto-detected mapping that the user can override; shared by all pages."""
    auto = cached("auto_detect", lambda: detect_columns(df))
    options = ["None"] + list(df.columns)
    dataset_id = st.session_state.dataset_id
    mapping = {}

    with st.sidebar.expander("🔎 Column mapping", expanded=False):
        st.caption("Detected from column names and values. Change any field that is "
                   "wrong; every page uses these choices.")
        for field, label in FIELD_LABELS.items():
            default = auto.get(field)
            index = options.index(default) if default in options else 0
            choice = st.selectbox(label, options, index=index,
                                  key=f"map_{field}_{dataset_id}")
            mapping[field] = None if choice == "None" else choice

        st.markdown("**Units & calendar**")
        units = list(SPEED_UNITS)
        guess = detect_speed_unit(df, auto.get("speed"))
        mapping["speed_unit"] = st.selectbox(
            "Speed unit in your data", units, index=units.index(guess),
            key=f"unit_{dataset_id}",
            help="Speeds are converted to km/h for congestion analysis.")

        presets = list(WEEKEND_PRESETS)
        choice = st.selectbox("Weekend days", presets, key=f"weekend_{dataset_id}",
                              help="Used to split weekday and weekend patterns.")
        mapping["weekend_days"] = WEEKEND_PRESETS[choice]

    return mapping
