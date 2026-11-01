"""Small Streamlit helpers shared by all pages."""

from contextlib import contextmanager

import streamlit as st

from .logging_setup import format_traceback, friendly_message, get_logger

log = get_logger("mobility.ui")

# Streamlit control-flow exceptions (st.rerun / st.stop) must pass through.
_CONTROL_FLOW = {"RerunException", "StopException", "ScriptControlException"}


def show_df(data, **kwargs):
    """st.dataframe that works on old and new Streamlit versions."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except Exception:
        st.dataframe(data, use_container_width=True, **kwargs)


def show_plot(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key)


def stretch_button(label, **kwargs):
    """st.button that fills the width on old and new Streamlit versions."""
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, use_container_width=True, **kwargs)


def show_map(fmap, key, height=550):
    from streamlit_folium import st_folium
    st_folium(fmap, width=None, height=height, returned_objects=[], key=key)


@contextmanager
def safe_section(label, show_details=True):
    """Run a block; show a friendly message (and log details) on failure.

    Usage::

        with safe_section("Training the model"):
            ...
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - intentionally broad at the UI edge
        if type(exc).__name__ in _CONTROL_FLOW:
            raise
        log.exception("%s failed", label)
        st.error(friendly_message(exc, label))
        if show_details:
            with st.expander("Technical details"):
                st.code(format_traceback(exc))


def metric_row(items):
    """items: list of (label, value[, delta])."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.metric(*item)


def fmt_minutes(minutes):
    minutes = float(minutes)
    if minutes < 60:
        return f"{minutes:g} min"
    if minutes < 1440:
        return f"{minutes / 60:g} h"
    return f"{minutes / 1440:g} day" + ("s" if minutes >= 2880 else "")
