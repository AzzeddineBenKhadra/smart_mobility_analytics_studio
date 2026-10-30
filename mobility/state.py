"""Streamlit session state: dataset, undo history, cleaning log, caching."""

import streamlit as st

from .config import MAX_HISTORY, MAX_HISTORY_MB
from .logging_setup import get_logger

log = get_logger("mobility.state")


def init_state():
    defaults = {
        "original_df": None,
        "df": None,
        "df_version": 0,         # bumped every time the dataset changes
        "dataset_id": 0,         # bumped every time a new file is loaded
        "history": [],           # undo snapshots: {"action","data","log_len"}
        "cleaning_log": [],      # list of dicts (one consistent format)
        "uploaded_name": None,
        "current_upload_id": None,
        "last_seen_upload": None,
        "forecast": {},          # forecasting results for the current dataset
        "flash": None,
        "_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_df(new_df):
    """Replace the working dataset and invalidate cached computations."""
    st.session_state.df = new_df
    st.session_state.df_version += 1
    st.session_state["_cache"] = {}


def cached(name, compute, *key_parts):
    """Per-dataset-version cache (cleared by ``set_df``). No hashing needed."""
    store = st.session_state["_cache"]
    key = (name, key_parts)
    if key not in store:
        store[key] = compute()
    return store[key]


def set_flash(level, text):
    st.session_state.flash = (level, text)


def show_flash():
    flash = st.session_state.get("flash")
    if flash:
        level, text = flash
        getattr(st, level, st.info)(text)
        st.session_state.flash = None


def load_dataset(incoming, name, data_id):
    incoming = incoming.copy()
    incoming.columns = [str(c).strip() for c in incoming.columns]

    st.session_state.original_df = incoming.copy(deep=True)
    st.session_state.history = []
    st.session_state.cleaning_log = []
    st.session_state.forecast = {}
    st.session_state.uploaded_name = name
    st.session_state.current_upload_id = data_id
    st.session_state.dataset_id += 1
    set_df(incoming.copy(deep=True))
    log.info("Loaded dataset %s: %s rows x %s columns", name, len(incoming),
             len(incoming.columns))


# ---------- undo history & cleaning log (one format) ----------

def push_history(action):
    """Save the current dataset before a change.

    Snapshots are stored by reference: the app never mutates the current
    DataFrame in place (every operation works on a copy), so this is safe and
    avoids a deep copy per step.
    """
    df = st.session_state.df
    if df is None:
        return

    history = st.session_state.history
    history.append({
        "action": action,
        "data": df,
        "log_len": len(st.session_state.cleaning_log),
    })
    history = history[-MAX_HISTORY:]

    def total_mb(items):
        return sum(i["data"].memory_usage(deep=False).sum() for i in items) / 1e6

    while len(history) > 1 and total_mb(history) > MAX_HISTORY_MB:
        history.pop(0)
    st.session_state.history = history


def undo_last():
    """Restore the previous snapshot and trim the log to match."""
    if not st.session_state.history:
        return None
    snapshot = st.session_state.history.pop()
    set_df(snapshot["data"])
    del st.session_state.cleaning_log[snapshot["log_len"]:]
    st.session_state.forecast = {}
    return snapshot["action"]


def reset_to_original():
    set_df(st.session_state.original_df.copy(deep=True))
    st.session_state.history = []
    st.session_state.cleaning_log = []
    st.session_state.forecast = {}
