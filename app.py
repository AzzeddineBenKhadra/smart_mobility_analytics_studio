"""
Smart Mobility Analytics Studio — V3 (fixed & cleaned)

Run with:  streamlit run app.py

Main changes vs V2:
  * st.set_page_config() is now the first Streamlit call (V2 crashed on start).
  * One undo/history format and one cleaning-log format everywhere.
  * One column-detection system + a sidebar "Column mapping" override
    that every page shares.
  * Heavy computations are cached per dataset version.
  * Map: sampling, clustering, heatmap, working colour option.
  * Prediction: chronological split, baseline comparison, feature influence,
    results that survive reruns (e.g. clicking a download button).
"""

import html
import json
import re
import warnings
import zipfile
from io import BytesIO

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import HeatMap, MarkerCluster
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium

# =========================================================
# PAGE CONFIGURATION  (must be the first Streamlit command)
# =========================================================

st.set_page_config(
    page_title="Smart Mobility Analytics Studio",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-aware styling: no hard-coded text/background colours, so metric
# cards stay readable in both light and dark mode.
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# CONSTANTS
# =========================================================

MAX_HISTORY = 10            # undo snapshots kept
MAX_HISTORY_MB = 400        # approximate memory cap for undo snapshots
MAX_PLOT_POINTS = 50_000    # points sent to the browser for scatter/line/hist
MAX_TRAIN_ROWS = 200_000    # training rows used for model fitting
MAX_EXCEL_ROWS = 1_048_575

FIELD_LABELS = {
    "timestamp": "Timestamp / date",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "speed": "Speed",
    "volume": "Traffic volume / count",
    "road_id": "Road / segment ID",
}

# Matching tiers per field: exact compact name -> substring -> name token.
FIELD_RULES = {
    "timestamp": {
        "exact": {"timestamp", "datetime", "dateandtime", "eventtime",
                  "recordedat", "measurementtime", "datetimeutc",
                  "timestamputc", "date", "time"},
        "contains": ("timestamp", "datetime"),
        "tokens": {"date", "time"},
        "exclude": set(),
    },
    "latitude": {
        "exact": {"latitude", "lat", "gpslatitude", "ycoordinate", "coordy",
                  "pointlatitude", "ycoord"},
        "contains": ("latitude",),
        "tokens": {"lat", "latitude"},
        "exclude": set(),
    },
    "longitude": {
        "exact": {"longitude", "lon", "lng", "long", "gpslongitude",
                  "xcoordinate", "coordx", "pointlongitude", "xcoord"},
        "contains": ("longitude",),
        "tokens": {"lon", "lng", "long", "longitude"},
        "exclude": set(),
    },
    "speed": {
        "exact": {"speed", "speedkmh", "speedkph", "speedmph", "vehiclespeed",
                  "avgspeed", "averagespeed", "velocity"},
        "contains": ("speed", "velocity"),
        "tokens": {"speed", "velocity"},
        "exclude": {"limit"},
    },
    "volume": {
        "exact": {"trafficvolume", "volume", "flow", "trafficflow",
                  "vehiclecount", "vehicles", "vehiclevolume", "flowrate",
                  "countvehicles", "trafficcount"},
        "contains": ("trafficvolume", "vehiclecount", "trafficcount",
                     "trafficflow", "vehiclevolume", "flowrate"),
        "tokens": {"volume", "vehicles", "flow"},
        "exclude": set(),
    },
    "road_id": {
        "exact": {"roadid", "roadidentifier", "roadcode", "segmentid",
                  "linkid", "streetid", "highwayid", "roadname",
                  "streetname", "segment", "road", "link"},
        "contains": ("roadid", "segmentid", "linkid", "roadname",
                     "streetname", "streetid"),
        "tokens": set(),
        "exclude": set(),
    },
}

ID_TOKENS = {"id", "uid", "uuid", "idx", "index", "key", "code"}


# =========================================================
# SESSION STATE
# =========================================================

def init_state():
    defaults = {
        "original_df": None,
        "df": None,
        "df_version": 0,        # bumped every time the dataset changes
        "dataset_id": 0,        # bumped every time a new file is loaded
        "history": [],          # undo snapshots: {"action","data","log_len"}
        "cleaning_log": [],     # list of dicts (one consistent format)
        "uploaded_name": None,
        "current_upload_id": None,
        "last_model": None,
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
    """Per-dataset-version cache (cleared by set_df). Cheap, no hashing."""
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
    st.session_state.last_model = None
    st.session_state.uploaded_name = name
    st.session_state.current_upload_id = data_id
    st.session_state.dataset_id += 1
    set_df(incoming.copy(deep=True))


# ---------- undo history & cleaning log (single format) ----------

def push_history(action):
    """Save the current dataset before a change.

    Snapshots are stored by reference: the app never mutates
    st.session_state.df in place (every operation works on a copy), so a
    reference is safe and avoids a full deep copy on each step.
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
        return sum(
            i["data"].memory_usage(deep=False).sum() for i in items
        ) / 1e6

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
    return snapshot["action"]


def reset_to_original():
    set_df(st.session_state.original_df.copy(deep=True))
    st.session_state.history = []
    st.session_state.cleaning_log = []


def make_log_entry(operation, column, rows_before, df_after, details=""):
    return {
        "operation": operation,
        "column": column,
        "details": details,
        "rows_before": int(rows_before),
        "rows_after": int(len(df_after)),
        "missing_after": int(df_after.isna().sum().sum()),
    }


# =========================================================
# GENERAL HELPERS
# =========================================================

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


def is_numeric_col(series):
    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    )


def is_text_col(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ) and not pd.api.types.is_numeric_dtype(series)


def numeric_columns(df):
    return [c for c in df.columns if is_numeric_col(df[c])]


def sample_for_plot(data, limit=MAX_PLOT_POINTS):
    if len(data) > limit:
        return data.sample(limit, random_state=42).sort_index(), True
    return data, False


def name_tokens(name):
    """'roadID' / 'road_id' / 'Road ID' -> ['road', 'id']."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def compact_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_datetime(series):
    """Robust datetime parsing. Returns naive datetimes (UTC if tz-aware).

    * numeric epoch seconds / milliseconds are recognised
    * mixed date formats are handled (pandas 2 infers one format from the
      first value, which silently turns the rest into NaT)
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dt, "tz", None) is not None:
            return series.dt.tz_convert("UTC").dt.tz_localize(None)
        return series

    if is_numeric_col(series):
        median = series.dropna().abs().median()
        if pd.isna(median) or median < 1e8:
            # Small numbers are not timestamps.
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        unit = "ms" if median > 1e11 else "s"
        return pd.to_datetime(series, unit=unit, errors="coerce")

    def convert(**kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(series, errors="coerce", utc=True, **kwargs)

    parsed = convert()
    present = int(series.notna().sum())

    if present and (present - int(parsed.notna().sum())) / present > 0.05:
        try:
            alternative = convert(format="mixed")
            if alternative.notna().sum() > parsed.notna().sum():
                parsed = alternative
        except (TypeError, ValueError):
            pass

    return parsed.dt.tz_localize(None)


# =========================================================
# COLUMN DETECTION (single system)
# =========================================================

def _numeric_ratio(series, sample=200):
    values = series.dropna().head(sample)
    if values.empty:
        return 0.0
    return float(pd.to_numeric(values, errors="coerce").notna().mean())


def _datetime_ratio(series, sample=50):
    if pd.api.types.is_datetime64_any_dtype(series):
        return 1.0
    values = series.dropna().head(sample)
    if values.empty:
        return 0.0
    return float(parse_datetime(values).notna().mean())


def _plausible(series, field):
    if field == "timestamp":
        return _datetime_ratio(series) >= 0.8
    if field in ("latitude", "longitude"):
        return _numeric_ratio(series) >= 0.8
    if field in ("speed", "volume"):
        return _numeric_ratio(series) >= 0.5
    return True


def detect_columns(df):
    """Map transport fields to columns using names, then sanity-check values."""
    columns = list(df.columns)
    compact = {c: compact_name(c) for c in columns}
    tokens = {c: set(name_tokens(c)) for c in columns}

    detected = {}
    used = set()

    for field, rule in FIELD_RULES.items():
        chosen = None

        for tier in ("exact", "contains", "tokens"):
            hits = []
            for col in columns:
                if col in used or tokens[col] & rule["exclude"]:
                    continue

                if tier == "exact":
                    matched = compact[col] in rule["exact"]
                elif tier == "contains":
                    matched = any(k in compact[col] for k in rule["contains"])
                else:
                    matched = bool(tokens[col] & rule["tokens"])

                if matched and _plausible(df[col], field):
                    hits.append(col)

            if hits:
                # Prefer the dtype we expect for this field.
                if field == "timestamp":
                    hits.sort(key=lambda c: not pd.api.types.is_datetime64_any_dtype(df[c]))
                elif field != "road_id":
                    hits.sort(key=lambda c: not is_numeric_col(df[c]))
                chosen = hits[0]
                break

        detected[field] = chosen
        if chosen:
            used.add(chosen)

    return detected


def column_mapping_ui(df):
    """Sidebar widget: auto-detected mapping the user can override."""
    auto = cached("auto_detect", lambda: detect_columns(df))
    options = ["None"] + list(df.columns)
    mapping = {}

    with st.sidebar.expander("🔎 Column mapping", expanded=False):
        st.caption(
            "Detected from column names and values. "
            "Change any field that is wrong; all pages use these choices."
        )
        for field, label in FIELD_LABELS.items():
            default = auto.get(field)
            index = options.index(default) if default in options else 0
            choice = st.selectbox(
                label,
                options,
                index=index,
                key=f"map_{field}_{st.session_state.dataset_id}",
            )
            mapping[field] = None if choice == "None" else choice

    return mapping


def looks_like_identifier(series, name):
    if set(name_tokens(name)) & ID_TOKENS:
        return True
    if (
        pd.api.types.is_integer_dtype(series)
        and len(series) >= 50
        and series.nunique() / len(series) > 0.95
    ):
        return True
    return False


def identifier_columns(df, mapping):
    """Columns that look like IDs (never filled, never used as features)."""
    exclude = {mapping.get("speed"), mapping.get("volume")}

    def compute():
        return {
            c for c in df.columns
            if c not in exclude and looks_like_identifier(df[c], c)
        }

    return cached(
        "identifier_columns", compute, mapping.get("speed"), mapping.get("volume")
    )


# =========================================================
# DATA-QUALITY CHECKS (shared by profiling AND cleaning)
# =========================================================

def coordinate_invalid_mask(df, lat_col, lon_col):
    """Rows where a coordinate is present but the pair is not usable.

    Invalid = unparseable, out of range, only one of the two present,
    or exactly (0, 0) which is almost always a GPS error.
    """
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    present = df[lat_col].notna() | df[lon_col].notna()
    ok = (
        lat.between(-90, 90)
        & lon.between(-180, 180)
        & ~((lat == 0) & (lon == 0))
    )
    return present & ~ok


def iqr_outlier_mask(values):
    valid = values.dropna()
    if len(valid) < 8:
        return pd.Series(False, index=values.index)

    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=values.index)

    return (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)


def unusual_mask(series, kind):
    """IQR outliers plus hard-impossible values for speed / volume."""
    values = pd.to_numeric(series, errors="coerce")
    mask = iqr_outlier_mask(values)

    if kind == "speed":
        # 250 is a screening threshold, not a universal limit.
        mask = mask | (values < 0) | (values > 250)
    else:
        mask = mask | (values < 0)

    return mask.fillna(False)


def column_profile(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(t) for t in df.dtypes],
        "Missing": df.isna().sum().to_numpy(),
        "Missing %": (df.isna().mean() * 100).round(2).to_numpy(),
        "Unique": df.nunique(dropna=True).to_numpy(),
    })


def profile_data(df, mapping):
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    findings = {
        "rows": rows,
        "columns": cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_cells / max(rows * cols, 1) * 100,
        "duplicate_rows": duplicates,
        "invalid_coordinates": 0,
        "invalid_timestamps": 0,
        "unusual_speed": 0,
        "unusual_volume": 0,
    }

    lat_col, lon_col = mapping.get("latitude"), mapping.get("longitude")
    if lat_col and lon_col:
        findings["invalid_coordinates"] = int(
            coordinate_invalid_mask(df, lat_col, lon_col).sum()
        )

    ts_col = mapping.get("timestamp")
    if ts_col:
        present = df[ts_col].notna()
        if present.any():
            parsed = parse_datetime(df.loc[present, ts_col])
            findings["invalid_timestamps"] = int(parsed.isna().sum())

    if mapping.get("speed"):
        findings["unusual_speed"] = int(
            unusual_mask(df[mapping["speed"]], "speed").sum()
        )
    if mapping.get("volume"):
        findings["unusual_volume"] = int(
            unusual_mask(df[mapping["volume"]], "volume").sum()
        )

    denom = max(rows, 1)
    penalty = (
        min(findings["missing_pct"] * 0.35, 35)
        + min(duplicates / denom * 100 * 0.20, 20)
        + min(findings["invalid_coordinates"] / denom * 100 * 0.20, 20)
        + min(findings["invalid_timestamps"] / denom * 100 * 0.10, 10)
        + min(
            (findings["unusual_speed"] + findings["unusual_volume"])
            / denom * 100 * 0.15,
            15,
        )
    )
    findings["quality_score"] = round(max(0, 100 - penalty))
    return findings


def fill_candidates(df, mapping):
    """Split columns with missing values into fillable vs skipped (+reason)."""
    protected = {
        mapping.get(k) for k in ("latitude", "longitude", "timestamp", "road_id")
    } - {None}
    ids = identifier_columns(df, mapping)

    eligible, skipped = [], []
    for col in df.columns:
        share = df[col].isna().mean()
        if share == 0:
            continue
        if col in protected:
            skipped.append((col, "mapped coordinate/time/road field"))
        elif col in ids:
            skipped.append((col, "looks like an identifier"))
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            skipped.append((col, "datetime column"))
        elif share > 0.5:
            skipped.append((col, "more than 50% missing"))
        else:
            eligible.append(col)
    return eligible, skipped


def run_smart_cleaning(df, actions, mapping):
    """Apply the approved smart-cleaning actions to a copy of df.

    Returns (cleaned_df, log_entries, total_changes).
    """
    work = df.copy(deep=True)
    entries = []
    changes = 0

    if actions.get("duplicates"):
        before = len(work)
        work = work.drop_duplicates()
        removed = before - len(work)
        changes += removed
        entries.append(make_log_entry(
            "Smart: remove duplicate rows", "(all)", before, work,
            f"Removed {removed:,} exact duplicate rows"))

    if actions.get("coordinates"):
        lat_col, lon_col = mapping["latitude"], mapping["longitude"]
        bad = coordinate_invalid_mask(work, lat_col, lon_col)
        count = int(bad.sum())
        if count:
            work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce").mask(bad)
            work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce").mask(bad)
        changes += count
        entries.append(make_log_entry(
            "Smart: invalid coordinates", f"{lat_col}, {lon_col}", len(work), work,
            f"Set {count:,} invalid coordinate pairs to missing"))

    if actions.get("missing"):
        eligible, _ = fill_candidates(work, mapping)
        filled = []
        for col in eligible:
            n_missing = int(work[col].isna().sum())
            if is_numeric_col(work[col]):
                value = work[col].median()
            else:
                modes = work[col].mode(dropna=True)
                value = modes.iloc[0] if not modes.empty else np.nan

            if pd.notna(value):
                work[col] = work[col].fillna(value)
                filled.append(f"{col} ({n_missing:,})")
                changes += n_missing

        entries.append(make_log_entry(
            "Smart: fill missing values", "multiple", len(work), work,
            "Filled: " + (", ".join(filled) if filled else "nothing eligible")))

    if actions.get("outliers"):
        changed = []
        for kind in ("speed", "volume"):
            col = mapping.get(kind)
            if not col:
                continue
            mask = unusual_mask(work[col], kind)
            count = int(mask.sum())
            if count:
                work[col] = pd.to_numeric(work[col], errors="coerce").mask(mask)
                changed.append(f"{col}: {count:,}")
                changes += count

        entries.append(make_log_entry(
            "Smart: unusual measurements", "speed / volume", len(work), work,
            "Set to missing — " + ("; ".join(changed) if changed else "none found")))

    return work, entries, changes


# =========================================================
# FILE LOADING
# =========================================================

@st.cache_data(show_spinner=False, max_entries=4)
def list_zip_members(_file_bytes, file_key):
    with zipfile.ZipFile(BytesIO(_file_bytes)) as archive:
        return [
            name for name in archive.namelist()
            if not name.endswith("/")
            and name.lower().endswith((".csv", ".xlsx"))
            and not name.startswith("__MACOSX/")
            and not name.split("/")[-1].startswith(("~$", "._"))
        ]


def extract_bytes(file_bytes, filename, member):
    if filename.lower().endswith(".zip") and member:
        with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
            return archive.read(member)
    return file_bytes


@st.cache_data(show_spinner=False, max_entries=4)
def list_excel_sheets(_file_bytes, file_key, member):
    data = extract_bytes(_file_bytes, file_key[0], member)
    return pd.ExcelFile(BytesIO(data), engine="openpyxl").sheet_names


def read_csv_bytes(data):
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(BytesIO(data), encoding=encoding, low_memory=False)
            if frame.shape[1] == 1:
                # Probably a ';' or tab separated file.
                try:
                    guessed = pd.read_csv(
                        BytesIO(data), encoding=encoding, sep=None, engine="python"
                    )
                    if guessed.shape[1] > 1:
                        frame = guessed
                except Exception:
                    pass
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError("The file encoding could not be read.")


def read_dataset(file_bytes, filename, member, sheet):
    data = extract_bytes(file_bytes, filename, member)
    inner = (member or filename).lower()
    if inner.endswith(".csv"):
        return read_csv_bytes(data)
    return pd.read_excel(BytesIO(data), sheet_name=sheet, engine="openpyxl")


# =========================================================
# PAGES
# =========================================================

def page_overview(df, m):
    st.header("Dashboard Overview")

    missing_total = cached("missing_total", lambda: int(df.isna().sum().sum()))
    duplicates = cached("duplicates", lambda: int(df.duplicated().sum()))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Total columns", f"{len(df.columns):,}")
    c3.metric("Missing cells", f"{missing_total:,}")
    c4.metric("Duplicate rows", f"{duplicates:,}")

    st.subheader("Mapped transportation fields")
    show_df(
        pd.DataFrame({
            "Field": list(FIELD_LABELS.values()),
            "Column": [m[f] or "Not mapped" for f in FIELD_LABELS],
        }),
        hide_index=True,
    )
    st.caption("Wrong or missing? Change it in the sidebar under **Column mapping**.")

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
            title="Share of missing values by column (%)"))


def page_intelligence(df, m):
    st.header("Smart Data Intelligence")
    st.caption(
        "Profiles the dataset, flags likely problems and lets you choose "
        "which cleaning actions to apply. Nothing changes until you approve."
    )

    mapping_key = tuple(m.items())
    with st.spinner("Profiling dataset..."):
        findings = cached("profile", lambda: profile_data(df, m), mapping_key)

    score = findings["quality_score"]
    label = "Good" if score >= 90 else "Needs attention" if score >= 70 else "Poor"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data quality score", f"{score}/100", label)
    c2.metric("Rows", f"{findings['rows']:,}")
    c3.metric("Columns", f"{findings['columns']:,}")
    c4.metric(
        "Missing cells",
        f"{findings['missing_cells']:,}",
        f"{findings['missing_pct']:.1f}% of all cells",
    )
    st.caption(
        "The score is a heuristic for prioritising checks, "
        "not a certified data-quality standard."
    )

    with st.expander("🩺 Data-quality findings", expanded=True):
        issues = [
            ("Missing cells", findings["missing_cells"],
             "Review affected columns and decide whether to fill them."),
            ("Duplicate rows", findings["duplicate_rows"],
             "Remove exact duplicates if they are accidental."),
            ("Invalid coordinate pairs", findings["invalid_coordinates"],
             "Out of range, unparseable, half-missing, or exactly (0, 0)."),
            ("Unparseable timestamps", findings["invalid_timestamps"],
             "Check date formats and timezone assumptions."),
            ("Unusual speed values", findings["unusual_speed"],
             "Check units, sensor errors and legitimate extremes."),
            ("Unusual traffic-volume values", findings["unusual_volume"],
             "Check negative counts, measurement intervals and peaks."),
        ]
        any_issue = False
        for name, count, advice in issues:
            if count > 0:
                any_issue = True
                st.warning(f"**{name}: {count:,}** — {advice}")
        if not any_issue:
            st.success(
                "No issues found by the current checks "
                "(this does not guarantee the data is error-free)."
            )

        if not (m["latitude"] and m["longitude"]):
            st.info("Coordinate checks need both latitude and longitude mapped.")
        if not m["timestamp"]:
            st.info("Timestamp checks need a timestamp column mapped.")

    with st.expander("📋 Column-level profile"):
        show_df(cached("column_profile", lambda: column_profile(df)), hide_index=True)

    st.subheader("🧹 Recommended cleaning actions")

    eligible, skipped = fill_candidates(df, m)
    has_coords = bool(m["latitude"] and m["longitude"])

    actions = {
        "duplicates": st.checkbox(
            f"Remove exact duplicate rows ({findings['duplicate_rows']:,})",
            disabled=findings["duplicate_rows"] == 0,
            key="sd_dups",
        ),
        "coordinates": st.checkbox(
            f"Set invalid latitude/longitude pairs to missing "
            f"({findings['invalid_coordinates']:,})",
            disabled=not has_coords or findings["invalid_coordinates"] == 0,
            key="sd_coords",
        ),
        "missing": st.checkbox(
            f"Fill missing values in {len(eligible)} eligible column(s) "
            "(numeric: median; text: most frequent value)",
            disabled=not eligible,
            key="sd_missing",
        ),
        "outliers": st.checkbox(
            "Set unusual speed / traffic-volume values to missing "
            f"({findings['unusual_speed'] + findings['unusual_volume']:,})",
            disabled=(findings["unusual_speed"] + findings["unusual_volume"]) == 0,
            key="sd_outliers",
        ),
    }

    if skipped:
        with st.expander("Columns skipped by 'fill missing values'"):
            show_df(
                pd.DataFrame(skipped, columns=["Column", "Reason"]),
                hide_index=True,
            )
    st.caption(
        "Outliers are found with statistical rules and may be real "
        "observations (e.g. rush-hour peaks). Review before approving."
    )

    selected = [name for name, on in actions.items() if on]

    if st.button(
        "✅ Approve and apply selected actions",
        type="primary",
        disabled=not selected,
        key="sd_apply",
    ):
        try:
            cleaned, entries, changes = run_smart_cleaning(df, actions, m)
        except Exception as exc:
            st.error(f"Cleaning could not be completed: {exc}")
            return

        if changes == 0:
            st.info("No data changes were necessary.")
        else:
            push_history("Smart cleaning: " + ", ".join(selected))
            st.session_state.cleaning_log.extend(entries)
            set_df(cleaned)
            set_flash(
                "success",
                "Applied:\n" + "\n".join(f"- {e['details']}" for e in entries),
            )
            st.rerun()

    if len(df) > 100_000:
        st.info(
            "Large dataset: results are cached per dataset version. For very "
            "large files consider Parquet, chunked reading or a database."
        )


def page_explorer(df, m):
    st.header("Data Explorer")

    st.subheader("Preview")
    n_rows = st.number_input(
        "Rows to preview", min_value=10, max_value=5000, value=100, step=10)
    show_df(df.head(int(n_rows)))
    st.caption(f"Showing up to {int(n_rows):,} of {len(df):,} rows.")

    st.subheader("Column statistics")
    column = st.selectbox("Select a column", list(df.columns))
    series = df[column]

    a, b, c = st.columns(3)
    a.metric("Data type", str(series.dtype))
    b.metric("Missing values", f"{int(series.isna().sum()):,}")
    c.metric("Unique values", f"{int(series.nunique(dropna=True)):,}")

    if is_numeric_col(series):
        show_df(series.describe().to_frame("Statistics"))
        plot_data, sampled = sample_for_plot(series.dropna().to_frame())
        show_plot(px.histogram(plot_data, x=column, title=f"Distribution of {column}"))
        if sampled:
            st.caption(f"Histogram uses a random sample of {MAX_PLOT_POINTS:,} values.")
    else:
        counts = (
            series.astype("string").fillna("(Missing)")
            .value_counts().head(30)
            .rename_axis("Value").reset_index(name="Count")
        )
        show_df(counts, hide_index=True)
        if not counts.empty:
            show_plot(px.bar(counts, x="Value", y="Count", title=f"Top values: {column}"))

    st.subheader("Custom chart")
    numbers = numeric_columns(df)
    if not numbers:
        st.info("No numeric columns are available for charting.")
        return

    chart_type = st.selectbox(
        "Chart type",
        ["Scatter", "Line", "Bar (mean of Y by X)", "Box", "Histogram"],
    )
    x_col = st.selectbox("X-axis", list(df.columns), key="explorer_x")
    y_col = None
    if chart_type != "Histogram":
        y_col = st.selectbox("Numeric Y-axis", numbers, key="explorer_y")

    try:
        cols = [x_col] + ([y_col] if y_col and y_col != x_col else [])
        data = df[cols].dropna()

        if chart_type == "Bar (mean of Y by X)":
            grouped = data.groupby(x_col)[y_col].mean().reset_index()
            if len(grouped) > 100:
                st.warning("More than 100 groups — showing the first 100.")
                grouped = grouped.head(100)
            fig = px.bar(grouped, x=x_col, y=y_col)
            sampled = False
        else:
            data, sampled = sample_for_plot(data)
            if chart_type == "Scatter":
                fig = px.scatter(data, x=x_col, y=y_col)
            elif chart_type == "Line":
                fig = px.line(data.sort_values(x_col), x=x_col, y=y_col)
            elif chart_type == "Box":
                fig = px.box(data, x=x_col if data[x_col].nunique() <= 30 else None, y=y_col)
            else:
                fig = px.histogram(data, x=x_col)

        show_plot(fig, key="explorer_custom_chart")
        if sampled:
            st.caption(f"Showing a random sample of {MAX_PLOT_POINTS:,} points.")
    except Exception as exc:
        st.warning(f"Could not create this chart: {exc}")


OPERATIONS = [
    "Remove duplicate rows",
    "Drop rows with missing values",
    "Remove rows with missing values in a selected column",
    "Fill missing values in one column",
    "Convert column to numeric",
    "Convert column to datetime",
    "Remove rows outside a numeric range",
    "Trim whitespace from text",
    "Standardize text to lowercase",
    "Remove outliers using IQR",
    "Clip a numeric column to a valid range",
]


def parse_bound(text):
    text = text.strip()
    if not text:
        return None, None
    try:
        return float(text), None
    except ValueError:
        return None, f"'{text}' is not a number."


def apply_operation(df, operation, col, opts):
    """Run one cleaning operation on a copy. Returns (new_df, details)."""
    work = df.copy(deep=True)
    details = ""

    if operation == "Remove duplicate rows":
        work = work.drop_duplicates()
        details = f"Removed {len(df) - len(work):,} duplicate rows"

    elif operation == "Drop rows with missing values":
        work = work.dropna()
        details = f"Removed {len(df) - len(work):,} rows"

    elif operation == "Remove rows with missing values in a selected column":
        work = work.dropna(subset=[col])
        details = f"Removed {len(df) - len(work):,} rows"

    elif operation == "Fill missing values in one column":
        method = opts["fill_method"]
        series = work[col]
        missing_before = int(series.isna().sum())

        if method in ("Median", "Mean", "Zero"):
            if not is_numeric_col(series):
                raise ValueError(
                    f"{method} fill needs a numeric column. Convert '{col}' to "
                    "numeric first, or use Mode / Forward fill / Backward fill."
                )
            value = {"Median": series.median(), "Mean": series.mean(), "Zero": 0}[method]
            if pd.isna(value):
                raise ValueError("The column has no valid values to compute a fill value.")
            work[col] = series.fillna(value)
        elif method == "Mode":
            modes = series.mode(dropna=True)
            if modes.empty:
                raise ValueError("The column has no values to compute a mode.")
            work[col] = series.fillna(modes.iloc[0])
        elif method == "Forward fill":
            work[col] = series.ffill()
        else:
            work[col] = series.bfill()

        filled = missing_before - int(work[col].isna().sum())
        details = f"Filled {filled:,} of {missing_before:,} missing values ({method.lower()})"

    elif operation == "Convert column to numeric":
        valid_before = int(work[col].notna().sum())
        converted = pd.to_numeric(work[col], errors="coerce")
        lost = valid_before - int(converted.notna().sum())
        work[col] = converted
        details = f"{lost:,} values could not be converted and became missing"

    elif operation == "Convert column to datetime":
        valid_before = int(work[col].notna().sum())
        converted = parse_datetime(work[col])
        lost = valid_before - int(converted.notna().sum())
        work[col] = converted
        details = f"{lost:,} values could not be parsed and became missing"

    elif operation == "Remove rows outside a numeric range":
        numeric = pd.to_numeric(work[col], errors="coerce")
        outside = pd.Series(False, index=work.index)
        if opts["lower"] is not None:
            outside |= numeric < opts["lower"]
        if opts["upper"] is not None:
            outside |= numeric > opts["upper"]
        if opts["drop_missing"]:
            outside |= numeric.isna()
        work = work.loc[~outside]
        details = f"Removed {len(df) - len(work):,} rows outside the range"

    elif operation in ("Trim whitespace from text", "Standardize text to lowercase"):
        if opts["text_scope"] == "All text columns":
            targets = [c for c in work.columns if is_text_col(work[c])]
        else:
            if not is_text_col(work[col]):
                raise ValueError("Choose a text column, or apply to all text columns.")
            targets = [col]

        func = (
            (lambda v: v.strip() if isinstance(v, str) else v)
            if operation.startswith("Trim")
            else (lambda v: v.lower() if isinstance(v, str) else v)
        )
        changed = 0
        for target in targets:
            old = work[target]
            new = old.map(func)
            changed += int((new.astype("string") != old.astype("string")).fillna(False).sum())
            work[target] = new
        details = f"Changed {changed:,} values in {len(targets)} column(s)"

    elif operation == "Remove outliers using IQR":
        numeric = pd.to_numeric(work[col], errors="coerce")
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr):
            raise ValueError("The column has no valid numeric values.")
        outside = (numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)
        work = work.loc[~outside.fillna(False)]
        details = f"Removed {len(df) - len(work):,} outlier rows (missing values kept)"

    elif operation == "Clip a numeric column to a valid range":
        numeric = pd.to_numeric(work[col], errors="coerce")
        clipped = numeric.clip(lower=opts["lower"], upper=opts["upper"])
        changed = int(((clipped != numeric) & numeric.notna()).sum())
        work[col] = clipped
        details = f"Clipped {changed:,} values"

    return work, details


def page_cleaning(df, m):
    st.header("Data Cleaning Workbench")
    st.write(
        "Each operation is applied to the current dataset. A snapshot is "
        "saved first, so you can undo."
    )

    missing_total = cached("missing_total", lambda: int(df.isna().sum().sum()))
    duplicates = cached("duplicates", lambda: int(df.duplicated().sum()))

    q1, q2, q3 = st.columns(3)
    q1.metric("Missing cells", f"{missing_total:,}")
    q2.metric("Duplicate rows", f"{duplicates:,}")
    q3.metric("Rows", f"{len(df):,}")

    operation = st.selectbox("Cleaning operation", OPERATIONS)

    needs_column = operation not in (
        "Remove duplicate rows", "Drop rows with missing values")
    col = None
    if needs_column:
        col = st.selectbox("Column", list(df.columns), key="cleaning_column")

    opts = {
        "fill_method": "Median", "lower": None, "upper": None,
        "drop_missing": False, "text_scope": "Selected column",
    }
    problems = []

    if operation == "Fill missing values in one column":
        opts["fill_method"] = st.selectbox(
            "Fill strategy",
            ["Median", "Mean", "Zero", "Mode", "Forward fill", "Backward fill"])
        if opts["fill_method"] in ("Median", "Mean", "Zero") and not is_numeric_col(df[col]):
            problems.append(
                "This column is not numeric. Use Mode / Forward / Backward fill, "
                "or convert it to numeric first.")

    if operation in ("Remove rows outside a numeric range",
                     "Clip a numeric column to a valid range"):
        st.caption("Enter only the bounds you need.")
        lower_text = st.text_input("Minimum value (optional)", "")
        upper_text = st.text_input("Maximum value (optional)", "")
        opts["lower"], err_lo = parse_bound(lower_text)
        opts["upper"], err_hi = parse_bound(upper_text)
        problems += [e for e in (err_lo, err_hi) if e]

        if not problems:
            if opts["lower"] is None and opts["upper"] is None:
                problems.append("Enter at least one bound.")
            elif (opts["lower"] is not None and opts["upper"] is not None
                  and opts["lower"] > opts["upper"]):
                problems.append("The minimum must not be larger than the maximum.")

        if operation == "Remove rows outside a numeric range":
            opts["drop_missing"] = st.checkbox(
                "Also remove rows where this column is missing / non-numeric",
                value=False)

    if operation in ("Trim whitespace from text", "Standardize text to lowercase"):
        opts["text_scope"] = st.radio(
            "Apply to", ["Selected column", "All text columns"], horizontal=True)

    for message in problems:
        st.error(message)

    if st.button("Apply cleaning operation", type="primary", disabled=bool(problems)):
        try:
            working, details = apply_operation(df, operation, col, opts)
        except Exception as exc:
            st.error(f"Cleaning operation failed: {exc}")
        else:
            if working.equals(df):
                st.info("Nothing changed — no snapshot was saved.")
            else:
                push_history(operation)
                st.session_state.cleaning_log.append(make_log_entry(
                    operation, col or "(all)", len(df), working, details))
                set_df(working)
                set_flash("success", f"{operation}: {details}. "
                                     f"Rows {len(df):,} → {len(working):,}.")
                st.rerun()

    st.divider()
    left, right = st.columns(2)

    with left:
        if st.button("↩️ Undo last operation"):
            action = undo_last()
            if action is None:
                st.warning("There is nothing to undo.")
            else:
                set_flash("success", f"Undid: {action}")
                st.rerun()

    with right:
        if st.button("🔄 Reset to original dataset"):
            reset_to_original()
            set_flash("success", "Dataset reset to the original upload.")
            st.rerun()

    st.subheader("Current data (first 100 rows)")
    show_df(df.head(100))

    st.subheader("Validation report")
    show_df(cached("column_profile", lambda: column_profile(df)), hide_index=True)


def page_transport(df, m):
    st.header("Smart Transportation Analysis")
    st.caption("Columns come from the sidebar **Column mapping**.")

    ts_col, speed_col = m["timestamp"], m["speed"]
    vol_col, road_col = m["volume"], m["road_id"]

    if not (ts_col or speed_col or vol_col):
        st.info("Map a timestamp, speed or volume column in the sidebar to see more.")

    if speed_col:
        speeds = pd.to_numeric(df[speed_col], errors="coerce").dropna()
        if not speeds.empty:
            st.subheader("Speed")
            cols = st.columns(5)
            cols[0].metric("Average", f"{speeds.mean():.2f}")
            cols[1].metric("Median", f"{speeds.median():.2f}")
            cols[2].metric("85th percentile", f"{speeds.quantile(0.85):.2f}")
            cols[3].metric("Maximum", f"{speeds.max():.2f}")
            cols[4].metric("Minimum", f"{speeds.min():.2f}")
            st.caption("Units depend on the source dataset; none are assumed.")
            plot_data, _ = sample_for_plot(speeds.to_frame(speed_col))
            show_plot(px.histogram(plot_data, x=speed_col, title="Speed distribution"))
        else:
            st.warning(f"'{speed_col}' has no numeric values.")

    if vol_col:
        volumes = pd.to_numeric(df[vol_col], errors="coerce").dropna()
        if not volumes.empty:
            st.subheader("Traffic volume")
            cols = st.columns(4)
            cols[0].metric("Average", f"{volumes.mean():,.2f}")
            cols[1].metric("Median", f"{volumes.median():,.2f}")
            cols[2].metric("Maximum", f"{volumes.max():,.2f}")
            cols[3].metric("Sum of values", f"{volumes.sum():,.2f}")
            plot_data, _ = sample_for_plot(volumes.to_frame(vol_col))
            show_plot(px.histogram(plot_data, x=vol_col, title="Traffic volume distribution"))
        else:
            st.warning(f"'{vol_col}' has no numeric values.")

    if ts_col:
        st.subheader("Time patterns")
        ts = cached("parsed_ts", lambda: parse_datetime(df[ts_col]), ts_col)
        valid = ts.notna().to_numpy()
        st.caption(
            f"{int(valid.sum()):,} of {len(ts):,} timestamps parsed "
            "(timezone-aware values are converted to UTC)."
        )
        metrics = [c for c in (speed_col, vol_col) if c]

        if valid.sum() == 0:
            st.warning("No timestamps could be parsed. Check the mapped column.")
        elif not metrics:
            st.info("Map a speed or volume column to chart it over time.")
        else:
            metric = st.selectbox("Metric", metrics)
            work = pd.DataFrame({
                "ts": ts.to_numpy()[valid],
                "value": pd.to_numeric(df[metric], errors="coerce").to_numpy()[valid],
            })
            if road_col:
                work["road"] = df[road_col].astype("string").to_numpy()[valid]
            work = work.dropna(subset=["value"])

            tab_trend, tab_hour, tab_day = st.tabs(
                ["Trend", "By hour of day", "By day of week"])

            with tab_trend:
                days = (work["ts"].max() - work["ts"].min()).total_seconds() / 86400
                options = {"15 minutes": "15min", "Hourly": "1h",
                           "6 hours": "6h", "Daily": "1D", "Weekly": "1W"}
                default = ("15 minutes" if days <= 3 else "Hourly" if days <= 30
                           else "Daily" if days <= 365 else "Weekly")
                label = st.selectbox(
                    "Aggregation interval", list(options),
                    index=list(options).index(default))
                by_road = bool(road_col) and st.checkbox(
                    "Split by road / segment (5 busiest)", value=False)

                if by_road:
                    top = work["road"].value_counts().head(5).index
                    trend = (
                        work[work["road"].isin(top)]
                        .groupby(["road", pd.Grouper(key="ts", freq=options[label])])["value"]
                        .mean().dropna().reset_index()
                    )
                    fig = px.line(trend, x="ts", y="value", color="road")
                else:
                    trend = (
                        work.set_index("ts")["value"].sort_index()
                        .resample(options[label]).mean().dropna().reset_index()
                    )
                    fig = px.line(trend, x="ts", y="value")

                fig.update_layout(
                    title=f"{label} average of {metric}",
                    xaxis_title="Time", yaxis_title=metric)
                if trend.empty:
                    st.info("No values available for this trend.")
                else:
                    show_plot(fig, key="trend_chart")

            with tab_hour:
                hourly = (work.groupby(work["ts"].dt.hour)["value"]
                          .agg(["mean", "count"]).reset_index()
                          .rename(columns={"ts": "Hour", "mean": "Average", "count": "Records"}))
                show_plot(px.bar(hourly, x="Hour", y="Average",
                                 hover_data=["Records"],
                                 title=f"Average {metric} by hour of day"),
                          key="hour_chart")

            with tab_day:
                names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                daily = (work.groupby(work["ts"].dt.dayofweek)["value"]
                         .agg(["mean", "count"]).reset_index()
                         .rename(columns={"ts": "Day", "mean": "Average", "count": "Records"}))
                daily["Day"] = daily["Day"].map(lambda i: names[int(i)])
                show_plot(px.bar(daily, x="Day", y="Average",
                                 hover_data=["Records"],
                                 title=f"Average {metric} by day of week"),
                          key="day_chart")

    st.subheader("Numeric correlations")
    ids = identifier_columns(df, m)
    numbers = [c for c in numeric_columns(df) if c not in ids]

    if len(numbers) < 2:
        st.info("At least two numeric (non-ID) columns are needed.")
    else:
        chosen = st.multiselect("Columns", numbers, default=numbers[:12])
        if len(chosen) >= 2:
            corr = df[chosen].corr(numeric_only=True)
            show_plot(px.imshow(corr, text_auto=".2f", aspect="auto",
                                title="Correlation matrix",
                                color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1),
                      key="corr_chart")


def page_map(df, m):
    st.header("Interactive Mobility Map")

    lat_col, lon_col = m["latitude"], m["longitude"]
    if not (lat_col and lon_col):
        st.info(
            "Map both latitude and longitude in the sidebar (**Column mapping**) "
            "to display a map."
        )
        return

    valid = cached(
        "valid_coords",
        lambda: (
            pd.to_numeric(df[lat_col], errors="coerce").between(-90, 90)
            & pd.to_numeric(df[lon_col], errors="coerce").between(-180, 180)
        ).to_numpy(),
        lat_col, lon_col,
    )
    st.metric("Rows with valid coordinates", f"{int(valid.sum()):,}")
    if not valid.any():
        st.warning("No valid coordinate pairs were found.")
        return

    c1, c2, c3 = st.columns(3)
    mode = c1.radio("Display", ["Points", "Clusters", "Heatmap"], horizontal=True)
    max_points = c2.slider("Max points drawn", 500, 20000, 2000, step=500)

    color_col = None
    weight_col = None
    if mode == "Points":
        color_col = c3.selectbox("Colour by (optional)", ["None"] + list(df.columns))
        color_col = None if color_col == "None" else color_col
    elif mode == "Heatmap":
        weight = c3.selectbox("Weight by (optional)", ["None"] + numeric_columns(df))
        weight_col = None if weight == "None" else weight

    map_df = df.loc[valid].copy()
    map_df[lat_col] = pd.to_numeric(map_df[lat_col], errors="coerce")
    map_df[lon_col] = pd.to_numeric(map_df[lon_col], errors="coerce")

    total_valid = len(map_df)
    if total_valid > max_points:
        map_df = map_df.sample(max_points, random_state=42)

    lats, lons = map_df[lat_col].to_numpy(), map_df[lon_col].to_numpy()

    mobility_map = folium.Map(tiles="OpenStreetMap", control_scale=True)
    if np.isclose(lats.min(), lats.max()) and np.isclose(lons.min(), lons.max()):
        mobility_map.location = [float(lats[0]), float(lons[0])]
        mobility_map.options["zoom"] = 14
    else:
        mobility_map.fit_bounds([
            [float(lats.min()), float(lons.min())],
            [float(lats.max()), float(lons.max())],
        ])

    legend_html = ""

    if mode == "Heatmap":
        points = map_df[[lat_col, lon_col]].to_numpy().tolist()
        if weight_col:
            weights = pd.to_numeric(map_df[weight_col], errors="coerce").fillna(0)
            weights = weights - weights.min()
            top = weights.max()
            weights = weights / top if top > 0 else weights + 1
            points = [p + [float(w)] for p, w in zip(points, weights)]
        HeatMap(points, radius=12, blur=15).add_to(mobility_map)
    else:
        popup_cols = list(df.columns[:8])
        if color_col and color_col not in popup_cols:
            popup_cols.append(color_col)
        records = map_df[popup_cols].to_dict("records")

        colors = ["#3388ff"] * len(map_df)
        if color_col:
            raw = map_df[color_col]
            numeric = pd.to_numeric(raw, errors="coerce")
            if is_numeric_col(raw) and numeric.notna().any():
                vmin, vmax = numeric.quantile(0.02), numeric.quantile(0.98)
                if not vmax > vmin:
                    vmin, vmax = numeric.min(), numeric.max()
                if not vmax > vmin:
                    vmax = vmin + 1
                cmap = LinearColormap(
                    ["#2c7bb6", "#ffffbf", "#d7191c"],
                    vmin=float(vmin), vmax=float(vmax), caption=color_col)
                cmap.add_to(mobility_map)
                colors = [
                    cmap(float(min(max(v, vmin), vmax))) if pd.notna(v) else "#999999"
                    for v in numeric
                ]
            else:
                labels = raw.astype("string").fillna("(Missing)")
                top = list(labels.value_counts().head(10).index)
                palette = px.colors.qualitative.Safe
                lookup = {cat: palette[i % len(palette)] for i, cat in enumerate(top)}
                colors = [lookup.get(v, "#999999") for v in labels]
                legend_items = [(c, lookup[c]) for c in top]
                if labels.nunique() > len(top):
                    legend_items.append(("Other", "#999999"))
                legend_html = " &nbsp; ".join(
                    f"<span style='color:{col}'>●</span> {html.escape(str(name))}"
                    for name, col in legend_items)

        target = MarkerCluster().add_to(mobility_map) if mode == "Clusters" else mobility_map
        for lat, lon, record, color in zip(lats, lons, records, colors):
            popup = "<br>".join(
                f"<b>{html.escape(str(k))}</b>: {html.escape(str(v))}"
                for k, v in record.items() if pd.notna(v))
            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=4, color=color, fill=True, fill_color=color,
                fill_opacity=0.75,
                popup=folium.Popup(popup or "Mobility record", max_width=300),
            ).add_to(target)

    st_folium(mobility_map, width=None, height=550,
              returned_objects=[], key="mobility_map")

    if legend_html:
        st.markdown(legend_html, unsafe_allow_html=True)

    note = f"Showing {len(map_df):,} of {total_valid:,} valid records"
    st.caption(note + (" (random sample — raise the slider to draw more)."
                       if total_valid > max_points else "."))


def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 and y_true.nunique() > 1 else None
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2) if r2 is not None and np.isfinite(r2) else None,
    }


def train_model(df, m, target_col, feature_cols, use_time, split_mode,
                test_pct, model_type, n_estimators):
    ts_col = m["timestamp"]
    chronological = split_mode.startswith("Chronological")

    X = (df[feature_cols].apply(pd.to_numeric, errors="coerce")
         if feature_cols else pd.DataFrame(index=df.index))
    y = pd.to_numeric(df[target_col], errors="coerce")

    ts = None
    if ts_col and (use_time or chronological):
        ts = parse_datetime(df[ts_col])
    if chronological and ts is None:
        raise ValueError("A chronological split needs a mapped timestamp column.")

    if use_time and ts is not None:
        X["time_hour"] = ts.dt.hour
        X["time_dayofweek"] = ts.dt.dayofweek
        X["time_month"] = ts.dt.month
        X["time_is_weekend"] = (ts.dt.dayofweek >= 5).astype(float).where(ts.notna())

    X = X.replace([np.inf, -np.inf], np.nan)

    valid = (y.notna() & np.isfinite(y)).to_numpy().copy()
    if chronological:
        valid = valid & ts.notna().to_numpy()

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    ts_valid = ts.loc[valid].reset_index(drop=True) if ts is not None else None

    if len(X) < 20:
        raise ValueError("At least 20 rows with a valid target are required.")
    if y.nunique() < 2:
        raise ValueError("The target has fewer than two distinct values.")
    if X.shape[1] == 0:
        raise ValueError("Select at least one feature (or enable time features).")

    if chronological:
        order = np.argsort(ts_valid.to_numpy(), kind="stable")
        X, y, ts_valid = X.iloc[order], y.iloc[order], ts_valid.iloc[order]
        n_test = max(1, int(round(len(X) * test_pct / 100)))
        n_train = len(X) - n_test
        if n_train < 10:
            raise ValueError("Not enough rows left for training; lower the test share.")
        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
        ts_test = ts_valid.iloc[n_train:]
    else:
        arrays = [X, y] + ([ts_valid] if ts_valid is not None else [])
        parts = train_test_split(*arrays, test_size=test_pct / 100, random_state=42)
        X_train, X_test, y_train, y_test = parts[:4]
        ts_test = parts[5] if ts_valid is not None else None

    usable = [c for c in X_train.columns if X_train[c].nunique(dropna=True) > 1]
    if not usable:
        raise ValueError("All selected features are empty or constant in the training data.")
    X_train, X_test = X_train[usable], X_test[usable]

    notes = []
    if len(X_train) > MAX_TRAIN_ROWS:
        rng = np.random.default_rng(42)
        pick = np.sort(rng.choice(len(X_train), MAX_TRAIN_ROWS, replace=False))
        X_train, y_train = X_train.iloc[pick], y_train.iloc[pick]
        notes.append(f"Training used a random sample of {MAX_TRAIN_ROWS:,} rows.")

    leaks = X_train.corrwith(y_train).abs()
    leak_cols = [c for c, v in leaks.items() if pd.notna(v) and v > 0.98]

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_type == "Linear Regression":
        steps += [("scaler", StandardScaler()), ("model", LinearRegression())]
    else:
        steps.append(("model", RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=2,
            random_state=42, n_jobs=-1)))
    model = Pipeline(steps)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline_pred = np.full(len(y_test), float(y_train.mean()))

    fitted = model.named_steps["model"]
    if model_type == "Linear Regression":
        raw = np.abs(fitted.coef_)
        label = "Relative influence (|standardised coefficient|)"
    else:
        raw = fitted.feature_importances_
        label = "Importance (Random Forest)"
    total = raw.sum()
    importance = pd.DataFrame({
        "Feature": usable,
        "Influence": raw / total if total > 0 else raw,
    }).sort_values("Influence", ascending=False)

    results = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions,
    })
    results["Residual"] = results["Actual"] - results["Predicted"]
    if ts_test is not None:
        results.insert(0, "Timestamp", ts_test.to_numpy())

    return {
        "model_name": model_type,
        "target": target_col,
        "features": usable,
        "split": "chronological" if chronological else "random",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": evaluate(y_test, predictions),
        "baseline_metrics": evaluate(y_test, baseline_pred),
        "results": results,
        "importance": importance,
        "importance_label": label,
        "possible_leakage": leak_cols,
        "notes": notes,
        "trained_on_version": int(st.session_state.df_version),
    }


def render_model_results(res):
    st.subheader("Evaluation")

    if res["trained_on_version"] != st.session_state.df_version:
        st.warning(
            "The dataset has changed since this model was trained. "
            "Retrain to reflect the current data."
        )

    for note in res["notes"]:
        st.info(note)
    if res["possible_leakage"]:
        st.warning(
            "Possible target leakage — these features are almost perfectly "
            "correlated with the target: " + ", ".join(res["possible_leakage"])
        )

    model_m, base_m = res["metrics"], res["baseline_metrics"]

    def fmt(value):
        return f"{value:.4f}" if value is not None else "n/a"

    a, b, c = st.columns(3)
    a.metric("MAE", fmt(model_m["mae"]),
             f"{base_m['mae'] - model_m['mae']:+.4f} vs baseline",
             help="Positive delta = better than the baseline.")
    b.metric("RMSE", fmt(model_m["rmse"]),
             f"{base_m['rmse'] - model_m['rmse']:+.4f} vs baseline")
    c.metric("R²", fmt(model_m["r2"]))

    show_df(pd.DataFrame(
        {"MAE": [model_m["mae"], base_m["mae"]],
         "RMSE": [model_m["rmse"], base_m["rmse"]],
         "R²": [model_m["r2"], base_m["r2"]]},
        index=[f"{res['model_name']}", "Baseline (predict training mean)"]))

    if model_m["mae"] >= base_m["mae"]:
        st.warning("The model does not beat the simple mean baseline on the test data.")

    st.caption(
        f"Split: {res['split']} — {res['training_rows']:,} training rows, "
        f"{res['test_rows']:,} test rows. Errors are in the target's units."
    )
    if res["split"] == "random":
        st.caption(
            "A random split can be over-optimistic for time-dependent data. "
            "Use the chronological split to test on later observations."
        )

    results = res["results"]
    plot_data, sampled = sample_for_plot(results)

    fig = px.scatter(plot_data, x="Actual", y="Predicted", title="Actual vs predicted")
    low = float(min(plot_data["Actual"].min(), plot_data["Predicted"].min()))
    high = float(max(plot_data["Actual"].max(), plot_data["Predicted"].max()))
    fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high,
                  line=dict(dash="dash"))
    show_plot(fig, key="pred_scatter")
    if sampled:
        st.caption(f"Plot shows a random sample of {MAX_PLOT_POINTS:,} test rows.")

    left, right = st.columns(2)
    with left:
        show_plot(px.histogram(plot_data, x="Residual", title="Residuals"),
                  key="pred_resid")
    with right:
        show_plot(px.bar(res["importance"], x="Influence", y="Feature",
                         orientation="h", title=res["importance_label"]),
                  key="pred_importance")

    st.subheader("Prediction results")
    show_df(results.head(1000))
    if len(results) > 1000:
        st.caption("Showing the first 1,000 rows; the download contains all of them.")

    st.download_button(
        "Download prediction results (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="traffic_prediction_results.csv", mime="text/csv")

    summary = {k: res[k] for k in (
        "model_name", "target", "features", "split", "training_rows",
        "test_rows", "metrics", "baseline_metrics", "possible_leakage")}
    st.download_button(
        "Download model metrics (JSON)",
        data=json.dumps(summary, indent=2),
        file_name="traffic_model_metrics.json", mime="application/json")


def page_prediction(df, m):
    st.header("Traffic Prediction — Machine Learning")
    st.write(
        "Train a baseline model that estimates a numeric traffic metric from "
        "other fields. This is not a guaranteed real-world forecast."
    )

    numbers = numeric_columns(df)
    if len(numbers) < 2 and not m["timestamp"]:
        st.warning(
            "Need at least two numeric columns (one target, one feature), "
            "or one numeric column plus a timestamp."
        )
        return
    if not numbers:
        st.warning("No numeric columns are available for a target.")
        return

    default_target = m["volume"] or m["speed"]
    target_col = st.selectbox(
        "Prediction target", numbers,
        index=numbers.index(default_target) if default_target in numbers else 0)

    ids = identifier_columns(df, m)
    candidates = [c for c in numbers if c != target_col and c not in ids]
    feature_cols = st.multiselect(
        "Model features", candidates, default=candidates[:5],
        help="ID-like columns are excluded from the list.")

    use_time = False
    if m["timestamp"]:
        use_time = st.checkbox(
            "Add hour / day-of-week / month / weekend features from the timestamp",
            value=True)

    split_options = ["Random split"]
    if m["timestamp"]:
        split_options.insert(0, "Chronological (train on earlier, test on later)")
    split_mode = st.radio("Train/test split", split_options)

    c1, c2 = st.columns(2)
    test_pct = c1.slider("Test set %", 10, 40, 20, step=5)
    model_type = c2.selectbox("Model", ["Random Forest Regressor", "Linear Regression"])
    n_estimators = 150
    if model_type == "Random Forest Regressor":
        n_estimators = c2.slider("Number of trees", 50, 300, 150, step=50)

    if st.button("Train and evaluate model", type="primary"):
        with st.spinner("Training..."):
            try:
                st.session_state.last_model = train_model(
                    df, m, target_col, feature_cols, use_time, split_mode,
                    test_pct, model_type, n_estimators)
            except Exception as exc:
                st.session_state.last_model = None
                st.error(f"Model training failed: {exc}")

    # Rendered outside the button block so results survive reruns
    # (e.g. clicking a download button).
    if st.session_state.last_model is not None:
        st.divider()
        render_model_results(st.session_state.last_model)


def page_export(df, m):
    st.header("Export & Cleaning History")

    original = st.session_state.original_df
    st.subheader("Original vs current")
    orig_missing = int(original.isna().sum().sum())
    cur_missing = cached("missing_total", lambda: int(df.isna().sum().sum()))
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}", f"{len(df) - len(original):+,} vs original")
    b.metric("Columns", f"{len(df.columns):,}")
    c.metric("Missing cells", f"{cur_missing:,}", f"{cur_missing - orig_missing:+,} vs original",
             delta_color="inverse")

    st.subheader("Download current dataset")
    csv_bytes = cached("csv_bytes", lambda: df.to_csv(index=False).encode("utf-8"))
    st.download_button("Download cleaned CSV", data=csv_bytes,
                       file_name="smart_mobility_cleaned.csv", mime="text/csv")

    if len(df) > MAX_EXCEL_ROWS:
        st.caption("Excel export is unavailable: too many rows for one worksheet.")
    else:
        if st.button("Prepare Excel file (.xlsx)"):
            with st.spinner("Building workbook..."):
                buffer = BytesIO()
                df.to_excel(buffer, index=False, engine="openpyxl")
                st.session_state["xlsx_ready"] = (
                    st.session_state.df_version, buffer.getvalue())

        ready = st.session_state.get("xlsx_ready")
        if ready and ready[0] == st.session_state.df_version:
            st.download_button(
                "Download Excel file", data=ready[1],
                file_name="smart_mobility_cleaned.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.subheader("Cleaning history")
    if st.session_state.cleaning_log:
        history_df = pd.DataFrame(st.session_state.cleaning_log)
        show_df(history_df, hide_index=True)
        st.download_button(
            "Download cleaning history (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaning_history.csv", mime="text/csv")
    else:
        st.info("No cleaning operations have been recorded in this session.")

    st.subheader("Undo snapshots")
    if st.session_state.history:
        show_df(pd.DataFrame([
            {"Operation": s["action"],
             "Snapshot rows": len(s["data"]),
             "Snapshot columns": len(s["data"].columns)}
            for s in st.session_state.history
        ]), hide_index=True)
    else:
        st.info("No undo snapshots are currently available.")

    st.caption(
        "History and snapshots live in the current Streamlit session only; "
        "they are not a permanent audit log."
    )


PAGES = {
    "Overview": page_overview,
    "Data Intelligence": page_intelligence,
    "Data Explorer": page_explorer,
    "Data Cleaning": page_cleaning,
    "Transport Analysis": page_transport,
    "Interactive Map": page_map,
    "Traffic Prediction": page_prediction,
    "Export & History": page_export,
}


# ===== MAIN =====
# =========================================================
# APP FLOW
# =========================================================

def main():
    init_state()

    st.sidebar.title("🚦 Mobility Studio")
    st.sidebar.caption("Smart Mobility Analytics Studio — V3")

    page = st.sidebar.radio("Navigation", list(PAGES))
    st.sidebar.divider()

    st.title("Smart Mobility Analytics Studio")
    st.caption("Transportation data analytics & machine learning")

    uploaded_file = st.sidebar.file_uploader(
        "Upload transportation data",
        type=["csv", "xlsx", "zip"],
        help="Upload a CSV, an Excel workbook (.xlsx) or a ZIP archive.",
    )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            file_key = (uploaded_file.name, uploaded_file.size)
            member = None

            if uploaded_file.name.lower().endswith(".zip"):
                members = list_zip_members(file_bytes, file_key)
                if not members:
                    st.error("The ZIP archive contains no supported CSV or XLSX files.")
                    st.stop()
                member = st.sidebar.selectbox(
                    "Choose a file inside the ZIP", members,
                    key=f"zip_file_{file_key}")

            inner_name = member or uploaded_file.name
            sheet = None

            if inner_name.lower().endswith(".xlsx"):
                sheets = list_excel_sheets(file_bytes, file_key, member)
                if not sheets:
                    st.error("The Excel workbook contains no worksheets.")
                    st.stop()
                sheet = st.sidebar.selectbox(
                    "Choose a worksheet", sheets,
                    key=f"sheet_{file_key}_{member}")

            data_id = (file_key[0], file_key[1], member, sheet)

            if st.session_state.current_upload_id != data_id:
                with st.spinner("Loading data..."):
                    incoming = read_dataset(file_bytes, uploaded_file.name, member, sheet)

                if incoming.empty or len(incoming.columns) == 0:
                    st.error("The selected file contains no usable data.")
                    st.stop()

                load_dataset(incoming, inner_name, data_id)
                set_flash("success", f"Loaded {len(incoming):,} rows from {inner_name}.")

        except zipfile.BadZipFile:
            st.error("The uploaded ZIP file is invalid or corrupted.")
            st.stop()
        except Exception as exc:
            st.error(f"Could not load the uploaded data: {exc}")
            st.stop()

    df = st.session_state.df

    if df is None:
        st.info("👈 Upload a CSV, XLSX or ZIP file from the sidebar to begin.")
        st.markdown("### What this app can do")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🧹 Data quality")
            st.write("Profile data, fix missing values and duplicates, and undo changes.")
        with col2:
            st.subheader("🗺️ Spatial analytics")
            st.write("Explore coordinates as points, clusters or a heatmap.")
        with col3:
            st.subheader("🤖 Machine learning")
            st.write("Train and evaluate traffic prediction models against a baseline.")
        st.stop()

    st.sidebar.markdown("### Dataset")
    st.sidebar.write(f"**File:** {st.session_state.uploaded_name}")
    st.sidebar.write(f"**Rows:** {len(df):,}")
    st.sidebar.write(f"**Columns:** {len(df.columns)}")

    mapping = column_mapping_ui(df)

    show_flash()
    PAGES[page](df, mapping)

    st.divider()
    st.caption(
        "Smart Mobility Analytics Studio V3 | "
        "Streamlit, pandas, Plotly, Folium and scikit-learn"
    )


# Streamlit executes this file as a script, so call main() directly.
main()