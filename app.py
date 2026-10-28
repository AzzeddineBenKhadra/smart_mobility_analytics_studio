import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium

from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from io import StringIO
import re
import json
import zipfile
from io import BytesIO
import re




# =========================================================
# FIX: DARK MODE TEXT VISIBILITY IN WHITE METRIC CARDS
# =========================================================

st.markdown(
    """
    <style>
    /* Make text inside white metric cards readable */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        padding: 16px !important;
        border-radius: 10px !important;
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    [data-testid="stMetric"] p,
    [data-testid="stMetric"] div {
        color: #000000 !important;
    }

    /* Keep positive and negative metric indicators visible */
    [data-testid="stMetricDelta"] svg {
        fill: currentColor !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Mobility Analytics Studio V2",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 1.5rem;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e6eaf0;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "df" not in st.session_state:
    st.session_state.df = None

if "history" not in st.session_state:
    st.session_state.history = []

if "cleaning_log" not in st.session_state:
    st.session_state.cleaning_log = []

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_name(name):
    """Normalize a column name for automatic detection."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def detect_column(columns, keywords):
    """Find the first column whose name contains a keyword."""
    for col in columns:
        normalized = normalize_name(col)
        if any(keyword in normalized for keyword in keywords):
            return col
    return None


def detect_columns(df):
    """Automatically detect common transport-related columns."""
    columns = list(df.columns)

    timestamp = detect_column(
        columns,
        ["timestamp", "datetime", "date_time", "event_time",
         "recorded_at", "time_stamp"]
    )

    date = timestamp or detect_column(
        columns, ["date", "time"]
    )

    speed = detect_column(
        columns,
        ["speed", "velocity", "avg_speed", "average_speed"]
    )

    volume = detect_column(
        columns,
        ["traffic_volume", "vehicle_count", "vehicle_volume",
         "traffic_count", "volume", "flow_count", "vehicles"]
    )

    latitude = detect_column(
        columns, ["latitude", "lat"]
    )

    longitude = detect_column(
        columns, ["longitude", "lon", "lng", "long"]
    )

    target = volume or speed

    return {
        "timestamp": date,
        "speed": speed,
        "volume": volume,
        "latitude": latitude,
        "longitude": longitude,
        "target": target
    }


def save_history(action):
    """Save a snapshot before changing the current dataset."""
    if st.session_state.df is not None:
        st.session_state.history.append({
            "action": action,
            "data": st.session_state.df.copy(deep=True)
        })

        # Keep memory use manageable.
        st.session_state.history = st.session_state.history[-20:]


def apply_numeric_conversion(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def valid_coordinates(df, lat_col, lon_col):
    """Return rows with valid latitude and longitude."""
    if not lat_col or not lon_col:
        return pd.DataFrame()

    if lat_col not in df.columns or lon_col not in df.columns:
        return pd.DataFrame()

    result = df.copy()

    result[lat_col] = pd.to_numeric(result[lat_col], errors="coerce")
    result[lon_col] = pd.to_numeric(result[lon_col], errors="coerce")

    result = result.dropna(subset=[lat_col, lon_col])

    result = result[
        result[lat_col].between(-90, 90)
        & result[lon_col].between(-180, 180)
    ]

    return result


def convert_datetime(df, col):
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_numeric_column(series, method, lower=None, upper=None):
    numeric = pd.to_numeric(series, errors="coerce")

    if method == "Replace missing with median":
        median = numeric.median()
        if pd.notna(median):
            numeric = numeric.fillna(median)

    elif method == "Replace missing with mean":
        mean = numeric.mean()
        if pd.notna(mean):
            numeric = numeric.fillna(mean)

    elif method == "Replace missing with zero":
        numeric = numeric.fillna(0)

    elif method == "Remove outliers using IQR":
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1

        if pd.notna(iqr):
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            numeric = numeric.where(
                numeric.between(lower_bound, upper_bound)
            )

    elif method == "Clip to valid range":
        if lower is not None:
            numeric = numeric.clip(lower=lower)
        if upper is not None:
            numeric = numeric.clip(upper=upper)

    return numeric


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚦 Mobility Studio")
st.sidebar.caption("Smart Mobility Analytics Studio — V2")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Explorer",
        "Data Cleaning",
        "Transport Analysis",
        "Interactive Map",
        "Traffic Prediction",
        "Export & History"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Upload a CSV dataset to begin. "
    "Column detection is automatic where possible, "
    "and you can override detected columns."
)


# =========================================================
# DATA UPLOAD — CSV, XLSX AND ZIP SUPPORT
# =========================================================

st.title("Smart Mobility Analytics Studio")
st.caption("Version 2.0 | Transportation Data Analytics & Machine Learning")

uploaded_file = st.sidebar.file_uploader(
    "Upload transportation data",
    type=["csv", "xlsx", "zip"],
    help="Upload a CSV, Excel workbook (.xlsx), or ZIP archive."
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name.lower()

        selected_file = uploaded_file.name
        selected_sheet = None
        data_bytes = file_bytes

        # Handle ZIP archives containing CSV or Excel files
        if filename.endswith(".zip"):
            with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
                supported_files = [
                    name for name in archive.namelist()
                    if not name.endswith("/")
                    and name.lower().endswith((".csv", ".xlsx"))
                    and not name.startswith("__MACOSX/")
                    and not name.split("/")[-1].startswith("~$")
                    and not name.split("/")[-1].startswith("._")
                ]

                if not supported_files:
                    st.error(
                        "The ZIP archive contains no supported CSV or XLSX files."
                    )
                    st.stop()

                selected_file = st.sidebar.selectbox(
                    "Choose a file inside the ZIP",
                    options=supported_files,
                    key=f"zip_file_{uploaded_file.name}_{uploaded_file.size}"
                )

                data_bytes = archive.read(selected_file)

        # Choose an Excel worksheet when needed
        if selected_file.lower().endswith(".xlsx"):
            excel_file = pd.ExcelFile(BytesIO(data_bytes), engine="openpyxl")

            if not excel_file.sheet_names:
                st.error("The Excel workbook contains no worksheets.")
                st.stop()

            selected_sheet = st.sidebar.selectbox(
                "Choose an Excel worksheet",
                options=excel_file.sheet_names,
                key=(
                    f"sheet_{uploaded_file.name}_"
                    f"{uploaded_file.size}_{selected_file}"
                )
            )

        # Identify the selected dataset, including ZIP member and worksheet
        data_id = (
            uploaded_file.name,
            uploaded_file.size,
            selected_file,
            selected_sheet
        )

        # Load only when the selected dataset changes
        if st.session_state.get("current_upload_id") != data_id:
            if selected_file.lower().endswith(".csv"):
                incoming_df = pd.read_csv(BytesIO(data_bytes))
            else:
                incoming_df = pd.read_excel(
                    BytesIO(data_bytes),
                    sheet_name=selected_sheet,
                    engine="openpyxl"
                )

            if incoming_df.empty or len(incoming_df.columns) == 0:
                st.error("The selected file contains no usable data.")
                st.stop()

            st.session_state.original_df = incoming_df.copy(deep=True)
            st.session_state.df = incoming_df.copy(deep=True)
            st.session_state.history = []
            st.session_state.cleaning_log = []
            st.session_state.uploaded_name = selected_file
            st.session_state.current_upload_id = data_id

            st.sidebar.success(
                f"Loaded {len(incoming_df):,} rows successfully."
            )

    except zipfile.BadZipFile:
        st.error("The uploaded ZIP file is invalid or corrupted.")
        st.stop()

    except Exception as exc:
        st.error(f"Could not load the uploaded data: {exc}")
        st.stop()

df = st.session_state.df

if df is None:
    st.info("👈 Upload a CSV, XLSX, or ZIP file from the sidebar.")

    st.markdown("### What V2 can do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🧹 Data Quality")
        st.write(
            "Handle missing values, remove duplicates, "
            "and undo cleaning operations."
        )

    with col2:
        st.subheader("🗺️ Spatial Analytics")
        st.write(
            "Explore valid latitude and longitude records on a map."
        )

    with col3:
        st.subheader("🤖 Machine Learning")
        st.write(
            "Train and evaluate traffic prediction models."
        )

    st.stop()

    
# =========================================================
# PHASE 1 — SMART DATA INTELLIGENCE
# Insert before: # COMMON DATA INFORMATION
# =========================================================

def sd_normalize_name(name):
    """Normalize column names for matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def sd_detect_columns(data):
    """Identify common transport fields using names and aliases."""
    aliases = {
        "latitude": [
            "latitude", "lat", "gpslatitude", "ycoordinate",
            "coordy", "pointlatitude"
        ],
        "longitude": [
            "longitude", "lon", "lng", "long", "gpslongitude",
            "xcoordinate", "coordx", "pointlongitude"
        ],
        "timestamp": [
            "timestamp", "datetime", "dateandtime", "eventtime",
            "recordedat", "measurementtime", "datetimeutc"
        ],
        "speed": [
            "speed", "speedkmh", "speedkph", "speedmph",
            "vehiclespeed", "avgspeed", "averagespeed"
        ],
        "traffic_volume": [
            "trafficvolume", "volume", "flow", "trafficflow",
            "vehiclecount", "vehicles", "vehiclevolume",
            "flowrate", "countvehicles"
        ],
        "road_id": [
            "roadid", "roadidentifier", "roadcode", "segmentid",
            "linkid", "streetid", "highwayid", "roadname",
            "streetname", "segment", "road"
        ],
    }

    normalized = {
        column: sd_normalize_name(column)
        for column in data.columns
    }

    detected = {}

    for field, candidates in aliases.items():
        matches = [
            column for column, normalized_name in normalized.items()
            if normalized_name in candidates
        ]

        # Fallback to partial matching for common column-name variations.
        if not matches:
            for column, normalized_name in normalized.items():
                if field == "latitude" and (
                    normalized_name.endswith("latitude")
                    or normalized_name == "lat"
                ):
                    matches.append(column)
                elif field == "longitude" and (
                    normalized_name.endswith("longitude")
                    or normalized_name in ("lon", "lng", "long")
                ):
                    matches.append(column)
                elif field == "timestamp" and any(
                    word in normalized_name
                    for word in ("timestamp", "datetime")
                ):
                    matches.append(column)
                elif field == "speed" and "speed" in normalized_name:
                    matches.append(column)
                elif field == "traffic_volume" and any(
                    word in normalized_name
                    for word in ("trafficvolume", "vehiclecount", "trafficflow")
                ):
                    matches.append(column)
                elif field == "road_id" and any(
                    word in normalized_name
                    for word in ("roadid", "segmentid", "linkid", "roadname")
                ):
                    matches.append(column)

        if matches:
            detected[field] = matches[0]

    return detected


def sd_outlier_mask(series):
    """
    Find extreme numeric values using the IQR rule.
    Returns a Boolean mask aligned with the original index.
    """
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()

    if len(valid) < 8:
        return pd.Series(False, index=series.index)

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return ((values < lower) | (values > upper)).fillna(False)


def sd_profile_data(data, detected):
    """Profile the dataset and return quality findings."""
    row_count, column_count = data.shape
    total_cells = max(row_count * column_count, 1)

    missing_cells = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())

    findings = {
        "rows": row_count,
        "columns": column_count,
        "missing_cells": missing_cells,
        "missing_pct": missing_cells / total_cells * 100,
        "duplicate_rows": duplicate_rows,
        "invalid_coordinates": 0,
        "invalid_timestamps": 0,
        "unusual_speed": 0,
        "unusual_volume": 0,
        "coordinate_rows_checked": 0,
        "timestamp_rows_checked": 0,
    }

    # Validate geographic coordinates when both fields are available.
    lat_col = detected.get("latitude")
    lon_col = detected.get("longitude")

    if lat_col and lon_col:
        lat = pd.to_numeric(data[lat_col], errors="coerce")
        lon = pd.to_numeric(data[lon_col], errors="coerce")

        has_both = data[lat_col].notna() & data[lon_col].notna()
        findings["coordinate_rows_checked"] = int(has_both.sum())

        invalid = (
            has_both
            & (
                lat.isna()
                | lon.isna()
                | ~lat.between(-90, 90)
                | ~lon.between(-180, 180)
            )
        )

        # A coordinate pair that cannot be converted is invalid too.
        findings["invalid_coordinates"] = int(invalid.sum())

    # Check whether timestamp values can be parsed.
    time_col = detected.get("timestamp")

    if time_col:
        present = data[time_col].notna()
        findings["timestamp_rows_checked"] = int(present.sum())

        if present.any():
            parsed = pd.to_datetime(
                data.loc[present, time_col],
                errors="coerce",
                utc=True
            )
            findings["invalid_timestamps"] = int(parsed.isna().sum())

    # Check speed and traffic-volume columns.
    speed_col = detected.get("speed")

    if speed_col:
        speed = pd.to_numeric(data[speed_col], errors="coerce")
        speed_outliers = sd_outlier_mask(data[speed_col])

        # Negative speeds and values over 250 are suspicious.
        # The upper threshold is a screening rule, not a universal law.
        hard_invalid = (speed < 0) | (speed > 250)

        findings["unusual_speed"] = int(
            (speed_outliers | hard_invalid.fillna(False)).sum()
        )

    volume_col = detected.get("traffic_volume")

    if volume_col:
        volume = pd.to_numeric(data[volume_col], errors="coerce")
        volume_outliers = sd_outlier_mask(data[volume_col])
        hard_invalid = volume < 0

        findings["unusual_volume"] = int(
            (volume_outliers | hard_invalid.fillna(False)).sum()
        )

    # Transparent heuristic score: a screening indicator, not a standard.
    row_denominator = max(row_count, 1)
    missing_penalty = min(findings["missing_pct"] * 0.35, 35)
    duplicate_penalty = min(
        findings["duplicate_rows"] / row_denominator * 100 * 0.20, 20
    )
    coordinate_penalty = min(
        findings["invalid_coordinates"] / row_denominator * 100 * 0.20, 20
    )
    timestamp_penalty = min(
        findings["invalid_timestamps"] / row_denominator * 100 * 0.10, 10
    )
    unusual_penalty = min(
        (
            findings["unusual_speed"] + findings["unusual_volume"]
        ) / row_denominator * 100 * 0.15,
        15
    )

    findings["quality_score"] = round(max(
        0,
        100
        - missing_penalty
        - duplicate_penalty
        - coordinate_penalty
        - timestamp_penalty
        - unusual_penalty
    ))

    return findings


# ---------------------------------------------------------
# DATA INTELLIGENCE DASHBOARD
# ---------------------------------------------------------

st.header("Smart Data Intelligence")
st.caption(
    "Automatically profile transport data, identify potential issues, "
    "and choose which cleaning actions to apply."
)

sd_df = st.session_state.df

if sd_df is None or sd_df.empty:
    st.info("Upload a dataset to run Smart Data Intelligence.")
else:
    sd_detected = sd_detect_columns(sd_df)

    # Use the full dataset for counts, but a small sample for preview.
    sd_sample_size = min(len(sd_df), 5000)
    sd_sample = sd_df.head(sd_sample_size)

    with st.spinner("Profiling dataset and checking data quality..."):
        sd_findings = sd_profile_data(sd_df, sd_detected)

    sd_score = sd_findings["quality_score"]

    if sd_score >= 90:
        sd_score_label = "Good"
    elif sd_score >= 70:
        sd_score_label = "Needs attention"
    else:
        sd_score_label = "Poor"

    sd_col1, sd_col2, sd_col3, sd_col4 = st.columns(4)

    sd_col1.metric("Data quality score", f"{sd_score}/100", sd_score_label)
    sd_col2.metric("Rows", f"{len(sd_df):,}")
    sd_col3.metric("Columns", f"{len(sd_df.columns):,}")
    sd_col4.metric(
        "Missing cells",
        f"{sd_findings['missing_cells']:,}",
        f"{sd_findings['missing_pct']:.1f}% of all cells"
    )

    st.caption(
        "The score is a transparent heuristic for prioritizing data checks. "
        "It is not a certified data-quality standard."
    )

    # Field detection results.
    with st.expander("🔎 Detected transportation fields", expanded=True):
        sd_fields = [
            ("Latitude", "latitude"),
            ("Longitude", "longitude"),
            ("Timestamp", "timestamp"),
            ("Speed", "speed"),
            ("Traffic volume", "traffic_volume"),
            ("Road / segment identifier", "road_id"),
        ]

        sd_field_col1, sd_field_col2 = st.columns(2)

        for index, (label, field) in enumerate(sd_fields):
            target_col = sd_field_col1 if index % 2 == 0 else sd_field_col2

            with target_col:
                detected_column = sd_detected.get(field)

                if detected_column:
                    st.success(f"**{label}:** `{detected_column}`")
                else:
                    st.warning(f"**{label}:** Not detected")

        if sd_detected:
            st.caption(
                "Detection is based primarily on column names. "
                "Check the matches before relying on the results."
            )

    # Data quality findings.
    with st.expander("🩺 Data-quality findings", expanded=True):
        sd_issue_rows = [
            ("Missing cells", sd_findings["missing_cells"],
             "Review affected columns; choose whether to fill missing values."),
            ("Duplicate rows", sd_findings["duplicate_rows"],
             "Remove exact duplicate rows if they are accidental."),
            ("Invalid coordinate pairs", sd_findings["invalid_coordinates"],
             "Check latitude/longitude ranges and coordinate formats."),
            ("Unparseable timestamps", sd_findings["invalid_timestamps"],
             "Review date formats and timezone assumptions."),
            ("Unusual speed values", sd_findings["unusual_speed"],
             "Check units, sensor errors, and possible legitimate extremes."),
            ("Unusual traffic-volume values", sd_findings["unusual_volume"],
             "Check negative counts, measurement intervals, and extreme values."),
        ]

        sd_any_issue = False

        for issue_name, issue_count, explanation in sd_issue_rows:
            if issue_count > 0:
                sd_any_issue = True
                st.warning(
                    f"**{issue_name}: {issue_count:,}** — {explanation}"
                )

        if not sd_any_issue:
            st.success(
                "No issues were detected by the current checks. "
                "This does not guarantee the dataset is error-free."
            )

        if not sd_detected.get("latitude") or not sd_detected.get("longitude"):
            st.info(
                "Coordinate validation is unavailable until both latitude "
                "and longitude fields are identified."
            )

    # Missing-value breakdown.
    with st.expander("📋 Column-level data profile"):
        sd_profile_table = pd.DataFrame({
            "Column": sd_df.columns,
            "Data type": [str(dtype) for dtype in sd_df.dtypes],
            "Missing values": [
                int(sd_df[col].isna().sum()) for col in sd_df.columns
            ],
            "Missing %": [
                round(sd_df[col].isna().mean() * 100, 2)
                for col in sd_df.columns
            ],
            "Unique values": [
                int(sd_df[col].nunique(dropna=True))
                for col in sd_df.columns
            ],
        })

        st.dataframe(sd_profile_table, use_container_width=True)

        st.caption(
            f"Previewing the first {len(sd_sample):,} of "
            f"{len(sd_df):,} records below."
        )
        st.dataframe(sd_sample, use_container_width=True)

    # Recommend cleaning actions; changes only occur after approval.
    st.subheader("🧹 Recommended cleaning actions")
    st.write(
        "Select the actions you want to apply. Review the description "
        "before approving changes."
    )

    sd_actions = {}

    sd_actions["duplicates"] = st.checkbox(
        f"Remove exact duplicate rows ({sd_findings['duplicate_rows']:,})",
        value=False,
        disabled=sd_findings["duplicate_rows"] == 0,
        key="sd_action_duplicates"
    )

    sd_actions["coordinates"] = st.checkbox(
        "Set invalid latitude/longitude pairs to missing values",
        value=False,
        disabled=(
            not sd_detected.get("latitude")
            or not sd_detected.get("longitude")
            or sd_findings["invalid_coordinates"] == 0
        ),
        key="sd_action_coordinates"
    )

    sd_actions["missing"] = st.checkbox(
        "Fill missing values in eligible columns "
        "(numeric: median; text: most frequent value)",
        value=False,
        disabled=sd_findings["missing_cells"] == 0,
        key="sd_action_missing"
    )

    sd_actions["outliers"] = st.checkbox(
        "Set unusual speed and traffic-volume measurements to missing",
        value=False,
        disabled=(
            not sd_detected.get("speed")
            and not sd_detected.get("traffic_volume")
        ),
        key="sd_action_outliers"
    )

    st.caption(
        "Missing-value filling skips detected coordinates, timestamps, "
        "and road identifiers. Outliers are flagged using statistical "
        "rules and may be legitimate observations."
    )

    sd_selected_actions = [
        name for name, selected in sd_actions.items() if selected
    ]

    if st.button(
        "✅ Approve and apply selected cleaning actions",
        type="primary",
        disabled=len(sd_selected_actions) == 0,
        key="sd_apply_cleaning"
    ):
        # Create a working copy only after the user approves.
        sd_cleaned = st.session_state.df.copy(deep=True)
        sd_change_log = []

        try:
            # 1. Remove exact duplicate rows.
            if sd_actions["duplicates"]:
                before_rows = len(sd_cleaned)
                sd_cleaned = sd_cleaned.drop_duplicates()
                removed = before_rows - len(sd_cleaned)

                sd_change_log.append(
                    f"Removed {removed:,} exact duplicate rows."
                )

            # 2. Mark invalid coordinate pairs as missing.
            if sd_actions["coordinates"]:
                lat_col = sd_detected["latitude"]
                lon_col = sd_detected["longitude"]

                lat = pd.to_numeric(sd_cleaned[lat_col], errors="coerce")
                lon = pd.to_numeric(sd_cleaned[lon_col], errors="coerce")

                # Missing coordinates stay missing; invalid values are
                # converted to NaN, not replaced with guessed locations.
                invalid_lat = lat.notna() & ~lat.between(-90, 90)
                invalid_lon = lon.notna() & ~lon.between(-180, 180)

                sd_cleaned.loc[invalid_lat, lat_col] = np.nan
                sd_cleaned.loc[invalid_lon, lon_col] = np.nan

                sd_change_log.append(
                    "Set out-of-range latitude/longitude values to missing."
                )

            # 3. Fill eligible missing values.
            if sd_actions["missing"]:
                protected_fields = {
                    sd_detected.get("latitude"),
                    sd_detected.get("longitude"),
                    sd_detected.get("timestamp"),
                    sd_detected.get("road_id"),
                }
                protected_fields.discard(None)

                filled_columns = []

                for column in sd_cleaned.columns:
                    if column in protected_fields:
                        continue

                    missing_mask = sd_cleaned[column].isna()

                    if not missing_mask.any():
                        continue

                    if pd.api.types.is_numeric_dtype(sd_cleaned[column]):
                        median_value = sd_cleaned[column].median()

                        if pd.notna(median_value):
                            sd_cleaned[column] = sd_cleaned[column].fillna(
                                median_value
                            )
                            filled_columns.append(str(column))

                    else:
                        modes = sd_cleaned[column].mode(dropna=True)

                        if not modes.empty:
                            sd_cleaned[column] = sd_cleaned[column].fillna(
                                modes.iloc[0]
                            )
                            filled_columns.append(str(column))

                sd_change_log.append(
                    "Filled eligible columns: "
                    + (
                        ", ".join(filled_columns)
                        if filled_columns
                        else "no eligible columns required filling"
                    )
                )

            # 4. Mark suspicious speed/volume values as missing.
            if sd_actions["outliers"]:
                outlier_columns = [
                    sd_detected.get("speed"),
                    sd_detected.get("traffic_volume"),
                ]

                changed = []

                for column in outlier_columns:
                    if column is None:
                        continue

                    numeric_values = pd.to_numeric(
                        sd_cleaned[column], errors="coerce"
                    )

                    mask = sd_outlier_mask(sd_cleaned[column])

                    if column == sd_detected.get("speed"):
                        mask = mask | (
                            (numeric_values < 0) | (numeric_values > 250)
                        ).fillna(False)

                    if column == sd_detected.get("traffic_volume"):
                        mask = mask | (numeric_values < 0).fillna(False)

                    count = int(mask.sum())

                    if count:
                        sd_cleaned.loc[mask, column] = np.nan
                        changed.append(f"{column}: {count:,} values")

                sd_change_log.append(
                    "Marked unusual measurements as missing: "
                    + (
                        "; ".join(changed)
                        if changed
                        else "no unusual values found"
                    )
                )

            # Save a snapshot only when the user approves modifications.
            if not sd_cleaned.equals(st.session_state.df):
                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append(
                    st.session_state.df.copy(deep=True)
                )

                # Keep memory use bounded for undo history.
                st.session_state.history = st.session_state.history[-5:]

                st.session_state.df = sd_cleaned

                if "cleaning_log" not in st.session_state:
                    st.session_state.cleaning_log = []

                st.session_state.cleaning_log.extend(sd_change_log)

                st.success("Selected cleaning actions applied.")
                for item in sd_change_log:
                    st.write(f"- {item}")

                st.rerun()
            else:
                st.info("No data changes were necessary.")

        except Exception as exc:
            st.error(f"Cleaning could not be completed: {exc}")

    # Efficient processing note.
    if len(sd_df) > 100_000:
        st.info(
            "Large dataset detected. Profiling uses full-data quality counts, "
            "while the preview is limited to 5,000 rows. For very large files, "
            "consider Parquet, chunked CSV processing, or a database backend."
        )

# Refresh the reference for the rest of the existing app.
df = st.session_state.df



# =========================================================
# COMMON DATA INFORMATION
# =========================================================

detected = detect_columns(df)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

st.sidebar.markdown("### Dataset")
st.sidebar.write(f"**File:** {st.session_state.uploaded_name}")
st.sidebar.write(f"**Rows:** {len(df):,}")
st.sidebar.write(f"**Columns:** {len(df.columns)}")


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":
    st.header("Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Columns", len(df.columns))
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Detected Transportation Fields")

    detected_df = pd.DataFrame({
        "Field": list(detected.keys()),
        "Detected Column": [
            str(value) if value is not None else "Not detected"
            for value in detected.values()
        ]
    })

    st.dataframe(detected_df, use_container_width=True, hide_index=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Data Types")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing Values": [int(df[col].isna().sum()) for col in df.columns],
        "Unique Values": [int(df[col].nunique(dropna=True)) for col in df.columns]
    })

    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.subheader("Missing Values by Column")

    missing_df = (
        df.isna()
        .sum()
        .rename("Missing Values")
        .reset_index()
        .rename(columns={"index": "Column"})
    )

    missing_df = missing_df[missing_df["Missing Values"] > 0]

    if not missing_df.empty:
        fig = px.bar(
            missing_df,
            x="Column",
            y="Missing Values",
            title="Missing Data Overview"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values detected.")


# =========================================================
# DATA EXPLORER
# =========================================================

elif page == "Data Explorer":
    st.header("Data Explorer")

    st.subheader("Preview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Column Statistics")

    selected_column = st.selectbox(
        "Select a column",
        options=list(df.columns)
    )

    selected_series = df[selected_column]

    st.write("**Data type:**", str(selected_series.dtype))
    st.write("**Missing values:**", int(selected_series.isna().sum()))
    st.write("**Unique values:**", int(selected_series.nunique(dropna=True)))

    if pd.api.types.is_numeric_dtype(selected_series):
        st.dataframe(
            selected_series.describe().to_frame("Statistics"),
            use_container_width=True
        )

        fig = px.histogram(
            df,
            x=selected_column,
            title=f"Distribution of {selected_column}"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        value_counts = (
            selected_series.astype("string")
            .fillna("(Missing)")
            .value_counts()
            .head(30)
            .rename_axis("Value")
            .reset_index(name="Count")
        )

        st.dataframe(value_counts, use_container_width=True)

        if not value_counts.empty:
            fig = px.bar(
                value_counts,
                x="Value",
                y="Count",
                title=f"Top Values: {selected_column}"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Create a Custom Chart")

    if len(numeric_columns) >= 1:
        x_options = list(df.columns)
        x_col = st.selectbox("X-axis", x_options, key="explorer_x")
        y_col = st.selectbox(
            "Numeric Y-axis",
            numeric_columns,
            key="explorer_y"
        )

        chart_type = st.selectbox(
            "Chart type",
            ["Scatter", "Line", "Bar"]
        )

        try:
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col)
            elif chart_type == "Line":
                fig = px.line(df, x=x_col, y=y_col)
            else:
                fig = px.bar(df, x=x_col, y=y_col)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.warning(f"Could not create this chart: {exc}")
    else:
        st.info("No numeric columns are available for charting.")


# =========================================================
# DATA CLEANING
# =========================================================

elif page == "Data Cleaning":
    st.header("Data Cleaning Workbench")

    st.write(
        "Each operation is applied to the current dataset. "
        "A snapshot is saved so you can undo recent changes."
    )

    st.subheader("Current Data Quality")

    q1, q2, q3 = st.columns(3)
    q1.metric("Missing Cells", int(df.isna().sum().sum()))
    q2.metric("Duplicate Rows", int(df.duplicated().sum()))
    q3.metric("Rows", len(df))

    operation = st.selectbox(
        "Choose a cleaning operation",
        [
            "Remove duplicate rows",
            "Drop rows with missing values",
            "Fill missing values in one column",
            "Convert column to numeric",
            "Convert column to datetime",
            "Remove rows outside a numeric range",
            "Remove rows with missing values in a selected column",
            "Trim whitespace from text columns",
            "Standardize text to lowercase",
            "Remove outliers using IQR",
            "Clip a numeric column to a valid range"
        ]
    )

    selected_col = st.selectbox(
        "Select the relevant column",
        list(df.columns),
        key="cleaning_column"
    )

    numeric_col = selected_col in numeric_columns

    fill_method = "Median"
    if operation == "Fill missing values in one column":
        fill_method = st.selectbox(
            "Fill strategy",
            ["Median", "Mean", "Zero", "Mode", "Forward fill", "Backward fill"]
        )

    lower = None
    upper = None

    if operation in [
        "Remove rows outside a numeric range",
        "Clip a numeric column to a valid range"
    ]:
        st.caption("Enter only the bounds you need.")
        lower_text = st.text_input("Minimum value (optional)", "")
        upper_text = st.text_input("Maximum value (optional)", "")

        try:
            lower = float(lower_text) if lower_text.strip() else None
            upper = float(upper_text) if upper_text.strip() else None
        except ValueError:
            st.error("Bounds must be numeric.")
            lower = None
            upper = None

    if operation == "Fill missing values in one column" and fill_method in [
        "Median", "Mean", "Zero"
    ] and not numeric_col:
        st.warning(
            "The selected column is not numeric. "
            "Use Mode, Forward fill, or Backward fill, or choose another column."
        )

    if st.button("Apply Cleaning Operation", type="primary"):
        try:
            working = df.copy(deep=True)
            save_history(operation)

            if operation == "Remove duplicate rows":
                working = working.drop_duplicates()

            elif operation == "Drop rows with missing values":
                working = working.dropna()

            elif operation == "Fill missing values in one column":
                if fill_method == "Median":
                    if not pd.api.types.is_numeric_dtype(working[selected_col]):
                        raise ValueError("Median fill requires a numeric column.")
                    value = pd.to_numeric(
                        working[selected_col], errors="coerce"
                    ).median()
                    if pd.notna(value):
                        working[selected_col] = working[selected_col].fillna(value)

                elif fill_method == "Mean":
                    if not pd.api.types.is_numeric_dtype(working[selected_col]):
                        raise ValueError("Mean fill requires a numeric column.")
                    value = pd.to_numeric(
                        working[selected_col], errors="coerce"
                    ).mean()
                    if pd.notna(value):
                        working[selected_col] = working[selected_col].fillna(value)

                elif fill_method == "Zero":
                    if not pd.api.types.is_numeric_dtype(working[selected_col]):
                        raise ValueError("Zero fill requires a numeric column.")
                    working[selected_col] = working[selected_col].fillna(0)

                elif fill_method == "Mode":
                    modes = working[selected_col].mode(dropna=True)
                    if not modes.empty:
                        working[selected_col] = working[selected_col].fillna(modes.iloc[0])

                elif fill_method == "Forward fill":
                    working[selected_col] = working[selected_col].ffill()

                elif fill_method == "Backward fill":
                    working[selected_col] = working[selected_col].bfill()

            elif operation == "Convert column to numeric":
                working[selected_col] = pd.to_numeric(
                    working[selected_col], errors="coerce"
                )

            elif operation == "Convert column to datetime":
                working[selected_col] = pd.to_datetime(
                    working[selected_col], errors="coerce"
                )

            elif operation == "Remove rows outside a numeric range":
                working[selected_col] = pd.to_numeric(
                    working[selected_col], errors="coerce"
                )
                mask = working[selected_col].notna()

                if lower is not None:
                    mask &= working[selected_col] >= lower
                if upper is not None:
                    mask &= working[selected_col] <= upper

                working = working[mask]

            elif operation == "Remove rows with missing values in a selected column":
                working = working.dropna(subset=[selected_col])

            elif operation == "Trim whitespace from text columns":
                if not (
                    pd.api.types.is_object_dtype(working[selected_col])
                    or pd.api.types.is_string_dtype(working[selected_col])
                ):
                    raise ValueError("Choose a text column for this operation.")
                working[selected_col] = working[selected_col].map(
                    lambda value: value.strip() if isinstance(value, str) else value
                )

            elif operation == "Standardize text to lowercase":
                if not (
                    pd.api.types.is_object_dtype(working[selected_col])
                    or pd.api.types.is_string_dtype(working[selected_col])
                ):
                    raise ValueError("Choose a text column for this operation.")
                working[selected_col] = working[selected_col].map(
                    lambda value: value.lower() if isinstance(value, str) else value
                )

            elif operation in [
                "Remove outliers using IQR",
                "Clip a numeric column to a valid range"
            ]:
                if not pd.api.types.is_numeric_dtype(working[selected_col]):
                    working[selected_col] = pd.to_numeric(
                        working[selected_col], errors="coerce"
                    )

                if operation == "Remove outliers using IQR":
                    values = working[selected_col]
                    q1 = values.quantile(0.25)
                    q3 = values.quantile(0.75)
                    iqr = q3 - q1

                    if pd.notna(iqr):
                        lo = q1 - 1.5 * iqr
                        hi = q3 + 1.5 * iqr
                        working = working[
                            working[selected_col].isna()
                            | working[selected_col].between(lo, hi)
                        ]

                else:
                    if lower is not None:
                        working[selected_col] = working[selected_col].clip(lower=lower)
                    if upper is not None:
                        working[selected_col] = working[selected_col].clip(upper=upper)

            st.session_state.df = working
            st.session_state.cleaning_log.append({
                "operation": operation,
                "column": selected_col,
                "rows_before": len(df),
                "rows_after": len(working),
                "missing_after": int(working.isna().sum().sum())
            })

            st.success(
                f"Operation completed. Rows: {len(df):,} → {len(working):,}"
            )
            st.rerun()

        except Exception as exc:
            # Remove the snapshot if the operation failed.
            if st.session_state.history:
                st.session_state.history.pop()
            st.error(f"Cleaning operation failed: {exc}")

    st.divider()

    left, right = st.columns(2)

    with left:
        if st.button("Undo Last Operation"):
            if st.session_state.history:
                previous = st.session_state.history.pop()
                st.session_state.df = previous["data"]

                if st.session_state.cleaning_log:
                    st.session_state.cleaning_log.pop()

                st.success(f"Undid: {previous['action']}")
                st.rerun()
            else:
                st.warning("There is no operation to undo.")

    with right:
        if st.button("Reset to Original Dataset"):
            st.session_state.df = st.session_state.original_df.copy(deep=True)
            st.session_state.history = []
            st.session_state.cleaning_log = []
            st.success("Dataset reset to the original upload.")
            st.rerun()

    st.subheader("Cleaning Preview")
    st.dataframe(st.session_state.df.head(100), use_container_width=True)

    st.subheader("Validation Report")

    current = st.session_state.df
    validation = pd.DataFrame({
        "Column": current.columns,
        "Data Type": [str(current[c].dtype) for c in current.columns],
        "Missing": [int(current[c].isna().sum()) for c in current.columns],
        "Unique": [int(current[c].nunique(dropna=True)) for c in current.columns]
    })

    st.dataframe(validation, use_container_width=True, hide_index=True)


# =========================================================
# TRANSPORT ANALYSIS
# =========================================================

elif page == "Transport Analysis":
    st.header("Smart Transportation Analysis")

    st.subheader("Automatic Column Detection")

    st.dataframe(
        pd.DataFrame({
            "Field": list(detected.keys()),
            "Detected Column": [
                str(v) if v is not None else "Not detected"
                for v in detected.values()
            ]
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Choose Columns Manually")

    all_columns = list(df.columns)

    timestamp_col = st.selectbox(
        "Timestamp / date column",
        ["None"] + all_columns,
        index=(all_columns.index(detected["timestamp"]) + 1)
        if detected["timestamp"] in all_columns else 0
    )

    speed_col = st.selectbox(
        "Speed column",
        ["None"] + all_columns,
        index=(all_columns.index(detected["speed"]) + 1)
        if detected["speed"] in all_columns else 0
    )

    volume_col = st.selectbox(
        "Traffic volume / count column",
        ["None"] + all_columns,
        index=(all_columns.index(detected["volume"]) + 1)
        if detected["volume"] in all_columns else 0
    )

    if speed_col != "None":
        speed_values = pd.to_numeric(df[speed_col], errors="coerce").dropna()

        if not speed_values.empty:
            a, b, c = st.columns(3)
            a.metric("Average Speed", f"{speed_values.mean():.2f}")
            b.metric("Maximum Speed", f"{speed_values.max():.2f}")
            c.metric("Minimum Speed", f"{speed_values.min():.2f}")

            st.caption(
                "Speed units depend on the source dataset; "
                "the app does not assume km/h or mph."
            )

            fig = px.histogram(
                pd.DataFrame({speed_col: speed_values}),
                x=speed_col,
                title="Speed Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    if volume_col != "None":
        volume_values = pd.to_numeric(df[volume_col], errors="coerce").dropna()

        if not volume_values.empty:
            a, b, c = st.columns(3)
            a.metric("Average Traffic Volume", f"{volume_values.mean():,.2f}")
            b.metric("Maximum Traffic Volume", f"{volume_values.max():,.2f}")
            c.metric("Total of Values", f"{volume_values.sum():,.2f}")

            fig = px.histogram(
                pd.DataFrame({volume_col: volume_values}),
                x=volume_col,
                title="Traffic Volume Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    if timestamp_col != "None":
        temp = df.copy()
        temp[timestamp_col] = pd.to_datetime(
            temp[timestamp_col], errors="coerce"
        )
        temp = temp.dropna(subset=[timestamp_col])

        if not temp.empty:
            st.subheader("Traffic Trends Over Time")

            metric_options = []
            if speed_col != "None":
                metric_options.append(speed_col)
            if volume_col != "None" and volume_col not in metric_options:
                metric_options.append(volume_col)

            if metric_options:
                metric_col = st.selectbox(
                    "Metric to chart over time",
                    metric_options
                )

                temp[metric_col] = pd.to_numeric(
                    temp[metric_col], errors="coerce"
                )

                trend = (
                    temp.dropna(subset=[metric_col])
                    .set_index(timestamp_col)[metric_col]
                    .sort_index()
                    .resample("1h")
                    .mean()
                    .dropna()
                    .reset_index()
                )

                if not trend.empty:
                    fig = px.line(
                        trend,
                        x=timestamp_col,
                        y=metric_col,
                        title=f"Hourly Average: {metric_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid values were available for this trend.")

            st.subheader("Records by Hour of Day")
            temp["Hour"] = temp[timestamp_col].dt.hour

            hourly = temp.groupby("Hour").size().reset_index(name="Records")

            fig = px.bar(
                hourly,
                x="Hour",
                y="Records",
                title="Number of Records by Hour"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numeric Correlations")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("At least two numeric columns are needed for correlations.")


# =========================================================
# INTERACTIVE MAP
# =========================================================

elif page == "Interactive Map":
    st.header("Interactive Mobility Map")

    columns = list(df.columns)

    lat_default = detected["latitude"]
    lon_default = detected["longitude"]

    lat_col = st.selectbox(
        "Latitude column",
        ["None"] + columns,
        index=(columns.index(lat_default) + 1)
        if lat_default in columns else 0
    )

    lon_col = st.selectbox(
        "Longitude column",
        ["None"] + columns,
        index=(columns.index(lon_default) + 1)
        if lon_default in columns else 0
    )

    if lat_col == "None" or lon_col == "None":
        st.info(
            "Choose latitude and longitude columns to display a map. "
            "If your dataset has no geographic coordinates, "
            "you will need a dataset that includes them."
        )
    else:
        map_df = valid_coordinates(df, lat_col, lon_col)

        st.metric("Rows with Valid Coordinates", f"{len(map_df):,}")

        if map_df.empty:
            st.warning("No valid coordinate pairs were found.")
        else:
            map_color = st.selectbox(
                "Color points by (optional)",
                ["None"] + list(df.columns)
            )

            map_center = [
                float(map_df[lat_col].median()),
                float(map_df[lon_col].median())
            ]

            mobility_map = folium.Map(
                location=map_center,
                zoom_start=11,
                control_scale=True
            )

            # Avoid rendering too many individual markers.
            
            plotted_df = map_df

            for _, row in plotted_df.iterrows():
                popup_lines = [
                    f"{col}: {row[col]}"
                    for col in df.columns[:8]
                    if pd.notna(row[col])
                ]

                popup_text = "\n".join(popup_lines)

                folium.CircleMarker(
                    location=[
                        float(row[lat_col]),
                        float(row[lon_col])
                    ],
                    radius=4,
                    popup=folium.Popup(
                        popup_text or "Mobility record",
                        max_width=300
                    ),
                    tooltip="Mobility record",
                    fill=True
                ).add_to(mobility_map)

            st_folium(
                mobility_map,
                width=None,
                height=550,
                returned_objects=[]
            )

            st.caption(
                f"Showing all {len(plotted_df):,} valid records on the map."
            )   

            st.caption(
                "The optional color selection is available for future "
                "categorical map styling; this version displays markers "
                "with popups containing record details."
            )


# =========================================================
# TRAFFIC PREDICTION
# =========================================================

elif page == "Traffic Prediction":
    st.header("Traffic Prediction — Machine Learning")

    st.write(
        "Train a model to estimate a numeric traffic metric from other "
        "available fields. This is a baseline predictive model, not a "
        "guaranteed real-world traffic forecast."
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning(
            "At least two numeric columns are required: one target "
            "and at least one feature."
        )
        st.stop()

    default_target = detected["target"]

    target_col = st.selectbox(
        "Prediction target",
        numeric_cols,
        index=numeric_cols.index(default_target)
        if default_target in numeric_cols else 0
    )

    available_features = [
        col for col in numeric_cols if col != target_col
    ]

    feature_cols = st.multiselect(
        "Model features",
        available_features,
        default=available_features[:min(5, len(available_features))]
    )

    timestamp_feature = None

    if detected["timestamp"] and detected["timestamp"] in df.columns:
        use_time = st.checkbox(
            "Extract hour/day features from the detected timestamp",
            value=True
        )
        if use_time:
            timestamp_feature = detected["timestamp"]
    else:
        use_time = False

    test_size = st.slider(
        "Test set percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )

    model_type = st.selectbox(
        "Model",
        ["Random Forest Regressor", "Linear Regression"]
    )

    if st.button("Train and Evaluate Model", type="primary"):
        try:
            model_df = df.copy()

            if not feature_cols:
                raise ValueError("Select at least one numeric feature.")

            X = model_df[feature_cols].copy()
            y = pd.to_numeric(model_df[target_col], errors="coerce")

            if timestamp_feature:
                timestamps = pd.to_datetime(
                    model_df[timestamp_feature], errors="coerce"
                )

                X["timestamp_hour"] = timestamps.dt.hour
                X["timestamp_day_of_week"] = timestamps.dt.dayofweek
                X["timestamp_month"] = timestamps.dt.month

            # Ensure all model features are numeric.
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

            # Infinite values are invalid for model training.
            X = X.replace([np.inf, -np.inf], np.nan)

            valid_target = y.notna() & np.isfinite(y)
            X = X.loc[valid_target]
            y = y.loc[valid_target]

            if len(X) < 10:
                raise ValueError(
                    "At least 10 rows with a valid target are required. "
                    "More data is recommended."
                )

            if y.nunique() < 2:
                raise ValueError(
                    "The target has fewer than two distinct values."
                )

            if X.dropna(axis=1, how="all").shape[1] == 0:
                raise ValueError("All selected features are empty.")

            # Remove features that contain no valid data.
            X = X.dropna(axis=1, how="all")

            # Remove constant features.
            X = X.loc[:, X.nunique(dropna=True) > 1]

            if X.empty or X.shape[1] == 0:
                raise ValueError(
                    "No usable feature columns remain after validation."
                )

            # Random row split: appropriate for a basic baseline only.
            # For future forecasting, use a chronological split.
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size / 100,
                random_state=42
            )

            estimator = (
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1
                )
                if model_type == "Random Forest Regressor"
                else LinearRegression()
            )

            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator)
            ])

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions) if len(y_test) > 1 else np.nan

            st.session_state["last_model_results"] = {
                "model_name": model_type,
                "target": target_col,
                "features": list(X.columns),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2) if np.isfinite(r2) else None,
                "training_rows": int(len(X_train)),
                "test_rows": int(len(X_test))
            }

            st.subheader("Evaluation Metrics")

            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{mae:.4f}")
            m2.metric("RMSE", f"{rmse:.4f}")
            m3.metric(
                "R²",
                f"{r2:.4f}" if np.isfinite(r2) else "Unavailable"
            )

            st.caption(
                "MAE and RMSE are in the target's units. "
                "Lower errors are generally better. R² may be negative "
                "when predictions perform poorly. Metrics must be "
                "compared against a sensible baseline."
            )

            results_df = pd.DataFrame({
                "Actual": y_test.to_numpy(),
                "Predicted": predictions
            })

            fig = px.scatter(
                results_df,
                x="Actual",
                y="Predicted",
                title="Actual vs Predicted"
            )

            low = min(
                results_df["Actual"].min(),
                results_df["Predicted"].min()
            )
            high = max(
                results_df["Actual"].max(),
                results_df["Predicted"].max()
            )

            fig.add_shape(
                type="line",
                x0=low,
                y0=low,
                x1=high,
                y1=high,
                line=dict(dash="dash")
            )

            st.plotly_chart(fig, use_container_width=True)

            results_df["Residual"] = (
                results_df["Actual"] - results_df["Predicted"]
            )

            st.subheader("Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            st.download_button(
                "Download Prediction Results CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="traffic_prediction_results.csv",
                mime="text/csv"
            )

            st.download_button(
                "Download Model Metrics JSON",
                data=json.dumps(
                    st.session_state["last_model_results"],
                    indent=2
                ),
                file_name="traffic_model_metrics.json",
                mime="application/json"
            )

        except Exception as exc:
            st.error(f"Model training failed: {exc}")

    if "last_model_results" in st.session_state:
        st.divider()
        st.subheader("Last Model Evaluation")

        st.json(st.session_state["last_model_results"])

        st.warning(
            "For true future traffic forecasting, sort records by time "
            "and evaluate on later observations. A random train/test split "
            "can give overly optimistic results when records are time-dependent."
        )


# =========================================================
# EXPORT & HISTORY
# =========================================================

elif page == "Export & History":
    st.header("Export & Cleaning History")

    st.subheader("Download Current Dataset")

    st.download_button(
        "Download Cleaned CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="smart_mobility_cleaned.csv",
        mime="text/csv"
    )

    st.subheader("Cleaning History")

    if st.session_state.cleaning_log:
        history_df = pd.DataFrame(st.session_state.cleaning_log)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Cleaning History CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaning_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No cleaning operations have been recorded in this session.")

    st.subheader("Undo Snapshots")

    if st.session_state.history:
        snapshots = pd.DataFrame([
            {
                "Operation": item["action"],
                "Snapshot Rows": len(item["data"]),
                "Snapshot Columns": len(item["data"].columns)
            }
            for item in st.session_state.history
        ])
        st.dataframe(snapshots, use_container_width=True, hide_index=True)
    else:
        st.info("No undo snapshots are currently available.")

    st.caption(
        "Cleaning history and undo snapshots are stored in the current "
        "Streamlit session. They are not permanent audit logs."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Smart Mobility Analytics Studio V2 | "
    "Built with Streamlit, pandas, Plotly, Folium and scikit-learn"
)
