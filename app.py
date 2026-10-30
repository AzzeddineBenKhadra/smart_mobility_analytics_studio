"""Smart Mobility Analytics Studio — run with:  streamlit run app.py"""

import streamlit as st

# set_page_config must be the first Streamlit command.
st.set_page_config(
    page_title="Smart Mobility Analytics Studio",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

import zipfile  # noqa: E402

from mobility import __version__  # noqa: E402
from mobility import dataio  # noqa: E402
from mobility.logging_setup import friendly_message, get_logger  # noqa: E402
from mobility.samples import generate_sample_trajectories, generate_sample_traffic  # noqa: E402
from mobility.state import init_state, load_dataset, set_flash, show_flash  # noqa: E402
from views import (cleaning, explorer, export, forecast, insights, maps,  # noqa: E402
                   overview, quality, trips)
from views.sidebar import column_mapping_ui  # noqa: E402

log = get_logger("mobility.app")

# Theme-aware styling: no hard-coded colours, so it works in light and dark mode.
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

PAGES = {
    "Overview": overview,
    "Data Quality": quality,
    "Data Explorer": explorer,
    "Data Cleaning": cleaning,
    "Mobility Insights": insights,
    "Trips & OD": trips,
    "Maps": maps,
    "Forecasting": forecast,
    "Report & Export": export,
}

SAMPLES = {
    "traffic": ("🚦 Sample: road sensors (8 roads, 4 weeks)", generate_sample_traffic,
                "sample_traffic_sensors.csv"),
    "gps": ("🚗 Sample: GPS trajectories (25 vehicles)", generate_sample_trajectories,
            "sample_gps_trajectories.csv"),
}


@st.cache_data(show_spinner=False, max_entries=4)
def _zip_members(_file_bytes, file_key):
    return dataio.list_zip_members(_file_bytes)


@st.cache_data(show_spinner=False, max_entries=4)
def _excel_sheets(_file_bytes, file_key, member):
    data = dataio.extract_bytes(_file_bytes, file_key[0], member)
    return dataio.list_excel_sheets(data)


def stretch_button_sidebar(label, **kwargs):
    try:
        return st.sidebar.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.sidebar.button(label, use_container_width=True, **kwargs)


def handle_samples():
    st.sidebar.markdown("**No data yet? Try a sample**")
    for key, (label, factory, filename) in SAMPLES.items():
        if stretch_button_sidebar(label, key=f"sample_{key}"):
            with st.spinner("Generating sample data..."):
                frame = factory()
            load_dataset(frame, filename, ("sample", key))
            set_flash("success", f"Loaded the synthetic sample ({len(frame):,} rows). "
                                 "It is simulated data for demonstration only.")
    st.sidebar.caption("Samples are synthetic (simulated), not real traffic measurements.")


def handle_upload():
    uploaded = st.sidebar.file_uploader(
        "Upload transportation data", type=["csv", "xlsx", "parquet", "zip"],
        help="CSV, Excel (.xlsx), Parquet, or a ZIP containing those. "
             "Parquet is the fastest for large files.")
    if uploaded is None:
        return

    try:
        file_bytes = uploaded.getvalue()
        file_key = (uploaded.name, uploaded.size)
        member = None

        if uploaded.name.lower().endswith(".zip"):
            members = _zip_members(file_bytes, file_key)
            if not members:
                st.error("The ZIP archive contains no supported CSV, XLSX or Parquet files.")
                st.stop()
            member = st.sidebar.selectbox("Choose a file inside the ZIP", members,
                                          key=f"zip_{file_key}")

        inner = member or uploaded.name
        sheet = None
        if inner.lower().endswith(".xlsx"):
            sheets = _excel_sheets(file_bytes, file_key, member)
            if not sheets:
                st.error("The Excel workbook contains no worksheets.")
                st.stop()
            sheet = st.sidebar.selectbox("Choose a worksheet", sheets,
                                         key=f"sheet_{file_key}_{member}")

        data_id = (file_key[0], file_key[1], member, sheet)
        # Only load when the *upload* changed; otherwise a sample loaded afterwards
        # would be overwritten on the next rerun.
        if st.session_state.last_seen_upload == data_id:
            return

        with st.spinner("Loading data..."):
            incoming = dataio.read_dataset(file_bytes, uploaded.name, member, sheet)
        if incoming.empty or len(incoming.columns) == 0:
            st.error("The selected file contains no usable data.")
            st.stop()

        st.session_state.last_seen_upload = data_id
        load_dataset(incoming, inner, data_id)
        set_flash("success", f"Loaded {len(incoming):,} rows from {inner}.")

    except zipfile.BadZipFile:
        st.error("The uploaded ZIP file is invalid or corrupted.")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - last line of defence at the UI edge
        if type(exc).__name__ in {"RerunException", "StopException", "ScriptControlException"}:
            raise
        log.exception("Could not load upload")
        st.error(friendly_message(exc, "Could not load the uploaded data"))
        st.stop()


def landing():
    st.info("👈 Upload a CSV, Excel, Parquet or ZIP file — or load a sample from the sidebar.")
    st.markdown("### What this app does")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🧹 Data quality")
        st.write("Profile data, fix problems with full undo, and export a clean file.")
    with c2:
        st.subheader("🚦 Mobility insights")
        st.write("Peak hours, congestion index, reliability, incidents and speed–flow curves.")
    with c3:
        st.subheader("🔮 Forecasting")
        st.write("Predict speed or volume ahead, compared honestly with simple baselines.")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.subheader("🗺️ Maps")
        st.write("Points, heatmaps, congestion by segment, hourly animation, zones, districts.")
    with c5:
        st.subheader("🚗 Trips & OD")
        st.write("Detect trips and stops in GPS traces and see origin–destination flows.")
    with c6:
        st.subheader("📄 Reports")
        st.write("Export an HTML report (print to PDF), CSV, Excel or Parquet.")


def main():
    init_state()

    st.sidebar.title("🚦 Mobility Studio")
    st.sidebar.caption(f"Smart Mobility Analytics Studio — v{__version__}")
    page = st.sidebar.radio("Navigation", list(PAGES))
    st.sidebar.divider()

    st.title("Smart Mobility Analytics Studio")
    st.caption("Transportation data analytics & machine learning")

    handle_samples()
    handle_upload()
    show_flash()

    df = st.session_state.df
    if df is None:
        landing()
        st.stop()

    st.sidebar.divider()
    st.sidebar.markdown("### Dataset")
    st.sidebar.write(f"**File:** {st.session_state.uploaded_name}")
    st.sidebar.write(f"**Rows:** {len(df):,}")
    st.sidebar.write(f"**Columns:** {len(df.columns)}")

    mapping = column_mapping_ui(df)
    PAGES[page].render(df, mapping)

    st.divider()
    st.caption(f"Smart Mobility Analytics Studio v{__version__} | "
               "Streamlit, pandas, Plotly, Folium and scikit-learn")


main()
