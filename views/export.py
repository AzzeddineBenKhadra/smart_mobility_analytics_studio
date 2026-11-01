"""Report & Export page."""

from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from mobility import insights as I
from mobility.config import MAX_EXCEL_ROWS
from mobility.dataio import to_parquet_bytes
from mobility.forecasting import BASELINE_LABELS
from mobility.quality import profile_data
from mobility.report import build_report_html, sanitize_for_filename
from mobility.state import cached
from mobility.ui import metric_row, safe_section, show_df
from .common import get_panel, mapping_key


def _fig_html(fig, first):
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False)


def build_report(df, m):
    """Assemble the report from whatever the current mapping allows."""
    name = st.session_state.uploaded_name or "dataset"
    findings = cached("profile", lambda: profile_data(df, m), mapping_key(m, "speed_unit"))
    sections = [{
        "title": "Dataset",
        "metrics": [("Rows", f"{len(df):,}"), ("Columns", f"{len(df.columns):,}"),
                    ("Quality score", f"{findings['quality_score']}/100"),
                    ("Missing cells", f"{findings['missing_cells']:,}")],
        "paragraphs": [
            f"Source: {name}. Speed unit: {m.get('speed_unit', 'km/h')}.",
            "The quality score is a heuristic for prioritising checks, not a standard."],
        "tables": [("Mapped fields", pd.DataFrame({
            "Field": [k for k, v in m.items() if k in (
                "timestamp", "vehicle_id", "latitude", "longitude", "speed", "volume",
                "road_id")],
            "Column": [v or "not mapped" for k, v in m.items() if k in (
                "timestamp", "vehicle_id", "latitude", "longitude", "speed", "volume",
                "road_id")]}))],
    }]

    first_figure = True
    panel = get_panel(df, m)
    if panel is not None:
        weekend = tuple(m.get("weekend_days", (5, 6)))
        section = {"title": "Mobility insights",
                   "bullets": I.generate_findings(panel, weekend), "figures": [], "tables": []}

        metric = "ci" if "ci" in panel.columns else ("speed" if "speed" in panel.columns
                                                     else "volume")
        label = {"ci": "Congestion index", "speed": "Speed (km/h)",
                 "volume": "Volume per record"}[metric]
        profile = I.time_profile(panel, metric, weekend)
        fig = px.line(profile, x="hour", y="mean", color="day_type", markers=True,
                      labels={"hour": "Hour of day", "mean": label, "day_type": ""},
                      title=f"{label} by hour")
        section["figures"].append((f"{label}: weekday vs weekend",
                                   _fig_html(fig, first_figure)))
        first_figure = False

        heat = I.heat_table(panel, metric)
        fig = px.imshow(heat, aspect="auto", labels={"x": "Hour", "y": "", "color": label},
                        color_continuous_scale="RdYlGn_r" if metric == "ci" else "Viridis",
                        title=f"{label}: weekday × hour")
        section["figures"].append(("Weekday × hour heat map", _fig_html(fig, False)))

        ranking = I.segment_ranking(panel)
        if not ranking.empty:
            section["tables"].append((
                "Segments ranked by congestion",
                ranking.drop(columns=[c for c in ("lat", "lon") if c in ranking.columns])))
        sections.append(section)

    result = st.session_state.forecast.get("result")
    if result is not None:
        mm = result["metrics"]
        rows = [{"Method": f"{result['model_name']} (model)", "MAE": mm["mae"],
                 "RMSE": mm["rmse"], "R²": mm["r2"]}]
        for key, v in result["baseline_metrics"].items():
            rows.append({"Method": BASELINE_LABELS[key], "MAE": v["mae"],
                         "RMSE": v["rmse"], "R²": v["r2"]})
        skill = result["skill"]
        sections.append({
            "title": "Forecasting",
            "paragraphs": [
                f"Target: {result['target']}; horizon {result['horizon_min']:g} minutes; "
                f"tested on {result['comparable_rows']:,} later rows.",
                ("Skill versus the best baseline: " + (f"{skill * 100:+.1f}%" if skill is not None
                                                        else "n/a")
                 + f". Interval coverage {result['coverage_actual'] * 100:.0f}% "
                   f"(target {result['coverage_target'] * 100:.0f}%).")],
            "tables": [("Model vs baselines (same test rows)", pd.DataFrame(rows))],
        })

    log = st.session_state.cleaning_log
    if log:
        sections.append({"title": "Cleaning history",
                         "tables": [("Operations applied", pd.DataFrame(log))]})

    return build_report_html(
        "Smart Mobility Analysis", f"Dataset: {name}", sections,
        footer="Created with Smart Mobility Analytics Studio.")


def render(df, m):
    st.header("Report & Export")

    original = st.session_state.original_df
    orig_missing = int(original.isna().sum().sum())
    cur_missing = cached("missing_total", lambda: int(df.isna().sum().sum()))
    metric_row([
        ("Rows", f"{len(df):,}", f"{len(df) - len(original):+,} vs original"),
        ("Columns", f"{len(df.columns):,}"),
        ("Missing cells", f"{cur_missing:,}", f"{cur_missing - orig_missing:+,} vs original"),
    ])

    st.subheader("Analysis report (HTML)")
    st.caption("A single page with findings, charts, tables and your cleaning history. "
               "Open it in a browser and use **Print → Save as PDF** for a PDF. "
               "Charts load Plotly from a CDN, so viewing needs an internet connection.")
    if st.button("Build report", type="primary", key="build_report"):
        with safe_section("Building the report"), st.spinner("Building..."):
            html_doc = build_report(df, m)
            st.session_state["report_html"] = (st.session_state.df_version, html_doc)

    ready = st.session_state.get("report_html")
    if ready:
        if ready[0] != st.session_state.df_version:
            st.warning("The dataset changed since this report was built. Rebuild it.")
        stem = sanitize_for_filename((st.session_state.uploaded_name or "dataset").rsplit(".", 1)[0])
        st.download_button("Download report (.html)", ready[1], f"{stem}_report.html",
                           "text/html")

    st.subheader("Download the current dataset")
    c1, c2, c3 = st.columns(3)
    c1.download_button("CSV", cached("csv_bytes", lambda: df.to_csv(index=False).encode("utf-8")),
                       "smart_mobility_cleaned.csv", "text/csv")
    try:
        c2.download_button("Parquet (fast to reload)",
                           cached("parquet_bytes", lambda: to_parquet_bytes(df)),
                           "smart_mobility_cleaned.parquet", "application/octet-stream")
    except ImportError:
        c2.caption("Parquet export needs `pyarrow` (pip install pyarrow).")
    except Exception as exc:  # noqa: BLE001
        c2.caption(f"Parquet export unavailable: {exc}")

    if len(df) > MAX_EXCEL_ROWS:
        c3.caption("Excel export unavailable: too many rows for one worksheet.")
    elif c3.button("Prepare Excel (.xlsx)"):
        with st.spinner("Building workbook..."):
            buffer = BytesIO()
            df.to_excel(buffer, index=False, engine="openpyxl")
            st.session_state["xlsx_ready"] = (st.session_state.df_version, buffer.getvalue())
    ready_x = st.session_state.get("xlsx_ready")
    if ready_x and ready_x[0] == st.session_state.df_version:
        st.download_button("Download Excel file", ready_x[1], "smart_mobility_cleaned.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.subheader("Cleaning history")
    if st.session_state.cleaning_log:
        history = pd.DataFrame(st.session_state.cleaning_log)
        show_df(history, hide_index=True)
        st.download_button("Download cleaning history (CSV)",
                           history.to_csv(index=False).encode("utf-8"),
                           "cleaning_history.csv", "text/csv")
    else:
        st.info("No cleaning operations have been recorded in this session.")

    st.subheader("Undo snapshots")
    if st.session_state.history:
        show_df(pd.DataFrame([
            {"Operation": s["action"], "Snapshot rows": len(s["data"]),
             "Snapshot columns": len(s["data"].columns)}
            for s in st.session_state.history]), hide_index=True)
    else:
        st.info("No undo snapshots are currently available.")
    st.caption("History and snapshots live in the current session only; they are not a "
               "permanent audit log.")
