"""Data Quality page (profiling + approved cleaning actions)."""

import pandas as pd
import streamlit as st

from mobility.quality import column_profile, fill_candidates, profile_data, run_smart_cleaning
from mobility.state import cached, push_history, set_df, set_flash
from mobility.ui import metric_row, safe_section, show_df
from .common import mapping_key


def render(df, m):
    st.header("Data Quality")
    st.caption(
        "Profiles the dataset, flags likely problems and lets you choose which "
        "cleaning actions to apply. Nothing changes until you approve.")

    with st.spinner("Profiling dataset..."):
        findings = cached("profile", lambda: profile_data(df, m),
                          mapping_key(m, "speed_unit"))

    score = findings["quality_score"]
    label = "Good" if score >= 90 else "Needs attention" if score >= 70 else "Poor"
    metric_row([
        ("Data quality score", f"{score}/100", label),
        ("Rows", f"{findings['rows']:,}"),
        ("Columns", f"{findings['columns']:,}"),
        ("Missing cells", f"{findings['missing_cells']:,}",
         f"{findings['missing_pct']:.1f}% of all cells"),
    ])
    st.caption("The score is a heuristic for prioritising checks, not a certified "
               "data-quality standard. Statistical outliers barely affect it because "
               "rush-hour extremes are usually real.")

    impossible = findings["impossible_speed"] + findings["impossible_volume"]
    outliers = findings["outlier_speed"] + findings["outlier_volume"]

    with st.expander("🩺 Findings", expanded=True):
        issues = [
            ("Missing cells", findings["missing_cells"],
             "Review affected columns and decide whether to fill them."),
            ("Duplicate rows", findings["duplicate_rows"],
             "Remove exact duplicates if they are accidental."),
            ("Invalid coordinate pairs", findings["invalid_coordinates"],
             "Out of range, unparseable, half-missing, or exactly (0, 0)."),
            ("Unparseable timestamps", findings["invalid_timestamps"],
             "Check date formats and timezone assumptions."),
            ("Impossible speed values", findings["impossible_speed"],
             "Negative, or above 250 km/h — almost certainly sensor errors."),
            ("Impossible volume values", findings["impossible_volume"],
             "Negative counts."),
            ("Statistical outliers (speed + volume)", outliers,
             "IQR rule. Often legitimate (rush-hour peaks, incidents) — review first."),
        ]
        any_issue = False
        for name, count, advice in issues:
            if count > 0:
                any_issue = True
                st.warning(f"**{name}: {count:,}** — {advice}")
        if not any_issue:
            st.success("No issues found by the current checks "
                       "(this does not guarantee the data is error-free).")
        if not (m.get("latitude") and m.get("longitude")):
            st.info("Coordinate checks need both latitude and longitude mapped.")
        if not m.get("timestamp"):
            st.info("Timestamp checks need a timestamp column mapped.")

    with st.expander("📋 Column-level profile"):
        show_df(cached("column_profile", lambda: column_profile(df)), hide_index=True)

    st.subheader("🧹 Recommended cleaning actions")
    eligible, skipped = fill_candidates(df, m)
    has_coords = bool(m.get("latitude") and m.get("longitude"))

    actions = {
        "duplicates": st.checkbox(
            f"Remove exact duplicate rows ({findings['duplicate_rows']:,})",
            disabled=findings["duplicate_rows"] == 0, key="q_dups"),
        "coordinates": st.checkbox(
            f"Set invalid latitude/longitude pairs to missing "
            f"({findings['invalid_coordinates']:,})",
            disabled=not has_coords or findings["invalid_coordinates"] == 0,
            key="q_coords"),
        "impossible": st.checkbox(
            f"Set impossible speed/volume values to missing ({impossible:,})",
            disabled=impossible == 0, key="q_impossible"),
        "missing": st.checkbox(
            f"Fill missing values in {len(eligible)} eligible column(s) "
            "(numeric: median; text: most frequent value)",
            disabled=not eligible, key="q_missing"),
        "outliers": st.checkbox(
            f"Also set statistical (IQR) outliers to missing ({outliers:,}) — "
            "use with care",
            disabled=outliers == 0, key="q_outliers"),
    }

    if skipped:
        with st.expander("Columns skipped by 'fill missing values'"):
            show_df(pd.DataFrame(skipped, columns=["Column", "Reason"]), hide_index=True)

    selected = [name for name, on in actions.items() if on]

    if st.button("✅ Approve and apply selected actions", type="primary",
                 disabled=not selected, key="q_apply"):
        with safe_section("Cleaning"):
            cleaned, entries, changes = run_smart_cleaning(df, actions, m)
            if changes == 0:
                st.info("No data changes were necessary.")
            else:
                push_history("Smart cleaning: " + ", ".join(selected))
                st.session_state.cleaning_log.extend(entries)
                set_df(cleaned)
                set_flash("success", "Applied:\n" + "\n".join(
                    f"- {e['details']}" for e in entries))
                st.rerun()

    if len(df) > 100_000:
        st.info("Large dataset: results are cached per dataset version. For very "
                "large files use Parquet (or DuckDB for CSV) to load faster.")
