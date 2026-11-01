"""Data Cleaning workbench."""

import streamlit as st

from mobility.quality import OPERATIONS, apply_operation, make_log_entry, parse_bound
from mobility.state import (cached, push_history, reset_to_original, set_df,
                            set_flash, undo_last)
from mobility.quality import column_profile
from mobility.ui import metric_row, safe_section, show_df
from mobility.utils import is_numeric_col


def render(df, m):
    st.header("Data Cleaning Workbench")
    st.write("Each operation is applied to the current dataset. A snapshot is saved "
             "first, so you can undo.")

    metric_row([
        ("Missing cells", f"{cached('missing_total', lambda: int(df.isna().sum().sum())):,}"),
        ("Duplicate rows", f"{cached('duplicates', lambda: int(df.duplicated().sum())):,}"),
        ("Rows", f"{len(df):,}"),
    ])

    operation = st.selectbox("Cleaning operation", OPERATIONS)
    needs_column = operation not in ("Remove duplicate rows", "Drop rows with missing values")
    col = st.selectbox("Column", list(df.columns), key="cleaning_column") if needs_column else None

    opts = {"fill_method": "Median", "lower": None, "upper": None,
            "drop_missing": False, "text_scope": "Selected column"}
    problems = []

    if operation == "Fill missing values in one column":
        opts["fill_method"] = st.selectbox(
            "Fill strategy", ["Median", "Mean", "Zero", "Mode", "Forward fill", "Backward fill"])
        if opts["fill_method"] in ("Median", "Mean", "Zero") and not is_numeric_col(df[col]):
            problems.append("This column is not numeric. Use Mode / Forward / Backward "
                            "fill, or convert it to numeric first.")

    if operation in ("Remove rows outside a numeric range",
                     "Clip a numeric column to a valid range"):
        st.caption("Enter only the bounds you need.")
        opts["lower"], err_lo = parse_bound(st.text_input("Minimum value (optional)", ""))
        opts["upper"], err_hi = parse_bound(st.text_input("Maximum value (optional)", ""))
        problems += [e for e in (err_lo, err_hi) if e]
        if not problems:
            if opts["lower"] is None and opts["upper"] is None:
                problems.append("Enter at least one bound.")
            elif (opts["lower"] is not None and opts["upper"] is not None
                  and opts["lower"] > opts["upper"]):
                problems.append("The minimum must not be larger than the maximum.")
        if operation == "Remove rows outside a numeric range":
            opts["drop_missing"] = st.checkbox(
                "Also remove rows where this column is missing / non-numeric", value=False)

    if operation in ("Trim whitespace from text", "Standardize text to lowercase"):
        opts["text_scope"] = st.radio(
            "Apply to", ["Selected column", "All text columns"], horizontal=True)

    for message in problems:
        st.error(message)

    if st.button("Apply cleaning operation", type="primary", disabled=bool(problems)):
        with safe_section("Cleaning operation"):
            working, details = apply_operation(df, operation, col, opts)
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
