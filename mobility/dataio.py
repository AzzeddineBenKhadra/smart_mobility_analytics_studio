"""File loading / saving: CSV, Excel, Parquet, ZIP. DuckDB & Parquet optional.

Large-file strategy
-------------------
* Parquet is read directly (columnar, typed, fast) and offered as an export.
* CSVs above ``DUCKDB_THRESHOLD_MB`` use DuckDB's multi-threaded reader when
  it is installed; otherwise pandas is used.
"""

import os
import tempfile
import zipfile
from io import BytesIO

import pandas as pd

from .config import DUCKDB_THRESHOLD_MB
from .logging_setup import get_logger

log = get_logger("mobility.io")

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".parquet")


def optional_module(name):
    try:
        return __import__(name)
    except ImportError:
        return None


def list_zip_members(file_bytes):
    with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
        return [
            name for name in archive.namelist()
            if not name.endswith("/")
            and name.lower().endswith(SUPPORTED_EXTENSIONS)
            and not name.startswith("__MACOSX/")
            and not name.split("/")[-1].startswith(("~$", "._"))
        ]


def extract_bytes(file_bytes, filename, member):
    if filename.lower().endswith(".zip") and member:
        with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
            return archive.read(member)
    return file_bytes


def list_excel_sheets(data):
    return pd.ExcelFile(BytesIO(data), engine="openpyxl").sheet_names


def _read_csv_duckdb(data):
    duckdb = optional_module("duckdb")
    if duckdb is None:
        return None

    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            handle.write(data)
            path = handle.name
        frame = duckdb.read_csv(path, sample_size=-1).df()
        log.info("Read CSV with DuckDB: %s rows", len(frame))
        return frame
    except Exception as exc:  # fall back to pandas on any DuckDB problem
        log.warning("DuckDB read failed (%s); falling back to pandas", exc)
        return None
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def read_csv_bytes(data):
    if len(data) > DUCKDB_THRESHOLD_MB * 1_000_000:
        frame = _read_csv_duckdb(data)
        if frame is not None:
            return frame

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(BytesIO(data), encoding=encoding, low_memory=False)
            if frame.shape[1] == 1:  # probably ';' or tab separated
                try:
                    guessed = pd.read_csv(
                        BytesIO(data), encoding=encoding, sep=None, engine="python")
                    if guessed.shape[1] > 1:
                        frame = guessed
                except Exception:
                    pass
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError("The file encoding could not be read (tried UTF-8 and Latin-1).")


def read_parquet_bytes(data):
    if optional_module("pyarrow") is None and optional_module("fastparquet") is None:
        raise ImportError("pyarrow (needed to read Parquet files)")
    return pd.read_parquet(BytesIO(data))


def to_parquet_bytes(df):
    if optional_module("pyarrow") is None and optional_module("fastparquet") is None:
        raise ImportError("pyarrow (needed to write Parquet files)")
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def normalize_dtypes(df):
    """Convert pyarrow-backed numeric/bool/datetime columns to plain NumPy dtypes.

    Parquet and DuckDB can return Arrow-backed columns, which do not support every
    pandas operation (e.g. cumsum on booleans). Text columns are left alone.
    """
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if not isinstance(dtype, pd.ArrowDtype):
            continue
        kind = getattr(dtype, "kind", "O")
        try:
            if kind in "iuf":
                out[col] = out[col].astype("float64") if out[col].isna().any() \
                    else out[col].astype(dtype.numpy_dtype)
            elif kind == "b":
                out[col] = out[col].astype(bool) if not out[col].isna().any() \
                    else out[col].astype(object)
            elif kind == "M":
                out[col] = pd.to_datetime(out[col].astype("datetime64[ns]"))
            else:
                out[col] = out[col].astype(object)
        except Exception as exc:  # leave the column as it was
            log.warning("Could not normalise column %s: %s", col, exc)
    return out


def read_dataset(file_bytes, filename, member=None, sheet=None):
    """Load the selected CSV / Excel / Parquet file (optionally from a ZIP)."""
    data = extract_bytes(file_bytes, filename, member)
    inner = (member or filename).lower()

    if inner.endswith(".csv"):
        return normalize_dtypes(read_csv_bytes(data))
    if inner.endswith(".parquet"):
        return normalize_dtypes(read_parquet_bytes(data))
    if inner.endswith(".xlsx"):
        return pd.read_excel(BytesIO(data), sheet_name=sheet, engine="openpyxl")
    raise ValueError(f"Unsupported file type: {inner.rsplit('.', 1)[-1]}")
