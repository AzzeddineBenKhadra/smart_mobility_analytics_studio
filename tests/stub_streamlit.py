"""Stub harness: runs every view with fake streamlit/plotly/folium to catch Python errors."""
import sys, types, contextlib, traceback
from unittest.mock import MagicMock
import numpy as np, pandas as pd

class StopExc(BaseException): pass
class RerunExc(BaseException): pass

class SS(dict):
    __getattr__ = lambda s, k: s[k] if k in s else (_ for _ in ()).throw(AttributeError(k))
    def __setattr__(s, k, v): s[k] = v
    def __delattr__(s, k): del s[k]

class Fake:
    def __init__(self, root=None):
        self.root = root or self
    # context manager
    def __enter__(self): return self
    def __exit__(self, *a): return False
    # widgets
    def _ov(self, key, label, default):
        ov = self.root.overrides
        if key in ov: return ov[key]
        if label in ov: return ov[label]
        return default
    def selectbox(self, label, options=(), index=0, key=None, format_func=None, **k):
        options = list(options)
        v = self._ov(key, label, None)
        if v is not None: return v
        return options[index] if options and index is not None else None
    def radio(self, label, options=(), index=0, key=None, **k):
        options = list(options); v = self._ov(key, label, None)
        return v if v is not None else (options[index] if options else None)
    def multiselect(self, label, options=(), default=None, key=None, **k):
        v = self._ov(key, label, None)
        return v if v is not None else list(default or [])
    def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **k):
        v = self._ov(key, label, None)
        return v if v is not None else (value if value is not None else min_value)
    def number_input(self, label, min_value=None, max_value=None, value=0, key=None, **k):
        return self._ov(key, label, value)
    def text_input(self, label, value="", key=None, **k): return self._ov(key, label, value)
    def checkbox(self, label, value=False, key=None, **k): return self._ov(key, label, value)
    def button(self, label, key=None, **k):
        self.root.buttons_seen.append(key or label)
        return bool(self._ov(key, label, False))
    def download_button(self, *a, **k): return False
    def file_uploader(self, label, key=None, **k): return self._ov(key, label, None)
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [Fake(self.root) for _ in range(n)]
    def tabs(self, labels): return [Fake(self.root) for _ in labels]
    def expander(self, *a, **k): return Fake(self.root)
    def container(self, *a, **k): return Fake(self.root)
    def spinner(self, *a, **k): return Fake(self.root)
    def progress(self, *a, **k): return Fake(self.root)
    def empty(self, *a, **k): return None
    def dataframe(self, data, **k):
        assert isinstance(data, (pd.DataFrame, pd.Series)) or hasattr(data, "data"), type(data)
        self.root.counts["dataframe"] += 1
    def plotly_chart(self, fig, **k): self.root.counts["plot"] += 1
    def metric(self, *a, **k): self.root.counts["metric"] += 1
    def error(self, msg, *a, **k): self.root.errors.append(str(msg))
    def warning(self, msg, *a, **k): self.root.warnings.append(str(msg))
    def stop(self): raise StopExc()
    def rerun(self): raise RerunExc()
    def cache_data(self, *a, **k):
        if a and callable(a[0]): return a[0]
        return lambda f: f
    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        return lambda *a, **k: None

class Root(Fake):
    def __init__(self):
        self.root = self
        self.reset()
        self.sidebar = Fake(self)
    def reset(self, overrides=None):
        self.overrides = overrides or {}
        self.errors, self.warnings, self.buttons_seen = [], [], []
        self.counts = {"dataframe": 0, "plot": 0, "metric": 0}

st = Root(); st.session_state = SS()

class FakeFig:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n):
        if n == "to_html": return lambda *a, **k: "<div>fig</div>"
        return lambda *a, **k: self
def fake_module(name, **attrs):
    mod = types.ModuleType(name); mod.__dict__.update(attrs); sys.modules[name] = mod; return mod

fake_module("streamlit", **{k: getattr(st, k) for k in []})
sys.modules["streamlit"] = st  # attribute access goes to Root
px = types.ModuleType("plotly.express"); px.__getattr__ = lambda n: (lambda *a, **k: FakeFig())
go = types.ModuleType("plotly.graph_objects"); go.__getattr__ = lambda n: (lambda *a, **k: FakeFig())
plotly = types.ModuleType("plotly"); plotly.express = px; plotly.graph_objects = go
sys.modules.update({"plotly": plotly, "plotly.express": px, "plotly.graph_objects": go})
for n in ["folium", "folium.plugins", "branca", "branca.colormap", "streamlit_folium"]:
    sys.modules[n] = MagicMock()

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
from mobility.state import init_state, load_dataset
from mobility.detection import detect_columns
from mobility.samples import generate_sample_traffic, generate_sample_trajectories
from mobility.config import WEEKEND_PRESETS

PAGES = ["overview", "quality", "explorer", "cleaning", "insights", "trips", "maps", "forecast", "export"]

def run(kind, frame, overrides=None):
    st.session_state.clear(); st.reset(overrides)
    init_state(); load_dataset(frame, f"{kind}.csv", ("sample", kind))
    df = st.session_state.df
    m = detect_columns(df); m["speed_unit"] = "km/h"; m["weekend_days"] = list(WEEKEND_PRESETS[list(WEEKEND_PRESETS)[0]])
    fails = 0
    for p in PAGES:
        mod = importlib.import_module(f"views.{p}")
        st.reset(overrides)
        try:
            mod.render(df, m)
            status = "ok"
        except StopExc: status = "stop"
        except Exception:
            fails += 1; status = "FAIL"; traceback.print_exc()
        print(f"[{kind}] {p:10s} {status:5s} df={st.counts['dataframe']} plots={st.counts['plot']} errs={st.errors[:2]}")
    return fails

