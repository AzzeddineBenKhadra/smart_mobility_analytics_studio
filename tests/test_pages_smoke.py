"""Runs every page against a stubbed Streamlit (no browser). Catches Python errors only,
not layout problems; run `streamlit run app.py` for a real visual check.

Runs in a subprocess because the stub replaces streamlit/plotly/folium in sys.modules.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_pages_render_with_stubbed_streamlit():
    code = ("import sys; sys.path.insert(0, 'tests');"
            "import stub_streamlit as s;"
            "from mobility.samples import *;"
            "f = s.run('traffic', generate_sample_traffic()) + s.run('gps', generate_sample_trajectories());"
            "sys.exit(1 if f else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
