"""Write the synthetic sample datasets to data/ (simulated, not real traffic)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mobility.samples import generate_sample_trajectories, generate_sample_traffic  # noqa: E402

out = ROOT / "data"
out.mkdir(exist_ok=True)
generate_sample_traffic().to_csv(out / "sample_traffic_sensors.csv", index=False)
generate_sample_trajectories().to_csv(out / "sample_gps_trajectories.csv", index=False)
print("Wrote", *sorted(p.name for p in out.glob("sample_*.csv")))
