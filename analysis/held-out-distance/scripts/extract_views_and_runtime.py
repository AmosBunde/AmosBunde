"""Per-view results from the novel-view study and per-cell results from the runtime study.

Set STUDIES to the directory that holds the two source repositories, then:

    STUDIES=/path/to/studies uv run --with pyarrow python scripts/extract_views_and_runtime.py
"""
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
STUDIES = Path(os.environ.get("STUDIES", HERE.parent))
VIEWS = STUDIES / "Novel-View_3D-reconstruction-evaluation" / "results" / "runs"
RUNTIME = STUDIES / "Tensor-Workload_runtime-simulator" / "results" / "predictions.csv"


def views() -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(f) for f in sorted(VIEWS.glob("*.parquet"))], ignore_index=True)
    if "colour" in d:
        d = d[d.colour.fillna("view") == "view"]
    d = d[(d.role == "test") & (d.n_train == 96) & (d.poses == "true")]
    return d[["run_id", "scene", "design", "seed", "view", "angle_nearest", "psnr", "depth_mae"]].round(5)


def runtime() -> pd.DataFrame:
    d = pd.read_csv(RUNTIME)
    d = d[d.design.isin(["random", "size"])]
    d.columns = [c if not c.endswith(".1") else c[:-2] + "_pred" for c in d.columns]
    return d


if __name__ == "__main__":
    views().to_csv(HERE / "data" / "view_tests.csv.gz", index=False)
    runtime().to_csv(HERE / "data" / "runtime_cells.csv.gz", index=False)
