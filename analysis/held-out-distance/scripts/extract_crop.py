"""Per-field predictions from the crop-map study, with each test field's distance to its nearest training field.

Run inside the crop-map-spatial-validation repository's environment:

    cd crop-map-spatial-validation
    uv run python /path/to/this/scripts/extract_crop.py /path/to/this/data/crop_fields.csv.gz

The study stored out-of-fold predictions but not the folds. They are deterministic, so
this script rebuilds them with the study's own code and measures the distances.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from cropval.splits import block_ids, folds
from cropval.suite import load

MODEL = "tcn"
DESIGNS = [("main", "random", 0.0), ("main", "block", 0.0), ("main", "tile", 0.0),
           ("buffer", "block", 250.0), ("buffer", "block", 500.0), ("buffer", "block", 1000.0), ("buffer", "block", 2000.0)]


def main(out: str) -> None:
    fields, _, _ = load(Path("results/fields.gpkg"), Path("results/features.parquet"))
    oof = pd.read_parquet("results/suite/oof.parquet")
    oof = oof[oof.model == MODEL]
    xy = fields[["x_m", "y_m"]].to_numpy()
    block = block_ids(fields, 2000.0)
    rows = []
    for group, split, buffer_m in DESIGNS:
        for seed in sorted(oof.seed.unique()):
            part = oof[(oof.group == group) & (oof.split == split) & (oof.seed == seed) & (oof.buffer_m == buffer_m)]
            if part.empty:
                continue
            part = fields[["tile", "field_id"]].merge(part, on=["tile", "field_id"], how="left")
            assert part.pred.notna().all() and (part.y.to_numpy() == fields.label.to_numpy() - 1).all()
            dist = np.full(len(fields), np.nan)
            for train, test in folds(fields, split, seed=int(seed), block_m=2000.0, buffer_m=buffer_m):
                dist[test] = cKDTree(xy[train]).query(xy[test])[0]
            assert np.isfinite(dist).all()
            rows.append(pd.DataFrame({"design": split if not buffer_m else f"block+{int(buffer_m)}m", "seed": int(seed),
                                      "block": block, "y": part.y.astype(int).to_numpy(),
                                      "pred": part.pred.astype(int).to_numpy(), "distance_m": np.round(dist, 1)}))
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1])
