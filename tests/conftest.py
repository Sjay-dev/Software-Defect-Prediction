from __future__ import annotations

import pandas as pd
import pytest

from defect_prediction.schema import FEATURE_COLUMNS


@pytest.fixture
def sample_data() -> pd.DataFrame:
    rows = []
    for project_index, project in enumerate(("alpha", "beta", "heldout")):
        for release in ("1.0", "2.0"):
            for class_index in range(4):
                base = project_index * 10 + class_index + 1
                row = {
                    "project": project,
                    "release": release,
                    "class_name": f"{project}.R{release}.Class{class_index}",
                    "bug_count": int(class_index in (1, 3)),
                }
                row.update({name: float(base + offset) for offset, name in enumerate(FEATURE_COLUMNS)})
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[0, "wmc"] = float("nan")
    return frame

