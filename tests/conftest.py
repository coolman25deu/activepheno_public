import pandas as pd
import pytest


@pytest.fixture
def rows():
    return pd.DataFrame([
        {"compound_name_raw": "compound A", "moa_id": "M1", "moa_name": "M1",
         "cell_line": "A549-WT", "plate_id": "P1", "well_id": "A2",
         "replicate_id": "P1/A2", "site_id": "1", "channel": "Ph",
         "time_hours": t, "image_path": f"frame{t}.tif"}
        for t in [4, 0, 2]
    ])
