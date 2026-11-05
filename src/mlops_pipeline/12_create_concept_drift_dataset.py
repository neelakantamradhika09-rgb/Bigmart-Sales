from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(
    "data/raw/Train.csv"
)

OUTPUT = Path(
    "data/monitoring/drift/"
    "concept_drift_data.csv"
)

TARGET = "Item_Outlet_Sales"


df = pd.read_csv(
    SOURCE
)

rng = np.random.default_rng(42)

median_mrp = df[
    "Item_MRP"
].median()

sales_multiplier = np.where(
    df["Item_MRP"] > median_mrp,
    1.18,
    0.82
)

noise = rng.normal(
    loc=0,
    scale=max(
        df[TARGET].std() * 0.02,
        1
    ),
    size=len(df)
)

df[TARGET] = (
    df[TARGET]
    * sales_multiplier
    + noise
).clip(lower=0)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)

print(
    f"Concept drift dataset created: {OUTPUT}"
)

print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)