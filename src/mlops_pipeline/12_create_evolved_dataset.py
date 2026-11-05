from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(
    "data/raw/Test.csv"
)

OUTPUT = Path(
    "data/monitoring/drift/evolved_data.csv"
)

rng = np.random.default_rng(42)

df = pd.read_csv(
    SOURCE
)

if "Item_MRP" in df.columns:

    df["Item_MRP"] = (
        df["Item_MRP"]
        * rng.normal(
            1.05,
            0.02,
            len(df)
        )
    )


if "Item_Visibility" in df.columns:

    df["Item_Visibility"] = (
        df["Item_Visibility"]
        * rng.normal(
            1.15,
            0.03,
            len(df)
        )
    ).clip(
        lower=0,
        upper=1
    )


if "Item_Weight" in df.columns:

    df["Item_Weight"] = (
        df["Item_Weight"]
        * rng.normal(
            0.98,
            0.02,
            len(df)
        )
    ).clip(
        lower=0.1
    )


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT,
    index=False
)

print(
    f"Evolved dataset created: {OUTPUT}"
)

print(
    f"Rows: {len(df)}"
)