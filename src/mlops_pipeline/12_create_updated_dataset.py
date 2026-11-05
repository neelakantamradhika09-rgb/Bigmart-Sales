from pathlib import Path

import pandas as pd


SOURCE = Path(
    "data/monitoring/drift/evolved_data.csv"
)

OUTPUT = Path(
    "data/monitoring/updated_data.csv"
)

if not SOURCE.exists():

    raise FileNotFoundError(
        f"Source dataset not found: {SOURCE}"
    )


df = pd.read_csv(
    SOURCE
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
    "Updated monitoring dataset created."
)

print(
    f"Path: {OUTPUT}"
)

print(
    f"Rows: {len(df)}"
)