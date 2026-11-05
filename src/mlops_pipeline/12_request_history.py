import json
from datetime import datetime
from pathlib import Path


OUTPUT = Path(
    "artifacts/request_history.jsonl"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

request_record = {
    "timestamp": datetime.now().isoformat(),
    "service": "bigmart-sales-api",
    "endpoint": "/predict",
    "method": "POST",
    "status": "recorded",
}

with OUTPUT.open(
    "a",
    encoding="utf-8"
) as f:

    f.write(
        json.dumps(
            request_record
        ) + "\n"
    )

print(
    f"Request history updated: {OUTPUT}"
)