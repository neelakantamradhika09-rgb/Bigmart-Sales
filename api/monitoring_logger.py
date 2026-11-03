import json
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "api_requests.jsonl"


def log_request(endpoint, status, latency, records=1):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "status": status,
        "latency_seconds": round(latency, 6),
        "records": records
    }

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")