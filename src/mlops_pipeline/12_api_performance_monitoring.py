import json
import statistics
import time
from pathlib import Path

import requests


BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/predict"

INPUT_DATA = {
    "Item_Identifier": "FDA15",
    "Item_Weight": 9.3,
    "Item_Fat_Content": "Low Fat",
    "Item_Visibility": 0.016,
    "Item_Type": "Dairy",
    "Item_MRP": 249.8,
    "Outlet_Identifier": "OUT049",
    "Outlet_Establishment_Year": 1999,
    "Outlet_Size": "Medium",
    "Outlet_Location_Type": "Tier 1",
    "Outlet_Type": "Supermarket Type1",
}

REQUEST_COUNT = 20

latencies = []
success_count = 0
failure_count = 0
predictions = []

for i in range(REQUEST_COUNT):
    start = time.perf_counter()

    try:
        response = requests.post(
            ENDPOINT,
            json=INPUT_DATA,
            timeout=10
        )

        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        if response.status_code == 200:
            success_count += 1

            try:
                result = response.json()

                if "predicted_sales" in result:
                    predictions.append(
                        float(result["predicted_sales"])
                    )

            except Exception:
                pass

        else:
            failure_count += 1

        print(
            f"Request {i + 1}: "
            f"status={response.status_code}, "
            f"latency={latency_ms:.2f} ms"
        )

    except Exception as exc:
        failure_count += 1

        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        print(f"Request {i + 1} failed: {exc}")


total_time = sum(latencies) / 1000
failure_rate = failure_count / REQUEST_COUNT

sorted_latencies = sorted(latencies)

p95_index = max(
    0,
    int(0.95 * len(sorted_latencies)) - 1
)

p95_latency = sorted_latencies[p95_index]

report = {
    "endpoint": ENDPOINT,
    "total_requests": REQUEST_COUNT,
    "successful_requests": success_count,
    "failed_requests": failure_count,
    "failure_rate": failure_rate,
    "mean_latency_ms": statistics.mean(latencies),
    "p95_latency_ms": p95_latency,
    "throughput_requests_per_second": (
        REQUEST_COUNT / total_time
        if total_time > 0
        else 0
    ),
    "prediction_count": len(predictions),
    "prediction_mean": (
        statistics.mean(predictions)
        if predictions
        else None
    ),
    "status": (
        "PASS"
        if failure_rate <= 0.05 and p95_latency <= 500
        else "ALERT"
    ),
}

Path("artifacts").mkdir(
    parents=True,
    exist_ok=True
)

output = Path(
    "artifacts/api_performance_monitoring_report.json"
)

output.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\nAPI performance monitoring completed.")
print(json.dumps(report, indent=2))