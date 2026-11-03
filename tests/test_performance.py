import time
import statistics
import requests
import json
from pathlib import Path


API_URL = "http://127.0.0.1:8000"

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts" / "api_performance_report.json"


payload = {
    "Item_Identifier": "FD001",
    "Item_Weight": 9.3,
    "Item_Fat_Content": "Low Fat",
    "Item_Visibility": 0.016,
    "Item_Type": "Dairy",
    "Item_MRP": 249.8,
    "Outlet_Identifier": "OUT049",
    "Outlet_Establishment_Year": 1999,
    "Outlet_Size": "Medium",
    "Outlet_Location_Type": "Tier 1",
    "Outlet_Type": "Supermarket Type1"
}


def main():
    latencies = []
    successful = 0
    failed = 0
    predictions = []

    total_requests = 50

    for _ in range(total_requests):
        start = time.perf_counter()

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )

            latency = time.perf_counter() - start
            latencies.append(latency)

            if response.status_code == 200:
                successful += 1
                predictions.append(
                    response.json()["predicted_sales"]
                )
            else:
                failed += 1

        except requests.RequestException:
            failed += 1

    average_latency = (
        statistics.mean(latencies) if latencies else 0
    )

    throughput = (
        successful / sum(latencies) if sum(latencies) > 0 else 0
    )

    report = {
        "experiment": "Experiment 10 - API Performance",
        "total_requests": total_requests,
        "successful_requests": successful,
        "failed_requests": failed,
        "average_latency_seconds": round(average_latency, 6),
        "throughput_requests_per_second": round(throughput, 2),
        "response_consistency": (
            len(set(predictions)) == 1 if predictions else False
        )
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nBIGMART API PERFORMANCE REPORT")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()