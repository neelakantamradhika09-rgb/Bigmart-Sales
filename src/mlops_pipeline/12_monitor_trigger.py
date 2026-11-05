import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DRIFT_REPORT = (
    ROOT / "artifacts" / "data_drift_report.json"
)

PERFORMANCE_REPORT = (
    ROOT / "artifacts" / "performance_monitoring_report.json"
)

OUTPUT = (
    ROOT / "artifacts" / "monitor_trigger_report.json"
)


def main():

    with open(DRIFT_REPORT, "r", encoding="utf-8") as f:
        drift = json.load(f)

    with open(PERFORMANCE_REPORT, "r", encoding="utf-8") as f:
        performance = json.load(f)

    reasons = []

    if drift.get("overall_drift_detected"):
        reasons.append("DATA_DRIFT")

    failure_rate = performance.get(
        "failure_rate",
        0,
    )

    p95_latency = (
        performance.get("latency_ms", {})
        .get("p95")
    )

    if failure_rate > 0.05:
        reasons.append("HIGH_FAILURE_RATE")

    if p95_latency is not None and p95_latency > 500:
        reasons.append("HIGH_LATENCY")

    retraining_required = len(reasons) > 0

    report = {
        "stage": "12_monitor_trigger",
        "status": "SUCCESS",
        "retraining_required": retraining_required,
        "reasons": reasons,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("MONITORING TRIGGER")
    print("=" * 60)

    print(
        "Retraining required:",
        retraining_required,
    )

    print(
        "Reasons:",
        reasons if reasons else "NONE",
    )

    print(f"Report saved to: {OUTPUT}")


if __name__ == "__main__":
    main()