import json
from pathlib import Path


ARTIFACTS = Path(
    "artifacts"
)


def load_report(filename):

    path = ARTIFACTS / filename

    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


data_drift = load_report(
    "data_drift_report.json"
)

prediction_drift = load_report(
    "prediction_drift_report.json"
)

concept_drift = load_report(
    "concept_drift_report.json"
)


data_detected = data_drift.get(
    "overall_drift_detected",
    data_drift.get(
        "data_drift_detected",
        False
    )
)

prediction_detected = prediction_drift.get(
    "prediction_drift_detected",
    False
)

concept_detected = concept_drift.get(
    "concept_drift_detected",
    False
)

report = {
    "data_drift": bool(
        data_detected
    ),
    "prediction_drift": bool(
        prediction_detected
    ),
    "concept_drift": bool(
        concept_detected
    ),
}

report["any_drift_detected"] = any(
    report.values()
)

report["recommended_action"] = (
    "RETRAIN"
    if report["any_drift_detected"]
    else "CONTINUE_MONITORING"
)

output = (
    ARTIFACTS /
    "drift_analysis_report.json"
)

output.write_text(
    json.dumps(
        report,
        indent=2
    ),
    encoding="utf-8"
)

print(
    json.dumps(
        report,
        indent=2
    )
)