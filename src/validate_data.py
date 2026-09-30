import os
import json
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

TRAIN_FILE = "data/raw/Train.csv"
TEST_FILE = "data/raw/Test.csv"

OUTPUT_DIR = "artifacts/validation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Required columns
# ============================================================

COMMON_COLUMNS = [
    "Item_Identifier",
    "Item_Weight",
    "Item_Fat_Content",
    "Item_Visibility",
    "Item_Type",
    "Item_MRP",
    "Outlet_Identifier",
    "Outlet_Establishment_Year",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type"
]

# Train.csv contains the target
TRAIN_TARGET = "Item_Outlet_Sales"


# ============================================================
# 3. Validation function
# ============================================================

def validate_dataset(file_path, dataset_name, is_train=True):

    print("\n" + "=" * 60)
    print(f"VALIDATING {dataset_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    print(f"File: {file_path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = df.isnull().sum()

    missing_values = missing_values[
        missing_values > 0
    ]

    print("\nMissing Values:")

    if missing_values.empty:
        print("No missing values found.")
    else:
        print(missing_values)

    # Save missing-value report
    missing_report = missing_values.to_dict()

    missing_file = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name.lower()}_missing_attribute.csv"
    )

    if not missing_values.empty:
        missing_values.to_csv(
            missing_file,
            header=["missing_count"]
        )

    # --------------------------------------------------------
    # Datatype validation
    # --------------------------------------------------------

    print("\nData Types:")
    print(df.dtypes)

    datatype_report = {}

    for column in df.columns:
        datatype_report[column] = str(df[column].dtype)

    # --------------------------------------------------------
    # Check duplicate rows
    # --------------------------------------------------------

    duplicate_count = int(df.duplicated().sum())

    print("\nDuplicate Rows:", duplicate_count)

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = COMMON_COLUMNS.copy()

    if is_train:
        required_columns.append(TRAIN_TARGET)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    extra_columns = [
        column
        for column in df.columns
        if column not in required_columns
    ]

    print("\nMissing Required Columns:")

    if missing_columns:
        print(missing_columns)
    else:
        print("None")

    print("\nExtra Columns:")

    if extra_columns:
        print(extra_columns)
    else:
        print("None")

    # --------------------------------------------------------
    # Categorical validation
    # --------------------------------------------------------

    categorical_columns = [
        "Item_Fat_Content",
        "Item_Type",
        "Outlet_Size",
        "Outlet_Location_Type",
        "Outlet_Type"
    ]

    categorical_report = {}

    print("\nCategorical Values:")

    for column in categorical_columns:

        if column in df.columns:

            values = (
                df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            categorical_report[column] = values

            print(f"\n{column}:")
            print(values)

    # --------------------------------------------------------
    # Target validation for Train.csv
    # --------------------------------------------------------

    target_report = {}

    if is_train and TRAIN_TARGET in df.columns:

        target = df[TRAIN_TARGET]

        target_report = {
            "target_column": TRAIN_TARGET,
            "missing_values": int(target.isnull().sum()),
            "minimum": float(target.min()),
            "maximum": float(target.max()),
            "mean": float(target.mean())
        }

        print("\nTarget Information:")
        print("Target:", TRAIN_TARGET)
        print("Missing:", target_report["missing_values"])
        print("Minimum:", target_report["minimum"])
        print("Maximum:", target_report["maximum"])
        print("Mean:", target_report["mean"])

    # --------------------------------------------------------
    # Final validation status
    # --------------------------------------------------------

    validation_status = "PASSED"

    if missing_columns:
        validation_status = "FAILED"

    if is_train and TRAIN_TARGET not in df.columns:
        validation_status = "FAILED"

    # --------------------------------------------------------
    # Create validation report
    # --------------------------------------------------------

    report = {
        "dataset_name": dataset_name,
        "file_path": file_path,
        "validation_status": validation_status,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": missing_report,
        "duplicate_rows": duplicate_count,
        "missing_required_columns": missing_columns,
        "extra_columns": extra_columns,
        "data_types": datatype_report,
        "categorical_values": categorical_report,
        "target_information": target_report
    }

    report_file = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name.lower()}_validation_report.json"
    )

    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)

    print("\nValidation Status:", validation_status)

    print("Report saved to:")
    print(report_file)


# ============================================================
# 4. Validate Train.csv
# ============================================================

validate_dataset(
    TRAIN_FILE,
    "train",
    is_train=True
)


# ============================================================
# 5. Validate Test.csv
# ============================================================

validate_dataset(
    TEST_FILE,
    "test",
    is_train=False
)


# ============================================================
# 6. Final message
# ============================================================

print("\n" + "=" * 60)
print("DATA VALIDATION COMPLETED")
print("=" * 60)

print("\nTrain.csv and Test.csv were validated separately.")
print("The datasets were NOT combined.")