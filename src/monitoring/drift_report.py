# generates an Evidently AI data drift report
# compares train feature distributions vs test feature distributions
# output is a standalone HTML report saved to data/processed/reports/
import os
import pandas as pd
from pathlib import Path

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

# paths
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
REPORTS_DIR   = PROJECT_ROOT / "data" / "processed" / "reports"

FEATURE_COLS = [
    "value",
    "rolling_mean",
    "rolling_std",
    "rolling_zscore",
    "rate_of_change",
    "lag_1",
    "lag_2"
]


def generate_drift_report():
    print("Loading feature store...")
    df = pd.read_parquet(FEATURES_PATH)

    # reference = train split (what the model was trained on)
    # current   = test split  (what the model sees in production)
    reference_df = df[df["split"] == "train"][FEATURE_COLS].reset_index(drop=True)
    current_df   = df[df["split"] == "test"][FEATURE_COLS].reset_index(drop=True)

    print(f"Reference (train): {len(reference_df):,} rows")
    print(f"Current   (test):  {len(current_df):,} rows")

    # define schema — all features are numerical
    definition = DataDefinition(numerical_columns=FEATURE_COLS)

    reference = Dataset.from_pandas(reference_df, data_definition=definition)
    current   = Dataset.from_pandas(current_df,   data_definition=definition)

    # build and run the drift report
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference, current)

    # save HTML report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = REPORTS_DIR / "drift_report.html"
    result.save_html(str(report_path))

    print(f"\nDrift report saved: {report_path}")
    print(f"Open {report_path} in a browser to view the interactive report")


if __name__ == "__main__":
    generate_drift_report()