import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "flood-prediction.csv"
OUTPUT_DIR = BASE_DIR / "flood_model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["MonthName"] = df["Date"].dt.month_name()
    return df


def summarize_data(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Locations: {', '.join(sorted(df['Location'].unique()))}")
    print(f"Flood cases: {int(df['FloodOccurrence'].sum()):,}")
    print(f"Flood rate: {df['FloodOccurrence'].mean() * 100:.2f}%")
    print("\nMissing values:")
    print(df.isna().sum())

    numeric_summary = df[["Rainfall_mm", "WaterLevel_m", "SoilMoisture_pct", "Elevation_m"]].describe().round(2)
    numeric_summary.to_csv(OUTPUT_DIR / "numeric_summary.csv")

    location_summary = (
        df.groupby("Location")
        .agg(
            Records=("Location", "size"),
            FloodCases=("FloodOccurrence", "sum"),
            FloodRate=("FloodOccurrence", "mean"),
            AvgRainfall_mm=("Rainfall_mm", "mean"),
            AvgWaterLevel_m=("WaterLevel_m", "mean"),
            AvgSoilMoisture_pct=("SoilMoisture_pct", "mean"),
            Elevation_m=("Elevation_m", "first"),
        )
        .round(4)
    )
    location_summary.to_csv(OUTPUT_DIR / "location_summary.csv")

    print("\nLocation summary:")
    print(location_summary)


def build_pipeline() -> Pipeline:
    numeric_features = [
        "Rainfall_mm",
        "WaterLevel_m",
        "SoilMoisture_pct",
        "Elevation_m",
        "Month",
        "DayOfYear",
    ]
    categorical_features = ["Location"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_model(df: pd.DataFrame) -> None:
    feature_columns = [
        "Location",
        "Rainfall_mm",
        "WaterLevel_m",
        "SoilMoisture_pct",
        "Elevation_m",
        "Month",
        "DayOfYear",
    ]

    X = df[feature_columns]
    y = df["FloodOccurrence"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba),
    }

    metrics_df = pd.DataFrame(
        {"Metric": list(metrics.keys()), "Value": [round(v, 4) for v in metrics.values()]}
    )
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    print(metrics_df)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_No_Flood", "Actual_Flood"],
        columns=["Predicted_No_Flood", "Predicted_Flood"],
    )
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    numeric_features = [
        "Rainfall_mm",
        "WaterLevel_m",
        "SoilMoisture_pct",
        "Elevation_m",
        "Month",
        "DayOfYear",
    ]

    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    encoded_location_names = list(encoder.get_feature_names_out(["Location"]))
    raw_importances = pd.Series(
        pipeline.named_steps["model"].feature_importances_,
        index=numeric_features + encoded_location_names,
    )

    grouped_importances = pd.Series(
        {
            "Rainfall (mm)": raw_importances["Rainfall_mm"],
            "Soil Moisture (%)": raw_importances["SoilMoisture_pct"],
            "Water Level (m)": raw_importances["WaterLevel_m"],
            "Day of Year": raw_importances["DayOfYear"],
            "Month": raw_importances["Month"],
            "Elevation (m)": raw_importances["Elevation_m"],
            "Location (combined)": raw_importances[
                [name for name in raw_importances.index if name.startswith("Location_")]
            ].sum(),
        }
    ).sort_values(ascending=False)

    grouped_importances.round(6).to_csv(OUTPUT_DIR / "feature_importance.csv", header=["Importance"])

    plot_outputs(df, grouped_importances, cm)


def plot_outputs(df: pd.DataFrame, grouped_importances: pd.Series, cm) -> None:
    plt.style.use("default")

    plt.figure(figsize=(6, 4))
    df["FloodOccurrence"].map({0: "No Flood", 1: "Flood"}).value_counts().reindex(["No Flood", "Flood"]).plot(kind="bar")
    plt.title("Class Distribution of Flood Occurrence")
    plt.xlabel("Class")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6.5, 4))
    df.groupby("Location")["FloodOccurrence"].sum().sort_values(ascending=False).plot(kind="bar")
    plt.title("Recorded Flood Cases by Location")
    plt.xlabel("Location")
    plt.ylabel("Flood Cases")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "floods_by_location.png", dpi=200)
    plt.close()

    corr = df[["Rainfall_mm", "WaterLevel_m", "SoilMoisture_pct", "Elevation_m", "FloodOccurrence"]].corr(numeric_only=True)
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title("Correlation Matrix of Numerical Variables")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    grouped_importances.sort_values().plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Flood", "Flood"],
        yticklabels=["No Flood", "Flood"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    plt.figure(figsize=(8, 4))
    df.groupby("MonthName")["FloodOccurrence"].sum().reindex(month_order).plot(kind="bar")
    plt.title("Flood Cases by Month")
    plt.xlabel("Month")
    plt.ylabel("Flood Cases")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "floods_by_month.png", dpi=200)
    plt.close()


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = load_data(CSV_PATH)
    summarize_data(df)
    evaluate_model(df)

    print("\nAll reports, tables, and figures were saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
