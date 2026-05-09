from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from competitions.titanic.src.feature_v1 import add_feature_v1_columns
from competitions.titanic.src.model_comparison import build_pipeline, find_model_spec

RAW_DATA_DIR = Path("competitions/titanic/data/raw")
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
EXPERIMENTS_PATH = Path("competitions/titanic/experiments/experiments.csv")
FIGURES_DIR = Path("competitions/titanic/reports/figures")
REPORT_PATH = Path("competitions/titanic/reports/titanic_visual_error_analysis.md")
OOF_PREDICTIONS_PATH = Path("competitions/titanic/reports/titanic_oof_predictions.csv")
SEGMENT_ERRORS_PATH = Path("competitions/titanic/reports/titanic_segment_error_analysis.csv")
TARGET_COLUMN = "Survived"
FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked", "Title"]
SEGMENT_COLUMNS = ["Sex", "Pclass", "Title", "AgeBin", "FareBin", "Embarked"]

@dataclass(frozen=True)
class VisualAnalysisResult:
    oof_accuracy: float
    oof_error_count: int
    row_count: int
    figure_paths: list[Path]

def load_train_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)

def add_analysis_bins(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["AgeBin"] = pd.cut(
        result["Age"],
        bins=[-1.0, 12.0, 18.0, 35.0, 60.0, 200.0],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    ).astype(str)
    result["FareBin"] = pd.qcut(result["Fare"], q=4, duplicates="drop").astype(str)
    return result

def prepare_analysis_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    featured = add_feature_v1_columns(raw_df)
    return add_analysis_bins(featured)

def save_bar_chart(labels: list[str], values: list[float], title: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_survival_rate_by_category(df: pd.DataFrame, column: str, path: Path) -> None:
    grouped = df.groupby(column, dropna=False)[TARGET_COLUMN].agg(["mean", "count"]).reset_index().sort_values("mean", ascending=False)
    labels = [f"{row[column]}\n(n={int(row['count'])})" for _, row in grouped.iterrows()]
    values = [float(value) for value in grouped["mean"]]
    save_bar_chart(labels=labels, values=values, title=f"Survival Rate by {column}", ylabel="Survival rate", path=path)

def plot_numeric_distribution_by_survival(df: pd.DataFrame, column: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    survived = df[df[TARGET_COLUMN] == 1][column].dropna()
    died = df[df[TARGET_COLUMN] == 0][column].dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(died, bins=30, alpha=0.7, label="Died")
    plt.hist(survived, bins=30, alpha=0.7, label="Survived")
    plt.title(f"{column} Distribution by Survival")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def parse_first_float(value: str) -> float | None:
    match = re.search(r"\d+\.\d+", value)
    if match is None:
        return None
    return float(match.group(0))

def load_experiments() -> pd.DataFrame:
    experiments = pd.read_csv(EXPERIMENTS_PATH)
    experiments["local_cv_mean"] = experiments["local_score"].astype(str).apply(parse_first_float)
    experiments["public_lb_numeric"] = experiments["public_lb_score"].astype(str).apply(parse_first_float)
    return experiments

def plot_public_scores(experiments: pd.DataFrame, path: Path) -> None:
    scored = experiments.dropna(subset=["public_lb_numeric"]).copy()
    labels = [str(value).replace("titanic_", "").replace("_", "\n") for value in scored["experiment_id"]]
    values = [float(value) for value in scored["public_lb_numeric"]]
    save_bar_chart(labels=labels, values=values, title="Public Leaderboard Scores by Experiment", ylabel="Public LB score", path=path)

def plot_cv_vs_public(experiments: pd.DataFrame, path: Path) -> None:
    scored = experiments.dropna(subset=["local_cv_mean", "public_lb_numeric"]).copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(scored["local_cv_mean"], scored["public_lb_numeric"])
    for _, row in scored.iterrows():
        label = str(row["experiment_id"]).replace("titanic_exp_", "exp_")
        plt.annotate(label, (float(row["local_cv_mean"]), float(row["public_lb_numeric"])), fontsize=8)
    plt.title("Local CV Mean vs Public LB")
    plt.xlabel("Local CV mean")
    plt.ylabel("Public LB score")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def build_oof_predictions(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    oof_prediction = pd.Series(index=df.index, dtype="float64")
    oof_probability = pd.Series(index=df.index, dtype="float64")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    spec = find_model_spec("hist_gradient_boosting")
    for train_index, valid_index in cv.split(X, y):
        X_train = X.iloc[train_index].copy()
        X_valid = X.iloc[valid_index].copy()
        y_train = y.iloc[train_index].copy()
        pipeline = build_pipeline(spec.estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_valid)
        probabilities = pipeline.predict_proba(X_valid)[:, 1]
        oof_prediction.iloc[valid_index] = predictions
        oof_probability.iloc[valid_index] = probabilities
    result = df.copy()
    result["oof_prediction"] = oof_prediction.astype(int)
    result["oof_probability_survived"] = oof_probability.astype(float)
    result["oof_correct"] = result["oof_prediction"] == result[TARGET_COLUMN]
    result["oof_error"] = ~result["oof_correct"]
    return result

def summarize_errors_by_segment(oof_df: pd.DataFrame, segment_columns: list[str] = SEGMENT_COLUMNS) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in segment_columns:
        grouped = (
            oof_df.groupby(column, dropna=False)
            .agg(rows=("PassengerId", "count"), errors=("oof_error", "sum"), survival_rate=(TARGET_COLUMN, "mean"), predicted_survival_rate=("oof_prediction", "mean"))
            .reset_index()
        )
        for _, row in grouped.iterrows():
            row_count = int(row["rows"])
            error_count = int(row["errors"])
            rows.append({"segment": column, "value": str(row[column]), "rows": row_count, "errors": error_count, "error_rate": error_count / row_count, "survival_rate": float(row["survival_rate"]), "predicted_survival_rate": float(row["predicted_survival_rate"])})
    return pd.DataFrame(rows).sort_values(["error_rate", "rows"], ascending=[False, False])

def plot_oof_errors_by_segment(segment_errors: pd.DataFrame, segment: str, path: Path) -> None:
    segment_df = segment_errors[segment_errors["segment"] == segment].copy()
    segment_df = segment_df[segment_df["rows"] >= 5].sort_values("error_rate", ascending=False)
    labels = [f"{row['value']}\n(n={int(row['rows'])})" for _, row in segment_df.iterrows()]
    values = [float(value) for value in segment_df["error_rate"]]
    save_bar_chart(labels=labels, values=values, title=f"OOF Error Rate by {segment}", ylabel="OOF error rate", path=path)

def build_report(result: VisualAnalysisResult, segment_errors: pd.DataFrame) -> str:
    top_errors = segment_errors.head(10)
    top_error_lines = ["| Segment | Value | Rows | Errors | Error rate | Survival rate | Predicted survival rate |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in top_errors.iterrows():
        top_error_lines.append(f"| {row['segment']} | {row['value']} | {int(row['rows'])} | {int(row['errors'])} | {float(row['error_rate']):.3f} | {float(row['survival_rate']):.3f} | {float(row['predicted_survival_rate']):.3f} |")
    figure_lines = [f"- `{path}`" for path in result.figure_paths]
    return f"""# Titanic Visual EDA and Error Analysis

## Goal

Use graphs and out-of-fold error analysis to understand where the current best model succeeds and fails.

Current best model:

```text
plus_title + HistGradientBoosting
```

## OOF Performance

- Rows: {result.row_count}
- OOF accuracy: {result.oof_accuracy:.5f}
- OOF errors: {result.oof_error_count}

## Top Error Segments

{chr(10).join(top_error_lines)}

## Generated Figures

{chr(10).join(figure_lines)}

## Interpretation Guide

Use these figures to answer:

```text
Which features separate survival strongly?
Which segments have high model error?
Which segments are under-predicted or over-predicted?
Where could the next +12 correct predictions come from?
```

## Important Warning

A chart is useful only if it changes the next experiment decision.

The next experiment should come from a concrete observation in this report, not from random feature guessing.
"""

def run_pipeline() -> VisualAnalysisResult:
    raw_df = load_train_data()
    analysis_df = prepare_analysis_dataframe(raw_df)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_paths = [FIGURES_DIR / "survival_by_sex.png", FIGURES_DIR / "survival_by_pclass.png", FIGURES_DIR / "survival_by_title.png", FIGURES_DIR / "survival_by_age_bin.png", FIGURES_DIR / "survival_by_fare_bin.png", FIGURES_DIR / "age_distribution_by_survival.png", FIGURES_DIR / "fare_distribution_by_survival.png", FIGURES_DIR / "experiment_public_scores.png", FIGURES_DIR / "experiment_cv_vs_public.png", FIGURES_DIR / "oof_errors_by_title.png", FIGURES_DIR / "oof_errors_by_pclass.png", FIGURES_DIR / "oof_errors_by_sex.png"]
    plot_survival_rate_by_category(analysis_df, "Sex", figure_paths[0])
    plot_survival_rate_by_category(analysis_df, "Pclass", figure_paths[1])
    plot_survival_rate_by_category(analysis_df, "Title", figure_paths[2])
    plot_survival_rate_by_category(analysis_df, "AgeBin", figure_paths[3])
    plot_survival_rate_by_category(analysis_df, "FareBin", figure_paths[4])
    plot_numeric_distribution_by_survival(analysis_df, "Age", figure_paths[5])
    plot_numeric_distribution_by_survival(analysis_df, "Fare", figure_paths[6])
    experiments = load_experiments()
    plot_public_scores(experiments, figure_paths[7])
    plot_cv_vs_public(experiments, figure_paths[8])
    oof_df = build_oof_predictions(analysis_df)
    oof_df.to_csv(OOF_PREDICTIONS_PATH, index=False)
    segment_errors = summarize_errors_by_segment(oof_df)
    segment_errors.to_csv(SEGMENT_ERRORS_PATH, index=False)
    plot_oof_errors_by_segment(segment_errors, "Title", figure_paths[9])
    plot_oof_errors_by_segment(segment_errors, "Pclass", figure_paths[10])
    plot_oof_errors_by_segment(segment_errors, "Sex", figure_paths[11])
    oof_accuracy = float(accuracy_score(oof_df[TARGET_COLUMN], oof_df["oof_prediction"]))
    oof_error_count = int(oof_df["oof_error"].sum())
    result = VisualAnalysisResult(oof_accuracy=oof_accuracy, oof_error_count=oof_error_count, row_count=len(oof_df), figure_paths=figure_paths)
    REPORT_PATH.write_text(build_report(result, segment_errors), encoding="utf-8")
    return result

def main() -> None:
    result = run_pipeline()
    print("=== Titanic Visual EDA and Error Analysis ===")
    print(f"OOF accuracy: {result.oof_accuracy:.5f}")
    print(f"OOF errors: {result.oof_error_count}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved OOF predictions to: {OOF_PREDICTIONS_PATH}")
    print(f"Saved segment errors to: {SEGMENT_ERRORS_PATH}")
    print(f"Saved figures to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
