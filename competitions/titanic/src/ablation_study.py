from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold, cross_val_score  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from competitions.titanic.src.feature_v1 import add_feature_v1_columns


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
TRAIN_PATH = RAW_DATA_DIR / "train.csv"

REPORT_PATH = Path("competitions/titanic/reports/titanic_ablation_report.md")
RESULTS_PATH = Path("competitions/titanic/reports/titanic_ablation_results.csv")

TARGET_COLUMN = "Survived"

BASELINE_NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
]

BASELINE_CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
]


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    numeric_features: list[str]
    categorical_features: list[str]
    hypothesis: str


@dataclass(frozen=True)
class AblationResult:
    name: str
    numeric_features: list[str]
    categorical_features: list[str]
    hypothesis: str
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    delta_vs_baseline: float


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(TRAIN_PATH)
    featured_df = add_feature_v1_columns(train_df)

    y = featured_df[TARGET_COLUMN].copy()

    return featured_df, y


def build_feature_configs() -> list[FeatureConfig]:
    base_num = BASELINE_NUMERIC_FEATURES
    base_cat = BASELINE_CATEGORICAL_FEATURES

    return [
        FeatureConfig(
            name="baseline",
            numeric_features=base_num,
            categorical_features=base_cat,
            hypothesis="Original baseline features only.",
        ),
        FeatureConfig(
            name="plus_family_size",
            numeric_features=base_num + ["FamilySize"],
            categorical_features=base_cat,
            hypothesis="FamilySize may capture family survival patterns.",
        ),
        FeatureConfig(
            name="plus_is_alone",
            numeric_features=base_num + ["IsAlone"],
            categorical_features=base_cat,
            hypothesis="Traveling alone may affect survival probability.",
        ),
        FeatureConfig(
            name="plus_has_cabin",
            numeric_features=base_num + ["HasCabin"],
            categorical_features=base_cat,
            hypothesis="Cabin availability may proxy passenger class or record quality.",
        ),
        FeatureConfig(
            name="plus_title",
            numeric_features=base_num,
            categorical_features=base_cat + ["Title"],
            hypothesis="Title may capture gender, age, and social status information.",
        ),
        FeatureConfig(
            name="plus_family_size_is_alone",
            numeric_features=base_num + ["FamilySize", "IsAlone"],
            categorical_features=base_cat,
            hypothesis="FamilySize and IsAlone test whether family structure improves the model.",
        ),
        FeatureConfig(
            name="plus_family_size_title",
            numeric_features=base_num + ["FamilySize"],
            categorical_features=base_cat + ["Title"],
            hypothesis="Title and FamilySize may interact with age and family survival behavior.",
        ),
        FeatureConfig(
            name="plus_has_cabin_title",
            numeric_features=base_num + ["HasCabin"],
            categorical_features=base_cat + ["Title"],
            hypothesis="Title plus cabin availability may combine status signals.",
        ),
        FeatureConfig(
            name="feature_v1_all",
            numeric_features=base_num + ["FamilySize", "IsAlone", "HasCabin"],
            categorical_features=base_cat + ["Title"],
            hypothesis="Full feature v1 set.",
        ),
        FeatureConfig(
            name="feature_v1_without_has_cabin",
            numeric_features=base_num + ["FamilySize", "IsAlone"],
            categorical_features=base_cat + ["Title"],
            hypothesis="Check whether HasCabin helps or hurts the full feature set.",
        ),
        FeatureConfig(
            name="feature_v1_without_is_alone",
            numeric_features=base_num + ["FamilySize", "HasCabin"],
            categorical_features=base_cat + ["Title"],
            hypothesis="Check whether IsAlone is redundant when FamilySize is present.",
        ),
    ]


def build_model_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def run_cv_for_config(
    X_source: pd.DataFrame,
    y: pd.Series,
    config: FeatureConfig,
    n_splits: int = 5,
) -> tuple[list[float], float, float]:
    selected_features = config.numeric_features + config.categorical_features
    X = X_source[selected_features].copy()

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    model_pipeline = build_model_pipeline(
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
    )

    raw_scores = cross_val_score(
        model_pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    scores = [float(score) for score in raw_scores]
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_score = variance ** 0.5

    return scores, mean_score, std_score


def run_ablation_study() -> list[AblationResult]:
    X_source, y = load_training_data()

    configs = build_feature_configs()

    raw_results = []
    baseline_mean = 0.0

    for config in configs:
        scores, mean_score, std_score = run_cv_for_config(X_source, y, config)

        if config.name == "baseline":
            baseline_mean = mean_score

        raw_results.append(
            (
                config,
                scores,
                mean_score,
                std_score,
            )
        )

    results = []
    for config, scores, mean_score, std_score in raw_results:
        results.append(
            AblationResult(
                name=config.name,
                numeric_features=config.numeric_features,
                categorical_features=config.categorical_features,
                hypothesis=config.hypothesis,
                cv_scores=scores,
                cv_mean=mean_score,
                cv_std=std_score,
                delta_vs_baseline=mean_score - baseline_mean,
            )
        )

    return sorted(results, key=lambda result: result.cv_mean, reverse=True)


def results_to_dataframe(results: list[AblationResult]) -> pd.DataFrame:
    rows = []

    for result in results:
        rows.append(
            {
                "name": result.name,
                "cv_mean": result.cv_mean,
                "cv_std": result.cv_std,
                "delta_vs_baseline": result.delta_vs_baseline,
                "numeric_features": ", ".join(result.numeric_features),
                "categorical_features": ", ".join(result.categorical_features),
                "hypothesis": result.hypothesis,
                "fold_scores": ", ".join(f"{score:.5f}" for score in result.cv_scores),
            }
        )

    return pd.DataFrame(rows)


def save_results_csv(results: list[AblationResult], path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results_to_dataframe(results).to_csv(path, index=False)


def build_report(results: list[AblationResult]) -> str:
    table_lines = [
        "| Rank | Feature set | CV mean | CV std | Delta vs baseline |",
        "|---:|---|---:|---:|---:|",
    ]

    for rank, result in enumerate(results, start=1):
        table_lines.append(
            f"| {rank} | {result.name} | {result.cv_mean:.5f} | {result.cv_std:.5f} | {result.delta_vs_baseline:+.5f} |"
        )

    best = results[0]
    baseline = next(result for result in results if result.name == "baseline")

    return f"""# Titanic Feature Ablation Study

## Goal

Understand which engineered features actually help the model.

Feature v1 added multiple features at once:

- FamilySize
- IsAlone
- HasCabin
- Title

This report tests individual and grouped feature contributions using the same model and validation strategy.

## Validation Setup

- Model: LogisticRegression
- Metric: accuracy
- Validation: 5-fold StratifiedKFold
- Random state: 42

## Results

{chr(10).join(table_lines)}

## Best Local Feature Set

- Name: {best.name}
- CV mean: {best.cv_mean:.5f}
- CV std: {best.cv_std:.5f}
- Delta vs baseline: {best.delta_vs_baseline:+.5f}

## Baseline Reference

- Name: {baseline.name}
- CV mean: {baseline.cv_mean:.5f}
- CV std: {baseline.cv_std:.5f}

## Interpretation

Ablation study answers this question:

```text
When a score improves after adding several features,
which feature actually caused the improvement?
```

Important warning:

A feature can look good locally but fail on Kaggle public leaderboard.
Therefore, we use ablation to choose candidates, not to blindly trust local CV.

## Next Step

Use the best local feature set for model comparison:

- LogisticRegression
- RandomForest
- HistGradientBoosting
"""


def save_report(report: str, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_pipeline() -> list[AblationResult]:
    results = run_ablation_study()
    save_results_csv(results)
    save_report(build_report(results))

    return results


def main() -> None:
    results = run_pipeline()

    print("=== Titanic Feature Ablation Study ===")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.name} | "
            f"cv_mean={result.cv_mean:.5f} | "
            f"cv_std={result.cv_std:.5f} | "
            f"delta={result.delta_vs_baseline:+.5f}"
        )

    print()
    print(f"Saved CSV to: {RESULTS_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
