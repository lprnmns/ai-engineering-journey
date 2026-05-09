from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from competitions.titanic.src.feature_v1 import add_feature_v1_columns
from competitions.titanic.src.model_comparison import build_submission, validate_submission
from competitions.titanic.src.risky_flip_guard import (
    CURRENT_BEST_SUBMISSION_PATH,
    build_flip_rows,
    load_submission,
    summarize_guard,
)


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

REPORTS_DIR = Path("competitions/titanic/reports")
SUBMISSIONS_DIR = Path("competitions/titanic/submissions")

OUTPUT_DIR = REPORTS_DIR / "auto_experiment_suite"
SUBMISSION_OUTPUT_DIR = SUBMISSIONS_DIR / "auto_experiment_suite"

RESULTS_PATH = OUTPUT_DIR / "auto_experiment_results.csv"
QUALIFIED_PATH = OUTPUT_DIR / "auto_experiment_qualified.csv"
GUARD_PATH = OUTPUT_DIR / "auto_experiment_guard_summary.csv"
REPORT_PATH = OUTPUT_DIR / "auto_experiment_suite_report.md"

CURRENT_BEST_PUBLIC_LB = 0.77272
CURRENT_BEST_CV_MEAN = 0.83725

DEFAULT_SEEDS = [42, 7, 123]
DEFAULT_N_SPLITS = 5

DEFAULT_MIN_CV_MEAN = 0.838
DEFAULT_MAX_CV_STD = 0.018
DEFAULT_MIN_RISK_ADJUSTED = 0.820
DEFAULT_MAX_TOTAL_CHANGED = 12
DEFAULT_MAX_0_TO_1 = 6
DEFAULT_MAX_PCLASS3_FEMALE_0_TO_1 = 3
DEFAULT_MAX_MISS_MRS_0_TO_1 = 4


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    numeric_features: list[str]
    categorical_features: list[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "numeric_features", unique_columns(self.numeric_features))
        object.__setattr__(self, "categorical_features", unique_columns(self.categorical_features))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    scale_numeric: bool = False


def unique_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for column in columns:
        if column not in seen:
            seen.add(column)
            result.append(column)

    return result


@dataclass(frozen=True)
class AutoExperimentResult:
    experiment_id: str
    feature_set_name: str
    model_name: str
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    risk_adjusted_score: float
    submission_path: Path
    changed_count: int
    total_0_to_1: int
    total_1_to_0: int
    pclass3_female_0_to_1: int
    miss_mrs_0_to_1: int
    guard_decision: str
    guard_reason: str
    qualifies: bool
    qualify_reason: str


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    return train, test


def extract_surname(name: str) -> str:
    return str(name).split(",", maxsplit=1)[0].strip()


def extract_raw_title(name: str) -> str:
    import re

    match = re.search(r",\s*([^\.]+)\.", str(name))
    if match is None:
        return "Unknown"

    return match.group(1).strip()


def extract_text_after_title(name: str) -> str:
    import re

    match = re.search(r"\.\s*(.*)$", str(name))
    if match is None:
        return ""

    return match.group(1).strip()


def extract_parentheses_text(name: str) -> str:
    import re

    match = re.search(r"\((.*?)\)", str(name))
    if match is None:
        return ""

    return match.group(1).strip()


def map_title_group(raw_title: str) -> str:
    normalized = raw_title.strip()

    replacements = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }

    normalized = replacements.get(normalized, normalized)

    if normalized == "Mr":
        return "CommonMale"

    if normalized in {"Mrs", "Miss"}:
        return "CommonFemale"

    if normalized == "Master":
        return "ChildMale"

    if normalized == "Rev":
        return "Clergy"

    if normalized == "Dr":
        return "Professional"

    if normalized in {"Col", "Major", "Capt"}:
        return "Military"

    if normalized in {"Lady", "Countess", "Dona"}:
        return "NobleFemale"

    if normalized in {"Sir", "Don", "Jonkheer", "the Countess"}:
        return "Noble"

    return "RareOther"


def add_safe_family_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_featured = add_feature_v1_columns(train)
    test_featured = add_feature_v1_columns(test)

    train_featured["_dataset"] = "train"
    test_featured["_dataset"] = "test"

    combined = pd.concat([train_featured, test_featured], ignore_index=True, sort=False)

    combined["Surname"] = combined["Name"].astype(str).apply(extract_surname)
    combined["RawTitle"] = combined["Name"].astype(str).apply(extract_raw_title)
    combined["TitleGroup"] = combined["RawTitle"].apply(map_title_group)
    combined["ParenthesesText"] = combined["Name"].astype(str).apply(extract_parentheses_text)
    combined["HasParenthesesName"] = combined["ParenthesesText"].ne("")
    combined["HasQuotedNickname"] = combined["Name"].astype(str).str.contains('"', regex=False)

    surname_counts = combined["Surname"].value_counts()
    surname_pclass_counts = combined.groupby(["Surname", "Pclass"]).size()
    surname_ticket_counts = combined.groupby(["Surname", "Ticket"]).size()

    combined["SurnameGroupSize"] = combined["Surname"].map(surname_counts).astype(int)
    combined["SurnamePclassGroupSize"] = [
        int(surname_pclass_counts.loc[(row["Surname"], row["Pclass"])])
        for _, row in combined.iterrows()
    ]
    combined["SurnameTicketGroupSize"] = [
        int(surname_ticket_counts.loc[(row["Surname"], row["Ticket"])])
        for _, row in combined.iterrows()
    ]

    combined["IsSoloSurname"] = combined["SurnameGroupSize"] == 1
    combined["IsSmallSurnameGroup"] = combined["SurnameGroupSize"].between(2, 3)
    combined["IsLargeSurnameGroup"] = combined["SurnameGroupSize"] >= 4
    combined["IsSmallSurnameTicketGroup"] = combined["SurnameTicketGroupSize"].between(2, 4)
    combined["IsLargeSurnameTicketGroup"] = combined["SurnameTicketGroupSize"] >= 5

    train_out = combined[combined["_dataset"] == "train"].drop(columns=["_dataset"]).copy()
    test_out = combined[combined["_dataset"] == "test"].drop(columns=["_dataset"]).copy()

    return train_out, test_out



def sanitize_categorical_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_sets: list[FeatureSetSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert all categorical columns used by any feature set to plain strings.

    Some sklearn versions dislike mixed boolean/object categorical blocks when
    SimpleImputer(strategy="most_frequent") receives them together. We avoid
    that by normalizing all categorical features to strings before CV/submission.
    """

    train_out = train_df.copy()
    test_out = test_df.copy()

    categorical_columns = sorted(
        {
            column
            for feature_set in feature_sets
            for column in feature_set.categorical_features
        }
    )

    for column in categorical_columns:
        if column in train_out.columns:
            train_out[column] = train_out[column].astype("string").fillna("MISSING").astype(str)

        if column in test_out.columns:
            test_out[column] = test_out[column].astype("string").fillna("MISSING").astype(str)

    return train_out, test_out


def build_feature_sets() -> list[FeatureSetSpec]:
    base_numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    base_categorical = ["Sex", "Embarked", "Title"]

    family_numeric = [
        "SurnameGroupSize",
        "SurnamePclassGroupSize",
        "SurnameTicketGroupSize",
    ]
    family_categorical = [
        "IsSoloSurname",
        "IsSmallSurnameGroup",
        "IsLargeSurnameGroup",
        "IsSmallSurnameTicketGroup",
        "IsLargeSurnameTicketGroup",
    ]

    name_categorical = [
        "HasParenthesesName",
        "HasQuotedNickname",
    ]

    return [
        FeatureSetSpec(
            name="current_best_plus_title",
            numeric_features=base_numeric,
            categorical_features=base_categorical,
        ),
        FeatureSetSpec(
            name="safe_family_counts",
            numeric_features=base_numeric + family_numeric,
            categorical_features=base_categorical,
        ),
        FeatureSetSpec(
            name="safe_family_flags",
            numeric_features=base_numeric,
            categorical_features=base_categorical + family_categorical,
        ),
        FeatureSetSpec(
            name="safe_family_counts_flags",
            numeric_features=base_numeric + family_numeric,
            categorical_features=base_categorical + family_categorical,
        ),
        FeatureSetSpec(
            name="title_group_family_counts",
            numeric_features=base_numeric + family_numeric,
            categorical_features=["Sex", "Embarked", "TitleGroup"] + family_categorical,
        ),
        FeatureSetSpec(
            name="name_structure_family",
            numeric_features=base_numeric + family_numeric,
            categorical_features=["Sex", "Embarked", "TitleGroup"] + family_categorical + name_categorical,
        ),
    ]


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="hgb_reference",
            estimator=HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=0.0,
                random_state=42,
            ),
        ),
        ModelSpec(
            name="hgb_small_regular",
            estimator=HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.04,
                max_leaf_nodes=7,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=42,
            ),
        ),
        ModelSpec(
            name="hgb_large_regular",
            estimator=HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.04,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=0.1,
                random_state=42,
            ),
        ),
        ModelSpec(
            name="hgb_conservative",
            estimator=HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.03,
                max_leaf_nodes=7,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
        ModelSpec(
            name="random_forest_conservative",
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            name="logistic_l2",
            estimator=LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
            ),
            scale_numeric=True,
        ),
        ModelSpec(
            name="logistic_strong_l2",
            estimator=LogisticRegression(
                max_iter=1000,
                C=0.3,
                solver="lbfgs",
            ),
            scale_numeric=True,
        ),
    ]


def build_preprocessor(feature_set: FeatureSetSpec, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, feature_set.numeric_features),
            ("categorical", categorical_pipeline, feature_set.categorical_features),
        ],
        remainder="drop",
    )


def build_pipeline(feature_set: FeatureSetSpec, model_spec: ModelSpec) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_set, model_spec.scale_numeric)),
            ("model", model_spec.estimator),
        ]
    )


def calculate_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def calculate_std(values: list[float]) -> float:
    mean_value = calculate_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)

    return math.sqrt(variance)


def evaluate_repeated_cv(
    train_df: pd.DataFrame,
    feature_set: FeatureSetSpec,
    model_spec: ModelSpec,
    seeds: list[int],
    n_splits: int,
) -> list[float]:
    selected_features = unique_columns(feature_set.numeric_features + feature_set.categorical_features)
    X = train_df[selected_features].copy()
    y = train_df["Survived"].copy()

    scores: list[float] = []

    for seed in seeds:
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        for train_index, valid_index in cv.split(X, y):
            X_train = X.iloc[train_index].copy()
            X_valid = X.iloc[valid_index].copy()
            y_train = y.iloc[train_index].copy()
            y_valid = y.iloc[valid_index].copy()

            pipeline = build_pipeline(feature_set, model_spec)
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_valid)
            scores.append(float(accuracy_score(y_valid, predictions)))

    return scores


def create_submission_for_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set: FeatureSetSpec,
    model_spec: ModelSpec,
    experiment_id: str,
) -> Path:
    selected_features = unique_columns(feature_set.numeric_features + feature_set.categorical_features)
    X = train_df[selected_features].copy()
    y = train_df["Survived"].copy()
    X_test = test_df[selected_features].copy()

    pipeline = build_pipeline(feature_set, model_spec)
    pipeline.fit(X, y)

    predictions = [int(value) for value in pipeline.predict(X_test)]

    submission = build_submission(
        test_raw=test_df,
        predictions=predictions,
    )
    validate_submission(submission, expected_rows=len(test_df))

    SUBMISSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUBMISSION_OUTPUT_DIR / f"{experiment_id}_submission.csv"
    submission.to_csv(submission_path, index=False)

    return submission_path


def run_guard_for_submission(
    test_df: pd.DataFrame,
    submission_path: Path,
    experiment_id: str,
) -> tuple[int, int, int, int, int, str, str]:
    best_submission = load_submission(CURRENT_BEST_SUBMISSION_PATH, "Survived_best")
    candidate_submission = load_submission(submission_path, f"Survived_{experiment_id}")

    changed = build_flip_rows(
        test_features=test_df,
        best_submission=best_submission,
        candidate_submission=candidate_submission,
        candidate_name=experiment_id,
    )

    summary = summarize_guard(
        changed=changed,
        candidate_name=experiment_id,
        candidate_path=submission_path,
        test_row_count=len(test_df),
    )

    return (
        summary.changed_count,
        summary.total_0_to_1,
        summary.total_1_to_0,
        summary.pclass3_female_0_to_1,
        summary.miss_mrs_0_to_1,
        summary.decision,
        summary.reason,
    )


def qualify_result(
    cv_mean: float,
    cv_std: float,
    risk_adjusted_score: float,
    changed_count: int,
    total_0_to_1: int,
    pclass3_female_0_to_1: int,
    miss_mrs_0_to_1: int,
    guard_decision: str,
    min_cv_mean: float,
    max_cv_std: float,
    min_risk_adjusted: float,
    max_total_changed: int,
    max_0_to_1: int,
    max_pclass3_female_0_to_1: int,
    max_miss_mrs_0_to_1: int,
) -> tuple[bool, str]:
    reasons = []

    if cv_mean < min_cv_mean:
        reasons.append(f"cv_mean {cv_mean:.5f} < {min_cv_mean:.5f}")

    if cv_std > max_cv_std:
        reasons.append(f"cv_std {cv_std:.5f} > {max_cv_std:.5f}")

    if risk_adjusted_score < min_risk_adjusted:
        reasons.append(f"risk_adjusted {risk_adjusted_score:.5f} < {min_risk_adjusted:.5f}")

    if changed_count > max_total_changed:
        reasons.append(f"changed_count {changed_count} > {max_total_changed}")

    if total_0_to_1 > max_0_to_1:
        reasons.append(f"0_to_1 {total_0_to_1} > {max_0_to_1}")

    if pclass3_female_0_to_1 > max_pclass3_female_0_to_1:
        reasons.append(
            f"pclass3_female_0_to_1 {pclass3_female_0_to_1} > {max_pclass3_female_0_to_1}"
        )

    if miss_mrs_0_to_1 > max_miss_mrs_0_to_1:
        reasons.append(f"miss_mrs_0_to_1 {miss_mrs_0_to_1} > {max_miss_mrs_0_to_1}")

    if guard_decision == "DO_NOT_SUBMIT_WITHOUT_REVIEW":
        reasons.append("guard_decision blocks submission")

    if guard_decision == "NO_NEW_INFORMATION":
        reasons.append("submission identical to current best")

    if reasons:
        return False, "; ".join(reasons)

    return True, "passed all thresholds"


def result_to_row(result: AutoExperimentResult) -> dict[str, object]:
    return {
        "experiment_id": result.experiment_id,
        "feature_set_name": result.feature_set_name,
        "model_name": result.model_name,
        "cv_mean": result.cv_mean,
        "cv_std": result.cv_std,
        "risk_adjusted_score": result.risk_adjusted_score,
        "cv_scores": ", ".join(f"{score:.5f}" for score in result.cv_scores),
        "submission_path": str(result.submission_path),
        "changed_count": result.changed_count,
        "total_0_to_1": result.total_0_to_1,
        "total_1_to_0": result.total_1_to_0,
        "pclass3_female_0_to_1": result.pclass3_female_0_to_1,
        "miss_mrs_0_to_1": result.miss_mrs_0_to_1,
        "guard_decision": result.guard_decision,
        "guard_reason": result.guard_reason,
        "qualifies": result.qualifies,
        "qualify_reason": result.qualify_reason,
    }


def run_kaggle_submit(submission_path: Path, message: str, dry_run: bool) -> None:
    command = [
        "kaggle",
        "competitions",
        "submit",
        "titanic",
        "-f",
        str(submission_path),
        "-m",
        message,
    ]

    if dry_run:
        print("[DRY RUN]", " ".join(command))
        return

    subprocess.run(command, check=True)


def build_report(results: list[AutoExperimentResult], submitted_ids: list[str]) -> str:
    results_df = pd.DataFrame([result_to_row(result) for result in results])
    top_df = results_df.sort_values(
        ["qualifies", "risk_adjusted_score", "cv_mean"],
        ascending=[False, False, False],
    ).head(20)

    qualified_df = results_df[results_df["qualifies"] == True]  # noqa: E712

    def df_to_md(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"

        selected = df[
            [
                "experiment_id",
                "feature_set_name",
                "model_name",
                "cv_mean",
                "cv_std",
                "risk_adjusted_score",
                "changed_count",
                "total_0_to_1",
                "guard_decision",
                "qualifies",
            ]
        ].copy()

        lines = [
            "| " + " | ".join(selected.columns) + " |",
            "| " + " | ".join("---" for _ in selected.columns) + " |",
        ]

        for _, row in selected.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in selected.columns) + " |")

        return "\\n".join(lines)

    submitted_text = "\\n".join(f"- {experiment_id}" for experiment_id in submitted_ids) if submitted_ids else "_None._"

    return f"""# Titanic Auto Experiment Suite Report

## Goal

Run a controlled suite of Titanic experiments:

1. Multiple safe feature sets.
2. Multiple model/parameter configurations.
3. Repeated CV across multiple seeds.
4. Current-best flip guard.
5. Optional Kaggle submission for only threshold-qualified candidates.

## Threshold Logic

```text
min_cv_mean
max_cv_std
min_risk_adjusted
max_total_changed
max_0_to_1
max_pclass3_female_0_to_1
max_miss_mrs_0_to_1
guard_decision != DO_NOT_SUBMIT_WITHOUT_REVIEW
guard_decision != NO_NEW_INFORMATION
```

## Top Results

{df_to_md(top_df)}

## Qualified Results

{df_to_md(qualified_df)}

## Submitted Experiment IDs

{submitted_text}

## Output Files

- `{RESULTS_PATH}`
- `{QUALIFIED_PATH}`
- `{GUARD_PATH}`
- `{REPORT_PATH}`
- `{SUBMISSION_OUTPUT_DIR}`

## Important Warning

This suite can submit to Kaggle if `--submit-qualified` is used.

Do not use that flag until you inspect the generated CSV/report.
"""


def run_suite(
    min_cv_mean: float,
    max_cv_std: float,
    min_risk_adjusted: float,
    max_total_changed: int,
    max_0_to_1: int,
    max_pclass3_female_0_to_1: int,
    max_miss_mrs_0_to_1: int,
    seeds: list[int],
    n_splits: int,
    max_experiments: int | None,
    submit_qualified: bool,
    dry_run_submit: bool,
) -> list[AutoExperimentResult]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_raw_data()
    train_df, test_df = add_safe_family_features(train_raw, test_raw)

    feature_sets = build_feature_sets()
    train_df, test_df = sanitize_categorical_columns(train_df, test_df, feature_sets)

    model_specs = build_model_specs()

    experiment_pairs = [
        (feature_set, model_spec)
        for feature_set in feature_sets
        for model_spec in model_specs
    ]

    if max_experiments is not None:
        experiment_pairs = experiment_pairs[:max_experiments]

    results: list[AutoExperimentResult] = []
    guard_rows: list[dict[str, object]] = []
    submitted_ids: list[str] = []

    for index, (feature_set, model_spec) in enumerate(experiment_pairs, start=1):
        experiment_id = f"auto_{index:03d}_{feature_set.name}_{model_spec.name}"

        print(f"=== Running {experiment_id} ===")

        cv_scores = evaluate_repeated_cv(
            train_df=train_df,
            feature_set=feature_set,
            model_spec=model_spec,
            seeds=seeds,
            n_splits=n_splits,
        )
        cv_mean = calculate_mean(cv_scores)
        cv_std = calculate_std(cv_scores)
        risk_adjusted_score = cv_mean - cv_std

        submission_path = create_submission_for_experiment(
            train_df=train_df,
            test_df=test_df,
            feature_set=feature_set,
            model_spec=model_spec,
            experiment_id=experiment_id,
        )

        (
            changed_count,
            total_0_to_1,
            total_1_to_0,
            pclass3_female_0_to_1,
            miss_mrs_0_to_1,
            guard_decision,
            guard_reason,
        ) = run_guard_for_submission(
            test_df=test_df,
            submission_path=submission_path,
            experiment_id=experiment_id,
        )

        qualifies, qualify_reason = qualify_result(
            cv_mean=cv_mean,
            cv_std=cv_std,
            risk_adjusted_score=risk_adjusted_score,
            changed_count=changed_count,
            total_0_to_1=total_0_to_1,
            pclass3_female_0_to_1=pclass3_female_0_to_1,
            miss_mrs_0_to_1=miss_mrs_0_to_1,
            guard_decision=guard_decision,
            min_cv_mean=min_cv_mean,
            max_cv_std=max_cv_std,
            min_risk_adjusted=min_risk_adjusted,
            max_total_changed=max_total_changed,
            max_0_to_1=max_0_to_1,
            max_pclass3_female_0_to_1=max_pclass3_female_0_to_1,
            max_miss_mrs_0_to_1=max_miss_mrs_0_to_1,
        )

        result = AutoExperimentResult(
            experiment_id=experiment_id,
            feature_set_name=feature_set.name,
            model_name=model_spec.name,
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            risk_adjusted_score=risk_adjusted_score,
            submission_path=submission_path,
            changed_count=changed_count,
            total_0_to_1=total_0_to_1,
            total_1_to_0=total_1_to_0,
            pclass3_female_0_to_1=pclass3_female_0_to_1,
            miss_mrs_0_to_1=miss_mrs_0_to_1,
            guard_decision=guard_decision,
            guard_reason=guard_reason,
            qualifies=qualifies,
            qualify_reason=qualify_reason,
        )

        print(
            f"cv_mean={cv_mean:.5f} cv_std={cv_std:.5f} "
            f"risk={risk_adjusted_score:.5f} changed={changed_count} "
            f"0_to_1={total_0_to_1} guard={guard_decision} qualifies={qualifies}"
        )

        results.append(result)
        guard_rows.append(result_to_row(result))

        pd.DataFrame([result_to_row(item) for item in results]).to_csv(RESULTS_PATH, index=False)
        pd.DataFrame([result_to_row(item) for item in results if item.qualifies]).to_csv(QUALIFIED_PATH, index=False)
        pd.DataFrame(guard_rows).to_csv(GUARD_PATH, index=False)

        if qualifies and submit_qualified:
            message = f"{experiment_id} auto suite qualified"
            run_kaggle_submit(submission_path, message, dry_run=dry_run_submit)
            submitted_ids.append(experiment_id)

    REPORT_PATH.write_text(build_report(results, submitted_ids), encoding="utf-8")

    return results


def parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cv-mean", type=float, default=DEFAULT_MIN_CV_MEAN)
    parser.add_argument("--max-cv-std", type=float, default=DEFAULT_MAX_CV_STD)
    parser.add_argument("--min-risk-adjusted", type=float, default=DEFAULT_MIN_RISK_ADJUSTED)
    parser.add_argument("--max-total-changed", type=int, default=DEFAULT_MAX_TOTAL_CHANGED)
    parser.add_argument("--max-0-to-1", type=int, default=DEFAULT_MAX_0_TO_1)
    parser.add_argument("--max-pclass3-female-0-to-1", type=int, default=DEFAULT_MAX_PCLASS3_FEMALE_0_TO_1)
    parser.add_argument("--max-miss-mrs-0-to-1", type=int, default=DEFAULT_MAX_MISS_MRS_0_TO_1)
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--submit-qualified", action="store_true")
    parser.add_argument("--dry-run-submit", action="store_true")
    args = parser.parse_args()

    results = run_suite(
        min_cv_mean=args.min_cv_mean,
        max_cv_std=args.max_cv_std,
        min_risk_adjusted=args.min_risk_adjusted,
        max_total_changed=args.max_total_changed,
        max_0_to_1=args.max_0_to_1,
        max_pclass3_female_0_to_1=args.max_pclass3_female_0_to_1,
        max_miss_mrs_0_to_1=args.max_miss_mrs_0_to_1,
        seeds=parse_seeds(args.seeds),
        n_splits=args.n_splits,
        max_experiments=args.max_experiments,
        submit_qualified=args.submit_qualified,
        dry_run_submit=args.dry_run_submit,
    )

    qualified = [result for result in results if result.qualifies]

    print()
    print("=== Auto Experiment Suite Finished ===")
    print(f"Total experiments: {len(results)}")
    print(f"Qualified experiments: {len(qualified)}")
    print(f"Saved results to: {RESULTS_PATH}")
    print(f"Saved qualified to: {QUALIFIED_PATH}")
    print(f"Saved guard summary to: {GUARD_PATH}")
    print(f"Saved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
