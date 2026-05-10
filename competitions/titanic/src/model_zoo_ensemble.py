from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]
from sklearn.naive_bayes import GaussianNB  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # type: ignore[import-untyped]
from sklearn.svm import SVC  # type: ignore[import-untyped]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from competitions.titanic.src.feature_v1 import add_feature_v1_columns
from competitions.titanic.src.model_comparison import build_submission, validate_submission


RAW_DATA_DIR = Path("competitions/titanic/data/raw")
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"

REPORTS_DIR = Path("competitions/titanic/reports")
SUBMISSIONS_DIR = Path("competitions/titanic/submissions")

RESULTS_PATH = REPORTS_DIR / "titanic_model_zoo_ensemble_results.csv"
REPORT_PATH = REPORTS_DIR / "titanic_model_zoo_ensemble_report.md"
BEST_SUBMISSION_PATH = SUBMISSIONS_DIR / "model_zoo_ensemble_best_submission.csv"

TARGET_COLUMN = "Survived"
RANDOM_STATE = 42
N_SPLITS = 5
CURRENT_BEST_PUBLIC_LB = 0.77272


@dataclass(frozen=True)
class FeatureSet:
    name: str
    numeric_features: list[str]
    categorical_features: list[str]
    ordinal_features: list[str]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    numeric_transform: str


@dataclass(frozen=True)
class EvaluationResult:
    candidate_name: str
    feature_set_name: str
    model_name: str
    candidate_type: str
    cv_scores: list[float]
    cv_mean: float
    cv_std: float
    delta_vs_current_best_public: float


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def extract_raw_title(name: object) -> str:
    match = re.search(r",\s*([^\.]+)\.", str(name))
    if match is None:
        return "Unknown"
    return match.group(1).strip()


def map_title_v2(raw_title: object) -> str:
    normalized = str(raw_title).strip()
    normalized = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}.get(normalized, normalized)
    if normalized in {"Mr", "Mrs", "Miss", "Master"}:
        return normalized
    if normalized == "Rev":
        return "Clergy"
    if normalized == "Dr":
        return "Professional"
    if normalized in {"Col", "Major", "Capt"}:
        return "Military"
    if normalized in {"Don", "Sir", "Jonkheer"}:
        return "NobleMale"
    if normalized in {"Lady", "Countess", "Dona"}:
        return "NobleFemale"
    return "RareOther"


def extract_ticket_number(ticket: object) -> str:
    parts = str(ticket).strip().split()
    if not parts:
        return "UNKNOWN"
    return parts[-1]


def ticket_has_letters(ticket: object) -> int:
    return int(any(char.isalpha() for char in str(ticket)))


def add_name_length_group(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["NameLength"] = result["Name"].astype(str).str.len()
    result["NameLengthGroup"] = pd.cut(
        result["NameLength"],
        bins=[0, 25, 45, 100],
        labels=["short", "medium", "long"],
        include_lowest=True,
    ).astype(str)
    return result


def add_competition_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_base = add_feature_v1_columns(train_raw).copy()
    test_base = add_feature_v1_columns(test_raw).copy()
    train_base["_dataset"] = "train"
    test_base["_dataset"] = "test"
    combined = pd.concat([train_base, test_base], ignore_index=True, sort=False)

    combined["FamilySize"] = combined["SibSp"] + combined["Parch"] + 1
    combined["FamilySizeGrouped"] = pd.cut(
        combined["FamilySize"],
        bins=[0, 1, 4, 6, 20],
        labels=["alone", "small", "medium", "large"],
        include_lowest=True,
    ).astype(str)

    combined["AgeFilledForBin"] = combined["Age"].fillna(combined["Age"].median())
    combined["FareFilledForBin"] = combined["Fare"].fillna(combined["Fare"].median())
    combined["AgeBin5"] = pd.qcut(
        combined["AgeFilledForBin"], q=5, labels=False, duplicates="drop"
    ).astype(str)
    combined["FareBin5"] = pd.qcut(
        combined["FareFilledForBin"], q=5, labels=False, duplicates="drop"
    ).astype(str)

    combined = add_name_length_group(combined)
    combined["RawTitle"] = combined["Name"].apply(extract_raw_title)
    combined["TitleV2"] = combined["RawTitle"].apply(map_title_v2)
    combined["TicketNumber"] = combined["Ticket"].apply(extract_ticket_number)
    ticket_counts = combined["TicketNumber"].value_counts()
    combined["TicketNumberCount"] = combined["TicketNumber"].map(ticket_counts).astype(int)
    combined["TicketHasLetters"] = combined["Ticket"].apply(ticket_has_letters)
    combined["CabinAssigned"] = combined["Cabin"].notna().astype(int)
    combined["CabinLetter"] = combined["Cabin"].fillna("U").astype(str).str[0].str.upper()
    combined["Sex_Pclass"] = combined["Sex"].astype(str) + "_P" + combined["Pclass"].astype(str)
    combined["Title_Pclass"] = combined["Title"].astype(str) + "_P" + combined["Pclass"].astype(str)

    train_result = combined[combined["_dataset"] == "train"].drop(columns=["_dataset"]).copy()
    test_result = combined[combined["_dataset"] == "test"].drop(columns=["_dataset"]).copy()
    return train_result, test_result


def build_feature_sets() -> list[FeatureSet]:
    return [
        FeatureSet(
            name="plus_title",
            numeric_features=["Pclass", "Age", "SibSp", "Parch", "Fare"],
            categorical_features=["Sex", "Embarked", "Title"],
            ordinal_features=[],
        ),
        FeatureSet(
            name="binned_competition",
            numeric_features=["Pclass", "TicketNumberCount", "CabinAssigned", "TicketHasLetters"],
            categorical_features=[
                "Sex", "Embarked", "Title", "AgeBin5", "FareBin5", "FamilySizeGrouped", "NameLengthGroup",
            ],
            ordinal_features=[],
        ),
        FeatureSet(
            name="wide_competition",
            numeric_features=[
                "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "TicketNumberCount",
                "CabinAssigned", "TicketHasLetters", "NameLength",
            ],
            categorical_features=[
                "Sex", "Embarked", "Title", "TitleV2", "AgeBin5", "FareBin5", "FamilySizeGrouped",
                "NameLengthGroup", "CabinLetter", "Sex_Pclass", "Title_Pclass",
            ],
            ordinal_features=[],
        ),
        FeatureSet(
            name="nb_competition_ordinal",
            numeric_features=["Pclass", "Age", "Fare", "FamilySize", "TicketNumberCount", "CabinAssigned", "NameLength"],
            categorical_features=[],
            ordinal_features=["Sex", "Embarked", "Title", "AgeBin5", "FareBin5", "FamilySizeGrouped", "NameLengthGroup"],
        ),
    ]


def get_all_feature_columns(feature_set: FeatureSet) -> list[str]:
    return feature_set.numeric_features + feature_set.categorical_features + feature_set.ordinal_features


def build_preprocessor(feature_set: FeatureSet, model_spec: ModelSpec) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if feature_set.numeric_features:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if model_spec.numeric_transform == "scaled":
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps=numeric_steps), feature_set.numeric_features))
    if feature_set.categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                feature_set.categorical_features,
            )
        )
    if feature_set.ordinal_features:
        transformers.append(
            (
                "ordinal",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                feature_set.ordinal_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec("gaussian_nb", GaussianNB(), "scaled"),
        ModelSpec("logistic_regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), "scaled"),
        ModelSpec("knn", KNeighborsClassifier(n_neighbors=7), "scaled"),
        ModelSpec("svc_rbf", SVC(C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE), "scaled"),
        ModelSpec("decision_tree", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), "raw"),
        ModelSpec(
            "random_forest",
            RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE),
            "raw",
        ),
        ModelSpec(
            "extra_trees",
            ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE),
            "raw",
        ),
        ModelSpec("adaboost", AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE), "raw"),
        ModelSpec(
            "gradient_boosting",
            GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
            "raw",
        ),
        ModelSpec(
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=RANDOM_STATE),
            "raw",
        ),
    ]
    try:
        from xgboost import XGBClassifier
        specs.append(
            ModelSpec(
                "xgboost",
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                ),
                "raw",
            )
        )
    except Exception:
        pass
    return specs


def build_pipeline(feature_set: FeatureSet, model_spec: ModelSpec) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_set, model_spec)),
            ("model", clone(model_spec.estimator)),
        ]
    )


def evaluate_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    candidate_name: str,
    feature_set_name: str,
    model_name: str,
    candidate_type: str,
    n_splits: int = N_SPLITS,
) -> EvaluationResult:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    for train_index, valid_index in cv.split(X, y):
        X_train = X.iloc[train_index].copy()
        X_valid = X.iloc[valid_index].copy()
        y_train = y.iloc[train_index].copy()
        y_valid = y.iloc[valid_index].copy()
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        predictions = fold_pipeline.predict(X_valid)
        scores.append(float(accuracy_score(y_valid, predictions)))
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_score = variance ** 0.5
    return EvaluationResult(
        candidate_name=candidate_name,
        feature_set_name=feature_set_name,
        model_name=model_name,
        candidate_type=candidate_type,
        cv_scores=scores,
        cv_mean=mean_score,
        cv_std=std_score,
        delta_vs_current_best_public=mean_score - CURRENT_BEST_PUBLIC_LB,
    )


def build_training_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw, test_raw = load_raw_data()
    return add_competition_features(train_raw, test_raw)


def evaluate_model_zoo(
    train_df: pd.DataFrame,
    feature_sets: list[FeatureSet],
    model_specs: list[ModelSpec],
) -> list[EvaluationResult]:
    y = train_df[TARGET_COLUMN].copy()
    results: list[EvaluationResult] = []
    for feature_set in feature_sets:
        X = train_df[get_all_feature_columns(feature_set)].copy()
        for model_spec in model_specs:
            pipeline = build_pipeline(feature_set, model_spec)
            candidate_name = f"{feature_set.name}__{model_spec.name}"
            result = evaluate_pipeline(
                X=X,
                y=y,
                pipeline=pipeline,
                candidate_name=candidate_name,
                feature_set_name=feature_set.name,
                model_name=model_spec.name,
                candidate_type="single_model",
            )
            results.append(result)
    return results


def get_model_spec(model_specs: list[ModelSpec], name: str) -> ModelSpec:
    for model_spec in model_specs:
        if model_spec.name == name:
            return model_spec
    raise ValueError(f"Unknown model spec: {name}")


def get_feature_set(feature_sets: list[FeatureSet], name: str) -> FeatureSet:
    for feature_set in feature_sets:
        if feature_set.name == name:
            return feature_set
    raise ValueError(f"Unknown feature set: {name}")


def build_voting_pipeline(
    feature_set: FeatureSet,
    model_specs: list[ModelSpec],
    voting: str,
    weights: list[int] | None = None,
) -> Pipeline:
    estimators = []
    for model_spec in model_specs:
        estimator_pipeline = build_pipeline(feature_set, model_spec)
        estimators.append((model_spec.name, estimator_pipeline))
    voter = VotingClassifier(estimators=estimators, voting=voting, weights=weights)
    return Pipeline(steps=[("voter", voter)])


def evaluate_voting_candidates(
    train_df: pd.DataFrame,
    feature_sets: list[FeatureSet],
    model_specs: list[ModelSpec],
    single_results: list[EvaluationResult],
) -> list[EvaluationResult]:
    y = train_df[TARGET_COLUMN].copy()
    voting_results: list[EvaluationResult] = []
    for feature_set in feature_sets:
        feature_results = [result for result in single_results if result.feature_set_name == feature_set.name]
        top_model_names = [result.model_name for result in sorted(feature_results, key=lambda item: item.cv_mean, reverse=True)[:5]]
        X = train_df[get_all_feature_columns(feature_set)].copy()
        if len(top_model_names) >= 3:
            top3_specs = [get_model_spec(model_specs, name) for name in top_model_names[:3]]
            voting_pipeline = build_voting_pipeline(feature_set=feature_set, model_specs=top3_specs, voting="hard")
            voting_results.append(
                evaluate_pipeline(
                    X=X,
                    y=y,
                    pipeline=voting_pipeline,
                    candidate_name=f"{feature_set.name}__top3_hard_voting",
                    feature_set_name=feature_set.name,
                    model_name="top3_hard_voting",
                    candidate_type="voting",
                )
            )
        if len(top_model_names) >= 5:
            top5_specs = [get_model_spec(model_specs, name) for name in top_model_names[:5]]
            voting_pipeline = build_voting_pipeline(feature_set=feature_set, model_specs=top5_specs, voting="hard")
            voting_results.append(
                evaluate_pipeline(
                    X=X,
                    y=y,
                    pipeline=voting_pipeline,
                    candidate_name=f"{feature_set.name}__top5_hard_voting",
                    feature_set_name=feature_set.name,
                    model_name="top5_hard_voting",
                    candidate_type="voting",
                )
            )
        preferred_names = [name for name in ["adaboost", "extra_trees", "gaussian_nb"] if any(spec.name == name for spec in model_specs)]
        if len(preferred_names) == 3:
            preferred_specs = [get_model_spec(model_specs, name) for name in preferred_names]
            voting_pipeline = build_voting_pipeline(
                feature_set=feature_set,
                model_specs=preferred_specs,
                voting="hard",
                weights=[1, 2, 3],
            )
            voting_results.append(
                evaluate_pipeline(
                    X=X,
                    y=y,
                    pipeline=voting_pipeline,
                    candidate_name=f"{feature_set.name}__adaboost_extra_nb_hard_weighted",
                    feature_set_name=feature_set.name,
                    model_name="adaboost_extra_nb_hard_weighted",
                    candidate_type="voting",
                )
            )
    return voting_results


def results_to_dataframe(results: list[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_name": result.candidate_name,
                "feature_set_name": result.feature_set_name,
                "model_name": result.model_name,
                "candidate_type": result.candidate_type,
                "cv_mean": result.cv_mean,
                "cv_std": result.cv_std,
                "delta_vs_current_best_public": result.delta_vs_current_best_public,
                "fold_scores": ", ".join(f"{score:.5f}" for score in result.cv_scores),
            }
            for result in sorted(results, key=lambda item: item.cv_mean, reverse=True)
        ]
    )


def find_best_result(results: list[EvaluationResult]) -> EvaluationResult:
    return sorted(results, key=lambda item: (item.cv_mean, -item.cv_std), reverse=True)[0]


def build_final_pipeline(
    best: EvaluationResult,
    feature_set: FeatureSet,
    model_specs: list[ModelSpec],
    all_results: list[EvaluationResult],
) -> Pipeline:
    if best.candidate_type == "single_model":
        return build_pipeline(feature_set, get_model_spec(model_specs, best.model_name))
    if best.model_name == "adaboost_extra_nb_hard_weighted":
        specs = [get_model_spec(model_specs, name) for name in ["adaboost", "extra_trees", "gaussian_nb"]]
        return build_voting_pipeline(feature_set, specs, voting="hard", weights=[1, 2, 3])
    feature_results = [
        result for result in all_results
        if result.feature_set_name == feature_set.name and result.candidate_type == "single_model"
    ]
    top_n = 3 if best.model_name == "top3_hard_voting" else 5
    top_names = [result.model_name for result in sorted(feature_results, key=lambda item: item.cv_mean, reverse=True)[:top_n]]
    specs = [get_model_spec(model_specs, name) for name in top_names]
    return build_voting_pipeline(feature_set, specs, voting="hard")


def train_best_and_save_submission(
    best: EvaluationResult,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_sets: list[FeatureSet],
    model_specs: list[ModelSpec],
    all_results: list[EvaluationResult],
) -> None:
    feature_set = get_feature_set(feature_sets, best.feature_set_name)
    X = train_df[get_all_feature_columns(feature_set)].copy()
    y = train_df[TARGET_COLUMN].copy()
    X_test = test_df[get_all_feature_columns(feature_set)].copy()
    pipeline = build_final_pipeline(best, feature_set, model_specs, all_results)
    pipeline.fit(X, y)
    predictions = [int(value) for value in pipeline.predict(X_test)]
    _, test_raw = load_raw_data()
    submission = build_submission(test_raw=test_raw, predictions=predictions)
    validate_submission(submission, expected_rows=len(test_raw))
    BEST_SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(BEST_SUBMISSION_PATH, index=False)


def build_report(results: list[EvaluationResult], best: EvaluationResult) -> str:
    table_lines = [
        "| Rank | Candidate | Type | Feature set | Model | CV mean | CV std |",
        "|---:|---|---|---|---|---:|---:|",
    ]
    for rank, result in enumerate(sorted(results, key=lambda item: item.cv_mean, reverse=True)[:30], start=1):
        table_lines.append(
            f"| {rank} | {result.candidate_name} | {result.candidate_type} | {result.feature_set_name} | "
            f"{result.model_name} | {result.cv_mean:.5f} | {result.cv_std:.5f} |"
        )
    code_fence = chr(96) * 3
    return f"""# Titanic Model Zoo + Ensemble Report

## Goal

Beat the current HGB-only approach by testing a broad tabular model zoo and voting ensembles.

Current best public reference:

{code_fence}text
model_comparison_best_submission.csv
public LB = {CURRENT_BEST_PUBLIC_LB:.5f}
{code_fence}

## What This Sprint Adds

{code_fence}text
Feature sets:
- plus_title
- binned_competition
- wide_competition
- nb_competition_ordinal

Models:
- GaussianNB
- LogisticRegression
- KNN
- SVC
- DecisionTree
- RandomForest
- ExtraTrees
- AdaBoost
- GradientBoosting
- HistGradientBoosting
- XGBoost if installed

Ensembles:
- top3 hard voting
- top5 hard voting
- AdaBoost + ExtraTrees + GaussianNB weighted hard voting
{code_fence}

## Top Results

{chr(10).join(table_lines)}

## Best Local Candidate

- Candidate: `{best.candidate_name}`
- Type: `{best.candidate_type}`
- Feature set: `{best.feature_set_name}`
- Model: `{best.model_name}`
- CV mean: `{best.cv_mean:.5f}`
- CV std: `{best.cv_std:.5f}`

## Submission

- Path: `{BEST_SUBMISSION_PATH}`

## Required Next Step

Before submitting:

{code_fence}bash
python competitions/titanic/src/risky_flip_guard.py \\
  --candidate model_zoo=competitions/titanic/submissions/model_zoo_ensemble_best_submission.csv
{code_fence}

Only submit if the guard report is acceptable.
"""


def run_pipeline() -> list[EvaluationResult]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = build_training_frames()
    feature_sets = build_feature_sets()
    model_specs = build_model_specs()
    single_results = evaluate_model_zoo(train_df, feature_sets, model_specs)
    voting_results = evaluate_voting_candidates(train_df, feature_sets, model_specs, single_results)
    all_results = single_results + voting_results
    best = find_best_result(all_results)
    results_to_dataframe(all_results).to_csv(RESULTS_PATH, index=False)
    REPORT_PATH.write_text(build_report(all_results, best), encoding="utf-8")
    train_best_and_save_submission(best, train_df, test_df, feature_sets, model_specs, all_results)
    return sorted(all_results, key=lambda item: item.cv_mean, reverse=True)


def main() -> None:
    results = run_pipeline()
    print("=== Titanic Model Zoo + Ensemble ===")
    for rank, result in enumerate(results[:30], start=1):
        print(
            f"{rank}. {result.candidate_name} | type={result.candidate_type} | "
            f"cv_mean={result.cv_mean:.5f} | cv_std={result.cv_std:.5f}"
        )
    print()
    print(f"Saved results to: {RESULTS_PATH}")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Saved submission to: {BEST_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
