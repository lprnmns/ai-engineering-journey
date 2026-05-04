import pandas as pd

from labs.ml.train_baseline_classifier import (
    BaselineModelResult,
    build_report,
    evaluate_model,
    train_baseline_model,
)


def build_sample_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.DataFrame(
        {
            "age": [21.0, 23.0, 22.0, 20.0],
            "study_hours": [8.0, 2.0, 7.0, 3.0],
            "previous_score": [90, 45, 85, 55],
        }
    )
    y_train = pd.Series([True, False, True, False])

    X_test = pd.DataFrame(
        {
            "age": [21.0, 24.0],
            "study_hours": [7.5, 2.5],
            "previous_score": [88, 50],
        }
    )
    y_test = pd.Series([True, False])

    return X_train, y_train, X_test, y_test


def test_train_baseline_model_can_predict() -> None:
    X_train, y_train, X_test, _ = build_sample_dataset()

    model = train_baseline_model(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == 2


def test_evaluate_model_returns_accuracy_between_zero_and_one() -> None:
    X_train, y_train, X_test, y_test = build_sample_dataset()

    model = train_baseline_model(X_train, y_train)
    result = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        train_rows=len(X_train),
    )

    assert 0.0 <= result.accuracy <= 1.0
    assert result.train_rows == 4
    assert result.test_rows == 2
    assert len(result.predictions) == 2
    assert len(result.actuals) == 2


def test_build_report_contains_core_sections() -> None:
    result = BaselineModelResult(
        accuracy=0.5,
        train_rows=4,
        test_rows=2,
        predictions=[True, False],
        actuals=[True, True],
        report_text="dummy classification report",
    )

    report = build_report(result)

    assert "# W3 Baseline Model Report" in report
    assert "Accuracy: 50.00%" in report
    assert "Logistic Regression" in report
    assert "clean data -> feature dataset -> train model" in report
