from labs.data.pandas_basics import load_students


def test_load_students_returns_expected_shape() -> None:
    df = load_students()

    assert df.shape == (7, 5)


def test_students_dataset_has_missing_values() -> None:
    df = load_students()

    missing_values = df.isna().sum()

    assert missing_values["age"] == 1
    assert missing_values["study_hours"] == 1


def test_passed_column_has_boolean_values() -> None:
    df = load_students()

    assert df["passed"].dtype == bool
