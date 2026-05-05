from competitions.titanic.src.check_data import REQUIRED_FILES


def test_titanic_required_files_are_defined() -> None:
    assert REQUIRED_FILES == [
        "train.csv",
        "test.csv",
        "gender_submission.csv",
    ]
