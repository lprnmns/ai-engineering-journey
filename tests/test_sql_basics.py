from pathlib import Path

import pandas as pd

from labs.data.sql_basics import query_students, write_students_to_db


def build_sample_students() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Ali", "Ayse", "Can"],
            "age": [21.0, 23.0, 21.0],
            "study_hours": [5.5, 3.5, 8.0],
            "previous_score": [72, 55, 90],
            "passed": [True, False, True],
        }
    )


def test_write_students_to_db_and_query_all_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "students.db"
    students = build_sample_students()

    write_students_to_db(students, db_path=db_path)

    result = query_students("SELECT * FROM students;", db_path=db_path)

    assert result.shape == (3, 5)


def test_query_passed_students(tmp_path: Path) -> None:
    db_path = tmp_path / "students.db"
    students = build_sample_students()

    write_students_to_db(students, db_path=db_path)

    result = query_students(
        """
        SELECT name
        FROM students
        WHERE passed = 1
        ORDER BY name;
        """,
        db_path=db_path,
    )

    assert list(result["name"]) == ["Ali", "Can"]


def test_query_top_student_by_previous_score(tmp_path: Path) -> None:
    db_path = tmp_path / "students.db"
    students = build_sample_students()

    write_students_to_db(students, db_path=db_path)

    result = query_students(
        """
        SELECT name, previous_score
        FROM students
        ORDER BY previous_score DESC
        LIMIT 1;
        """,
        db_path=db_path,
    )

    assert result.loc[0, "name"] == "Can"
    assert result.loc[0, "previous_score"] == 90
