from pathlib import Path
import sqlite3
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.data.clean_students import PROCESSED_DATA_PATH, clean_students, load_raw_students


DB_PATH = Path("data/processed/students.db")
TABLE_NAME = "students"


def load_clean_students_csv(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    raw_df = load_raw_students()
    return clean_students(raw_df)


def write_students_to_db(
    df: pd.DataFrame,
    db_path: Path = DB_PATH,
    table_name: str = TABLE_NAME,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        df.to_sql(table_name, connection, if_exists="replace", index=False)


def query_students(
    sql: str,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(sql, connection)


def initialize_database() -> None:
    df = load_clean_students_csv()
    write_students_to_db(df)


def main() -> None:
    initialize_database()

    print("=== All students ===")
    print(query_students("SELECT * FROM students;"))
    print()

    print("=== Passed students ===")
    print(
        query_students(
            """
            SELECT name, age, study_hours, previous_score
            FROM students
            WHERE passed = 1
            ORDER BY study_hours DESC;
            """
        )
    )
    print()

    print("=== Top 3 by previous score ===")
    print(
        query_students(
            """
            SELECT name, previous_score
            FROM students
            ORDER BY previous_score DESC
            LIMIT 3;
            """
        )
    )
    print()

    print("=== Average study hours by pass status ===")
    print(
        query_students(
            """
            SELECT passed, AVG(study_hours) AS avg_study_hours
            FROM students
            GROUP BY passed;
            """
        )
    )


if __name__ == "__main__":
    main()
