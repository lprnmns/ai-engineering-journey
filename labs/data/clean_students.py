from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/students.csv")
PROCESSED_DATA_PATH = Path("data/processed/students_clean.csv")


def load_raw_students(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_students(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    age_median = float(cleaned["age"].median())
    study_hours_mean = float(cleaned["study_hours"].mean())

    cleaned["age"] = cleaned["age"].fillna(age_median)
    cleaned["study_hours"] = cleaned["study_hours"].fillna(study_hours_mean)

    return cleaned


def save_clean_students(
    df: pd.DataFrame,
    path: Path = PROCESSED_DATA_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    raw_df = load_raw_students()
    clean_df = clean_students(raw_df)

    print("=== Missing values before cleaning ===")
    print(raw_df.isna().sum())
    print()

    print("=== Missing values after cleaning ===")
    print(clean_df.isna().sum())
    print()

    print("=== Cleaned data ===")
    print(clean_df)

    save_clean_students(clean_df)
    print()
    print(f"Saved cleaned dataset to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
