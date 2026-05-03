from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/students.csv")


def load_students() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def main() -> None:
    df = load_students()

    print("=== First rows ===")
    print(df.head())
    print()

    print("=== Shape ===")
    print(df.shape)
    print()

    print("=== Columns ===")
    print(list(df.columns))
    print()

    print("=== Data types ===")
    print(df.dtypes)
    print()

    print("=== Missing values ===")
    print(df.isna().sum())
    print()

    print("=== Numeric summary ===")
    print(df.describe())
    print()

    print("=== Pass rate ===")
    pass_rate = df["passed"].mean()
    print(f"{pass_rate:.2%}")

    print()
    print("=== Average study hours by passed ===")
    print(df.groupby("passed")["study_hours"].mean())


if __name__ == "__main__":
    main()
