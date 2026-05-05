from pathlib import Path


RAW_DATA_DIR = Path("competitions/titanic/data/raw")

REQUIRED_FILES = [
    "train.csv",
    "test.csv",
    "gender_submission.csv",
]


def find_missing_files() -> list[str]:
    missing_files = []

    for file_name in REQUIRED_FILES:
        file_path = RAW_DATA_DIR / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    return missing_files


def main() -> None:
    missing_files = find_missing_files()

    if missing_files:
        print("Missing Titanic data files:")
        for file_name in missing_files:
            print(f"- {file_name}")

        print()
        print("Place the missing files under:")
        print(RAW_DATA_DIR)
        raise SystemExit(1)

    print("All Titanic data files are present.")
    for file_name in REQUIRED_FILES:
        print("-", RAW_DATA_DIR / file_name)


if __name__ == "__main__":
    main()
