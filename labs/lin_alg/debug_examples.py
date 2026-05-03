#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.lin_alg.vec import cosine_similarity, dot


def main() -> None:
    examples = [
        ([1, 2, 3], [4, 5, 6]),
        ([1, 2], [1]),
        ([0, 0], [1, 2]),
    ]

    for left, right in examples:
        print("-" * 40)
        print("left:", left)
        print("right:", right)

        try:
            print("dot:", dot(left, right))
            print("cosine:", cosine_similarity(left, right))
        except ValueError as error:
            print("Handled error:", error)


if __name__ == "__main__":
    main()
