#!/usr/bin/env python3

from vec import cosine_similarity, dot


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
