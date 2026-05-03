from typing import TypeAlias


Matrix: TypeAlias = list[list[float]]


def shape(matrix: Matrix) -> tuple[int, int]:
    if not matrix:
        raise ValueError("Matrix cannot be empty.")

    row_length = len(matrix[0])

    if row_length == 0:
        raise ValueError("Matrix rows cannot be empty.")

    for row in matrix:
        if len(row) != row_length:
            raise ValueError("All matrix rows must have the same length.")

    return len(matrix), row_length


def transpose(matrix: Matrix) -> Matrix:
    row_count, column_count = shape(matrix)

    return [
        [matrix[row_index][column_index] for row_index in range(row_count)]
        for column_index in range(column_count)
    ]


def dot_product(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same length.")

    return sum(a * b for a, b in zip(left, right))


def matmul(left: Matrix, right: Matrix) -> Matrix:
    left_rows, left_columns = shape(left)
    right_rows, right_columns = shape(right)

    if left_columns != right_rows:
        raise ValueError(
            "Matrix dimensions are incompatible: "
            f"left shape=({left_rows}, {left_columns}), "
            f"right shape=({right_rows}, {right_columns})"
        )

    right_t = transpose(right)

    return [
        [dot_product(left_row, right_column) for right_column in right_t]
        for left_row in left
    ]


def main() -> None:
    a = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]

    b = [
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ]

    result = matmul(a, b)

    print("A shape:", shape(a))
    print("B shape:", shape(b))
    print("A x B:")
    for row in result:
        print(row)


if __name__ == "__main__":
    main()
