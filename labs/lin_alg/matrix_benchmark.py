import random
import time

import numpy as np

from labs.lin_alg.matrix import Matrix, matmul


def random_matrix(rows: int, columns: int) -> Matrix:
    return [
        [random.random() for _ in range(columns)]
        for _ in range(rows)
    ]


def benchmark_pure_python(left: Matrix, right: Matrix) -> float:
    start = time.perf_counter()
    matmul(left, right)
    end = time.perf_counter()

    return end - start


def benchmark_numpy(left: Matrix, right: Matrix) -> float:
    left_array = np.array(left)
    right_array = np.array(right)

    start = time.perf_counter()
    left_array @ right_array
    end = time.perf_counter()

    return end - start


def main() -> None:
    size = 100

    left = random_matrix(size, size)
    right = random_matrix(size, size)

    pure_python_duration = benchmark_pure_python(left, right)
    numpy_duration = benchmark_numpy(left, right)

    print(f"Matrix size: {size}x{size}")
    print(f"Pure Python duration: {pure_python_duration:.6f}s")
    print(f"NumPy duration:       {numpy_duration:.6f}s")

    if numpy_duration > 0:
        speedup = pure_python_duration / numpy_duration
        print(f"NumPy speedup:        {speedup:.2f}x")


if __name__ == "__main__":
    main()
