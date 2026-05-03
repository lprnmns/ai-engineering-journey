#!/usr/bin/env python3

import re
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.lin_alg.vec import cosine_similarity


def tokenize(text: str) -> List[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def word_count_vector(text: str) -> Dict[str, int]:
    """Convert text into a word-count dictionary."""
    return Counter(tokenize(text))


def align_vectors(
    a: Dict[str, int],
    b: Dict[str, int],
) -> Tuple[List[float], List[float]]:
    """Align two word-count dictionaries into numeric vectors."""
    vocabulary = sorted(set(a.keys()) | set(b.keys()))

    vector_a = [float(a.get(word, 0)) for word in vocabulary]
    vector_b = [float(b.get(word, 0)) for word in vocabulary]

    return vector_a, vector_b


def text_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using word counts."""
    counts_a = word_count_vector(text_a)
    counts_b = word_count_vector(text_b)

    if not counts_a or not counts_b:
        raise ValueError("Both texts must contain at least one word.")

    vector_a, vector_b = align_vectors(counts_a, counts_b)

    return cosine_similarity(vector_a, vector_b)


def main() -> None:
    first_text = input("First sentence: ")
    second_text = input("Second sentence: ")

    score = text_similarity(first_text, second_text)

    print(f"Similarity score: {score:.4f}")

    if score >= 0.75:
        print("Interpretation: very similar")
    elif score >= 0.40:
        print("Interpretation: somewhat similar")
    elif score > 0:
        print("Interpretation: weakly similar")
    else:
        print("Interpretation: not similar")


if __name__ == "__main__":
    main()
