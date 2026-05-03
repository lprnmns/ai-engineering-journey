def ensure_non_empty(value: str, field_name: str) -> None:
    """Raise ValueError if the provided string is empty or whitespace."""
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty.")


def ensure_non_negative(value: float, field_name: str) -> None:
    """Raise ValueError if the provided number is negative."""
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")
