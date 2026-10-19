"""Conversion of raw generated values into typed Python values."""


class ValueGenerationError(Exception):
    """Raised when the LLM cannot generate a valid value for a parameter."""


def convert_value(raw_value: str, param_type: str) -> int | float | str | bool:
    """Convert the raw generated text into the correct Python type."""
    try:
        if param_type == "number":
            return float(raw_value) if "." in raw_value else int(raw_value)
        if param_type == "boolean":
            return raw_value == "true"
        return raw_value
    except ValueError as error:
        raise ValueGenerationError(
            f"Generated value {raw_value!r} is not a valid {param_type}"
        ) from error
