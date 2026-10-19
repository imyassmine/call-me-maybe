from copy import copy

import numpy as np

from src.json_file_parsing import FunctionDefinition
from src.state_machine import JSONStateMachine


class NoValidTokenError(Exception):
    """Raised when no token in the vocabulary satisfies the constraint."""


def prepare_vocabulary(vocabulary: dict[str, int]) -> dict[int, str]:
    """Convert token -> id vocabulary to id -> token."""
    return {
        token_id: token.replace("Ġ", " ")
        for token, token_id in vocabulary.items()
    }


def should_stop_early(
    logits: list[float],
    vocabulary: dict[int, str],
    state_machine: JSONStateMachine,
) -> bool:
    """Check whether generation should stop even without an explicit delimiter.

    Only applies once the current value is already syntactically complete
    (e.g. a full number or boolean). In that case, we look at the model's
    own top, unconstrained choice: if it would NOT extend the value further,
    the model is signalling it wants to stop, so we honor that instead of
    forcing another digit just because one happens to be technically valid.
    """
    if not state_machine.is_forced_complete():
        return False

    top_id = int(np.argmax(np.asarray(logits)))
    top_token = vocabulary.get(top_id)

    if top_token is None:
        return True

    test_state = copy(state_machine)
    return not test_state.valid_value_token(top_token)


def get_next_token(
    logits: list[float],
    vocabulary: dict[int, str],
    state_machine: JSONStateMachine,
    functions: list[FunctionDefinition],
    mode: str,
) -> tuple[str, int]:
    """Return the highest-scoring valid token, or raise NoValidTokenError.

    mode must be one of "function" or "value".
    """
    if mode not in ("function", "value"):
        raise ValueError(f"Unknown mode: {mode}")

    scores = np.asarray(logits)

    for token_id in np.argsort(scores)[::-1]:
        token_id = int(token_id)
        token = vocabulary.get(token_id)

        if token is None:
            continue

        test_state = copy(state_machine)

        if mode == "function":
            valid = test_state.valid_function_token(token, functions)
        else:
            valid = test_state.valid_value_token(token)

        if valid:
            state_machine.__dict__.update(test_state.__dict__)
            return token, token_id

    raise NoValidTokenError("No valid token found")
