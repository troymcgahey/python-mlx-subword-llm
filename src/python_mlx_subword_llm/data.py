from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx

def load_text(path: Path) -> str:
    """Load a UTF-8 text corpus."""

    return path.read_text(encoding="utf-8")

def split_token_sequence(
    token_ids: Sequence[int],
    training_fraction: float = 0.9,
) -> tuple[list[int], list[int]]:
    """Divide token IDs into contiguous training and validation sets."""

    if not 0.0 < training_fraction < 1.0:
        raise ValueError("training_fraction must be between 0 and 1")

    split_index = int(len(token_ids) * training_fraction)

    training_ids = list(token_ids[:split_index])
    validation_ids = list(token_ids[split_index:])

    return training_ids, validation_ids

def sample_batch(
    token_ids: Sequence[int],
    context_size: int,
    batch_size: int,
) -> tuple[mx.array, mx.array]:
    """Sample random input and next-token targer windows."""

    if context_size <= 0:
        raise ValueError("context_size must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if len(token_ids) <= context_size:
        raise ValueErorr("Token sequence must be longer than context_size")

    maximum_start = len(token_ids) - context_size - 1

    start_positions = mx.random.randint(
        low=0,
        high=maximum_start + 1,
        shape=(batch_size,),
    )

    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []

    for start in start_positions.tolist():
        window = token_ids[start : start + context_size + 1]

        input_rows.append(list(window[:-1]))
        target_rows.append(list(window[1:]))

    inputs = mx.array(input_rows, dtype=mx.int32)
    targets = mx.array(target_rows, dtype=mx.int32)

    return inputs, targets







        
