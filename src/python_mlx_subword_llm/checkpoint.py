import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from python_mlx_subword_llm.config import (
    ModelConfig,
    TrainingConfig,
)
from python_mlx_subword_llm.model import SubwordLanguageModel

DEFAULT_CHECKPOINT_DIRECTORY = Path("checkpoints")
DEFAULT_TOKENIZER_PATH = Path("artifacts/tokenizer.json")



def save_checkpoint(
    model: SubwordLanguageModel,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    step: int,
    name: str,
    checkpoint_directory: Path = DEFAULT_CHECKPOINT_DIRECTORY,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
) -> tuple[Path, Path]:
    """Save model weights and the information needed to reload them."""

    if step < 0:
        raise ValueError("step cannot be negative")

    if not name:
        raise ValueError("checkpoint name cannot be empty")

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights_path = checkpoint_directory / f"{name}.safetensors"
    metadata_path = checkpoint_directory / f"{name}.json"

    # Ensure all lazily evaluated parameters are complete before saving.
    mx.eval(model.parameters())

    model.save_weights(str(weights_path))

    metadata = {
        "format_version": 1,
        "step": step,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "tokenizer_path": str(tokenizer_path),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return weights_path, metadata_path
            

