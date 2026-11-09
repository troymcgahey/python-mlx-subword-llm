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
            
def load_checkpoint(
    name: str,
    checkpoint_directory: Path=DEFAULT_CHECKPOINT_DIRECTORY,
) -> tuple[
    SubwordLanguageModel,
    ModelConfig,
    TrainingConfig,
    int, 
    Path,
]:
    """Reconstruct a model and load its saved weights."""

    if not name:
        raise ValueError("checkpoint name cannot be empty.")

    weights_path = checkpoint_directory / f"{name}.safetensors"
    metadata_path = checkpoint_directory / f"{name}.json"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint weights not found: {weights_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Checkpoint metadata not found: {metadata_path}"
        )
        
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if metadata.get("format_version") != 1:
        raise ValueError(
            "Unsuported checkpoint format version: "
            f"{metadata.get('format_version')}"
        )

    model_config = ModelConfig(
        **metadata["model_config"]
    )

    training_config = TrainingConfig(
        **metadata["training_config"]
    )

    model = SubwordLanguageModel(model_config)

    model.load_weights(str(weights_path))

    mx.eval(model.parameters())

    step = metadata["step"]
    tokenizer_path = Path(metadata["tokenizer_path"])

    return (
        model,
        model_config,
        training_config,
        step,
        tokenizer_path,
    )


