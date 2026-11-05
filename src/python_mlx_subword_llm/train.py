from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from python_mlx_subword_llm.config import ModelConfig
from python_mlx_subword_llm.data import (
    load_text,
    sample_batch,
    split_text,
)
from python_mlx_subword_llm.model import SubwordLanguageModel
from python_mlx_subword_llm.tokenizer import SubwordTokenizer

CORPUS_PATH = Path("data/raw/tiny_shakespeare.txt")

TRAINING_FRACTION = 0.9
BATCH_SIZE = 8
RANDOM_SEED = 42
LEARNING_RATE=3e-4
NUM_STEPS = 10

def language_model_loss(
    model: SubwordLanguageModel,
    inputs: mx.array,
    targets: mx.array,
) -> mx.array:
    """Calculate average next-token cross-entropy loss."""

    logits = model(inputs)

    vocabulary_size = logits.shape[-1]

    # Convert:
    #   logits  (batch, sequence, vocabulary)
    #   targets (batch, sequence)
    #
    # into:
    #   logits  (predictions, vocabulary)
    #   targets (predictions,)
    flat_logits = logits.reshape(-1, vocabulary_size)
    flat_targets = targets.reshape(-1)

    return nn.losses.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="mean",
    )

def main() -> None:
    mx.random.seed(RANDOM_SEED)

    tokenizer = SubwordTokenizer()

    config = ModelConfig(
        vocabulary_size=tokenizer.vocabulary_size
    )

    text = load_text(CORPUS_PATH)

    training_text, validation_text = split_text(
        text,
        training_fraction=TRAINING_FRACTION,
    )

    training_ids = tokenizer.encode(
        training_text,
        add_end_token=True,
    )

    validation_ids = tokenizer.encode(
        validation_text,
        add_end_token=True,
    )

    model = SubwordLanguageModel(config)

    optimizer = optim.AdamW(
        learning_rate=LEARNING_RATE,
    )

    loss_and_gradient = nn.value_and_grad(
        model,
        language_model_loss,
    )

    for step in range(NUM_STEPS):
        inputs, targets = sample_batch(
            training_ids,
            context_size=config.context_size,
            batch_size=BATCH_SIZE,
        )

        loss, gradients = loss_and_gradient(
            model,
            inputs,
            targets,
        )

        optimizer.update(
            model,
            gradients,
        )

        mx.eval(
            model.parameters(),
            optimizer.state,
            loss,
        )

        print(
            "Step:", step,
            "Batch loss", loss.item(),
        )

    evaluation_inputs, evaluation_targets = sample_batch(
        training_ids,
        context_size=config.context_size,
        batch_size=BATCH_SIZE,
    )

    evaluation_loss = language_model_loss(
        model,
        evaluation_inputs,
        evaluation_targets,
    )

    mx.eval(evaluation_loss)

    print(
        "Fresh-batch training loss:",
        evaluation_loss.item(),
    )
#    print("Vocabulary size:", tokenizer.vocabulary_size)
#    print("Training tokens:", len(training_ids))
#    print("Validation tokens:", len(validation_ids))
#    print("Input shape:", inputs.shape)
#    print("Target shape:", targets.shape)
#    print("Predictions per batch:", targets.size)
#    print("Initial loss:", loss.item())


if __name__ == "__main__":
    main()

