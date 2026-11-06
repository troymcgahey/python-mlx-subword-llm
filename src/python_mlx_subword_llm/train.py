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
from python_mlx_subword_llm.checkpoint import save_checkpoint
from python_mlx_subword_llm.config import ModelConfig, TrainingConfig

CORPUS_PATH = Path("data/raw/tiny_shakespeare.txt")

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

def estimate_loss(
    model: SubwordLanguageModel,
    token_ids: list[int],
    config: ModelConfig,
    batch_size: int,
    num_batches: int,
) -> float:
    """Average loss across several randomly sampled batches."""

    losses: list[float] = []

    for _ in range(num_batches):
        inputs, targets = sample_batch(
            token_ids,
            context_size=config.context_size,
            batch_size=batch_size,
        )

        loss = language_model_loss(
            model,
            inputs,
            targets,
        )

        mx.eval(loss)

        losses.append(loss.item())

    return sum(losses) / len(losses)

def main() -> None:

    tokenizer = SubwordTokenizer()

    model_config = ModelConfig(
        vocabulary_size=tokenizer.vocabulary_size
    )

    training_config = TrainingConfig()

    mx.random.seed(training_config.random_seed)

    text = load_text(CORPUS_PATH)

    training_text, validation_text = split_text(
        text,
        training_fraction=training_config.training_fraction,
    )

    training_ids = tokenizer.encode(
        training_text,
        add_end_token=True,
    )

    validation_ids = tokenizer.encode(
        validation_text,
        add_end_token=True,
    )

    print("Vocabulary size:", tokenizer.vocabulary_size)
    print("Training tokens:", len(training_ids))
    print("Validation tokens:", len(validation_ids))

    model = SubwordLanguageModel(model_config)

    optimizer = optim.AdamW(
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    loss_and_gradient = nn.value_and_grad(
        model,
        language_model_loss,
    )

    for step in range (1, training_config.num_steps + 1):
        inputs, targets = sample_batch(
            training_ids,
            context_size=model_config.context_size,
            batch_size=model_config.batch_size,
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

        if (
            
            step == 1 or step % training_config.log_interval == 0
        ):
            print(
                "Step:",
                step,
                "Batch loss",
                loss.item(),
            )

        should_evaluate = (
            step % training_config.evaluation_interval == 0
            or step == training_config.num_steps
        )

        if should_evaluate:
            estimated_training_loss = estimate_loss(
                model,
                training_ids,
                model_config,
                batch_size=training_config.batch_size,
                num_batches=training_config.evaluation_batches,
            )

            estimated_validation_loss = estimate_loss(
                model,
                validation_ids,
                model_config,
                batch_size=training_config.batch_size,
                num_batches=training_config.evaluation_batches,
            )

            print(
                "Step:",
                step,
                "Estimated training loss:",
                estimated_training_loss,
                "Estimated validation loss:",
                estimated_validation_loss,
            )

            checkpoint_name = (
                "final",
                if step == training_config.num_steps
                else "latest"
            )

            weights_path, metadata_path = save_checkpoint(
                model=model,
                model_config=model_config,
                training_config=training_config,
                step=step,
                name=checkpoint_name,
            )
            
            print("Saved weights to:", weights_path)
            print("Saved metadata to:", metadata_path)

        


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

    for step in range(NUM_STEPS + 1):
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

    estimated_training_loss = estimate_loss(
        model,
        training_ids,
        config,
        batch_size=BATCH_SIZE,
        num_batches=EVALUATION_BATCHES,
    )

    estimated_validation_loss = estimate_loss(
        model,
        validation_ids,
        config,
        batch_size=BATCH_SIZE,
        num_batches=EVALUATION_BATCHES,
    )

    print(
        "Estimated training loss:",
        estimated_training_loss,
    )

    print(
        "Estimated validation loss:",
        estimated_validation_loss,
    )

#    evaluation_inputs, evaluation_targets = sample_batch(
#        training_ids,
#        context_size=config.context_size,
#        batch_size=BATCH_SIZE,
#    )

#    evaluation_loss = language_model_loss(
#        model,
#        evaluation_inputs,
#       evaluation_targets,
#    )

#    mx.eval(evaluation_loss)

#    print(
#        "Fresh-batch training loss:",
#        evaluation_loss.item(),
#    )
#    print("Vocabulary size:", tokenizer.vocabulary_size)
#    print("Training tokens:", len(training_ids))
#    print("Validation tokens:", len(validation_ids))
#    print("Input shape:", inputs.shape)
#    print("Target shape:", targets.shape)
#    print("Predictions per batch:", targets.size)
#    print("Initial loss:", loss.item())


if __name__ == "__main__":
    main()

