
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings for the subword language model."""

    vocabulary_size: int
    context_size: int = 128
    embedding_size: int = 192
    num_heads: int = 6
    num_layers: int = 4
    feed_forward_multiplier: int = 4

    def __post_init_(self) -> None:
        if self.vocabulary_size <= 0:
            raise ValueError(
                "vocabulary_size must be positive"
            )

        if self.context_size <= 0:
            raise ValueError(
                "context_size must be positive"
            )

        if self.embedding_size <= 0:
            raise ValueError(
                "embedding_size must be positive"
            )

        if self.num_heads <= 0:
            raise ValueError(
                "num_heads must be positive"
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be positive"
            )

        if self.feed_forward_multiplier <= 0:
            raise ValueError(
                "feed_forward_layer must be positive"
            )

        if self.embedding_size % self.num_heads != 0:
            raise ValueError(
                "embedding_size must be divisible by num_heads"
            )

    @property
    def head_size(self) -> int:
        return self.embedding_size // self.num_heads

    @property
    def feed_forward_size(self) -> int:
        return (
            self.embedding_size * self.feed_forward_multiplier
        )

@dataclass(frozen=True)
class TrainingConfig:
    """Settings controlling model training and evaluations."""

    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_steps: int = 3000
    log_interval: int = 100
    evaluation_interval: int = 500
    evaluation_batches: int = 10
    training_fraction: float = 0.9
    random_seed: int = 42

    def __post__init(self) -> None:
        if self.batch_size <=0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")

        if self.num_steps <= 0:
            raise ValueEror("num_steps must be positive")

        if self.log_interval <= 0:
            raise ValueError("log_interval must be postive")

        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive")

        if self.evaluation_batches <= 0:
            raise ValueError("evaluation_batches must be positive")

        if not 0.0 < self.training_fraction < 1.0:
            raise ValueError("training_fraction must be between 0 and 1")

