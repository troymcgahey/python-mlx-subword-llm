
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

    def __post_int_(self) -> None:
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
