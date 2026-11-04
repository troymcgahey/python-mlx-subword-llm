import math

import mlx.core as mx
import mlx.nn as nn

from python_mlx_subword_llm.config import ModelConfig

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention that cannot inspect future tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.embedding_size = config.embedding_size

        #Produce queries, keys, and values with one projection.
        self.query_key_value = nn.Linear(
            config.embedding_size,
            3 * config.embedding_size,
            bias=False,
        )

        # Combine information from all attention heads.
        self.output_projection = nn.Linear(
            config.embedding_size,
            config.embedding_size,
            bias=False,
        )

    def __call__(
        self,
        inputs: mx.array,
        return_attention: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        batch_size, sequence_length, embedding_size = inputs.shape

        # Shape: (batch, sequence, 3 * embedding)
        query_key_value = self.query_key_value(inputs)

        queries, keys, values = mx.split(
            query_key_value,
            3,
            axis=-1,
        )

        #Separate the embedding dimension into attention heads.
        #
        # Before transpose:
        # batch, sequence, heads, head_size)
        #
        # After transpose:
        # (batch, heads, sequence, head_size)
        queries = queries.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size,
        ).transpose(0, 2, 1, 3)

        keys = keys.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size,
        ).transpose(0, 2, 1, 3)

        values = values.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size,
        ).transpose(0, 2, 1, 3)

        # Compare every query position with every key position.
        #
        # Shape:
        # (batch, heads, query_position, key_position)
        attention_scores = queries @ keys.transpose(0, 1, 3, 2)

        attention_scores = attention_scores / math.sqrt(self.head_size)

#        # Entries above diagnol represent future positions.
#        causal_mask = mx.triu(
#            mx.full(
#                (sequence_length, sequence_length),
#                float("-inf"),
#            ),
#            k=1,
#        )
#
#        attention_scores = attention_scores + causal_mask

        # Colunm positions are key positions.
        key_positions = mx.arange(sequence_length)[None, :]

        # Row positions are query positions.
        query_positions = mx.arange(sequence_length)[:, None]

        # True wherever a key is in the query's future.
        causal_mask = key_positions > query_positions

        attention_scores = mx.where(
            causal_mask,
            float("-inf"),
            attention_scores,
        )

        # Normalize across the key positions.
        attention_weights = mx.softmax(
            attention_scores,
            axis=-1,
        )

        # Shape: (batch, heads, sequence, head_size)
        attended_values = attention_weights @ values

        # Recombine all heads:
        # (batch, sequence, heads, head_size)
        attended_values = values.transpose(0, 2, 1, 3)

        # Shape: (batch, sequence, embedding)
        attended_values = attended_values.reshape(
            batch_size,
            sequence_length,
            embedding_size,
        )

        output = self.output_projection(attended_values)

        if return_attention:
            return output, attention_weights

        return output

class FeedForward(nn.Module):
    """Process each token position indepently."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                config.embedding_size,
                config.feed_forward_size,
            ),
            nn.GELU(),
            nn.Linear(
                config.feed_forward_size,
                config.embedding_size,
            ),
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.layers(inputs)

class TransformerBlock(nn.Module):
    """One pre-normalized Transformer decoder block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.attention_normalization = nn.LayerNorm(
            config.embedding_size
        )

        self.attention = CausalSelfAttention(config)

        self.feed_forward_normalization = nn.LayerNorm(
            config.embedding_size
        )

        self.feed_forward = FeedForward(config)

    def __call__(
        self,
        inputs: mx.array,
        return_attention: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        # Normalize before attention.
        normalized_inputs = self.attention_normalization(inputs)
            
        if return_attention:
            attention_output, attention_weights = self.attention(
                normalized_inputs,
                return_attention=True,
            )
        else:
            attention_output = self.attention(normalized_inputs)

        # First residual connection
        hidden_states = inputs + attention_output

        # Normalized before the feed-forward network.
        normalized_hidden_states = (
            self.feed_forward_normalization(hidden_states)
        )

        feed_forward_output = self.feed_forward(
            normalized_hidden_states
        )

        # Second residual connection.
        output = hidden_states + feed_forward_output

        if return_attention:
            return output, attention_weights

        return output

class SubwordLanguageModel(nn.Module):
    """Decoder-only Transfomer for next-token prediction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.token_embeddings = nn.Embedding(
            config.vocabulary_size,
            config.embedding_size,
        )

        self.position_embeddings = nn.Embedding(
            config.context_size,
            config.embedding_size,
        )

        self.blocks = [
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ]

        self.final_normalization = nn.LayerNorm(
            config.embedding_size
        )

        self.output_projection = nn.Linear(
            config.embedding_size,
            config.vocabulary_size,
            bias=False,
        )

    def __call__(
        self,
        token_ids: mx.array,
        return_attention: bool = False,
    ) -> mx.array | tuple[mx.array, list[mx.array]]:
        
        _, sequence_length = token_ids.shape

        if sequence_length > self.config.context_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context size {self.config.context_size}"
            )

        positions = mx.arange(sequence_length)

        # Shape: (batch, sequence, embedding)
        hidden_states = (
            self.token_embeddings(token_ids)
            + self.position_embeddings(positions)
        )

        attention_by_layer: list[mx.array] = []

        for block in self.blocks:
            if return_attention:
                hidden_states, attention_weights = block(
                    hidden_states,
                    return_attention=True,
                )

                attention_by_layer.append(attention_weights)
            else:
                hidden_states = block(hidden_states)

        hidden_states = self.final_normalization(hidden_states)

        # Shape: (batch, sequence, vocabulary)
        logits = self.output_projection(hidden_states)

        if return_attention:
            return logits, attention_by_layer

        return logits
