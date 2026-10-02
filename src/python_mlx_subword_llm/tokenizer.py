from tokenizers import Tokenizer

class SubwordTokenizer:
    """A small application-friendly wrapper around GPT-2's tokenizer."""

    END_TOKEN = "<|endoftext|>"

    def __init__(self) -> None:
        self._tokenizer = Tokenizer.from_pretrained("gpt2")

        end_token_id = self._tokenizer.token_to_id(self.END_TOKEN)

        if end_token_id is None:
            raise ValueError(f"Tokenizer does not contain {self.END_TOKEN!r}")

        self._end_token_id = end_token_id

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def end_token_id(self) -> int:
        return self._end_token_id

    def encode(
        self,
        text: str,
        add_end_token: bool=False,
    ) -> list[int]:
        token_ids = self._tokenizer.encode(text).ids

        if add_end_token:
            token_ids.append(self.end_token_id)

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool=True,
    ) -> str:
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def token_for_id(self, token_id: int) -> str | None:
        return self._tokenizer.id_to_token(token_id)
