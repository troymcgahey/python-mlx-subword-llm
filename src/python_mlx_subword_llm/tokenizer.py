from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer

DEFAULT_TOKENIZER_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "tokenizer.json"
)

class SubwordTokenizer:
    """Wrapper around the project's locally trained BPE tokenizer."""

    END_TOKEN = "<|endoftext|>"
    UNKNOWN_TOKEN = "<|unknown|>"

    def __init__(
        self,
        tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    ) -> None:
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {tokenizer_path}."
                "Run the train_tokenizer module first."
            )

        self._tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        end_token_id = self._tokenizer.token_to_id(
            self.END_TOKEN
        )

        unknown_token_id = self._tokenizer.token_to_id(
            self.UNKNOWN_TOKEN
        )

        if end_token_id is None:
            raise ValueError(
                f"Tokenizer does not contain {self.END_TOKEN!r}"
            )

        self._end_token_id = end_token_id

        if unknown_token_id is None:
            raise ValueError(
                f"Tokenizer does not contain {self.UNKNOWN_TOKEN!r}"
            )

        self._end_token_id = end_token_id
        self._unknown_token_id = unknown_token_id

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def end_token_id(self) -> int:
        return self._end_token_id

    @property
    def unknown_token_id(self) -> int:
        return self._unknown_token_id

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
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
        )

    def token_for_id(self, token_id: int) -> str | None:
        return self._tokenizer.id_to_token(token_id)
