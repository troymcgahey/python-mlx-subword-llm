from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

CORPUS_PATH = Path("data/raw/tiny_shakespeare.txt")
TOKENIZER_PATH = Path("artifacts/tokenizer.json")

END_TOKEN = "|<endoftext|>"
UNKNOWN_TOKEN = "<|unknwon|>"

VOCABULARY_SIZE = 4096
TRAINING_FRACTION = 0.9

def main() -> None:
    text = CORPUS_PATH.read_text(encoding="utf-8")

    split_index = int(len(text) * TRAINING_FRACTION)
    training_text = text[:split_index]

    tokenizer = Tokenizer (
        BPE(unk_token=UNKNOWN_TOKEN)
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCABULARY_SIZE,
        min_frequency=2,
        special_tokens=[
            END_TOKEN,
            UNKNOWN_TOKEN,
        ],
        initial_alphabet=ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(
        [training_text],
        trainer=trainer,
    )

    TOKENIZER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(str(TOKENIZER_PATH))

    sample_text = "First Citizen:\nBeofre we proceed any further."
    encoding = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(encoding.ids)

    print("Training characters:", len(training_text))
    print("Vocabulary size:", tokenizer.get_vocab_size())
    print("End token ID:", tokenizer.token_to_id(END_TOKEN))
    print("Unknown token ID:", tokenizer.token_to_id(UNKNOWN_TOKEN))
    print("Samplem IDs:", encoding.ids)
    print("Sample tokens:", encoding.tokens)
    print("Decoded sample:", repr(decoded_text))
    print("Round trip succeeded:", decoded_text == sample_text)
    print("Saved tokenizer to:", TOKENIZER_PATH)

if __name__ == "__main__":
    main()
