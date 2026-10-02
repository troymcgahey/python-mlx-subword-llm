from tokenizers import Tokenizer

def main() -> None:
    tokenizer = Tokenizer.from_pretrained("gpt2")

    text = "Tokenization turns unbelievable stories into tokens,"
    encoding = tokenizer.encode(text)

    print("Original text:", repr(text))
    print("Vocabulary size:", tokenizer.get_vocab_size())
    print("Token count:", len(encoding.ids))
    print("Token IDs:", encoding.ids)
    print("Tokens:", encoding.tokens)
    print("Offsets:", encoding.offsets)

    print("\nToken details:")

    for token, token_id, offset in zip(
        encoding.tokens,
        encoding.ids,
        encoding.offsets,
    ):
        start, end = offset
        original_piece = text[start:end]

        print(
            f"{token_id:>5} "
            f"{token!r:<18} "
            f"{offset!s:<10} "
            f"{original_piece!r}"
        )

    decoded_text = tokenizer.decode(encoding.ids)

    print("\nDecoded text:", repr(decoded_text))
    print("Round trip succeeded:", decoded_text == text)

if __name__ == "__main__":
    main()


