import argparse

import mlx.core as mx

from python_mlx_subword_llm.checkpoint import load_checkpoint
from python_mlx_subword_llm.tokenizer import SubwordTokenizer

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        decsription="Generate text with a trained subword model."
    )

    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint name without its file extension.",
    )

    parser.add_argument(
        "--prompt",
        default="ROMEO",
        help="Text the model should continue."
    )

    parser.add_argument(
        "--max-new_tokens",
        type=int,
        default=200,
        help="Maximum number of subword tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling randomness; must be greater than zero.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )

    return parser.parse_args()

def generate(
    checkpoint_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    random_seed: int,
) -> str:
    
    if not prompt:
        raise ValueError("prompt cannot be empty")

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    (
        model,
        model_config,
        _,
        checkpoint_step,
        tokenizer_path,
    ) = load_checkpoint(checkpoint_name)

    tokenizer = SubwordTokenizer(
        tokenizer_path=tokenizer_path
    )

    if tokenizer.vocabulary_size != model_config.vocabulary_size:
        raise ValueError("Tokenizer vocabulary size does not match the model"
    )

    generate_ids = tokenizer.encode(prompt)

    if not generate_ids:
        raise ValueError("The prompt did not produce any tokens."
    )

    mx.random.seed(random_seed)

    print(
        f"Loaded checkpoint {checkpoint_name!r} "
        f"from step {checkpoint_step}"
    )

    for _ in range(max_new_token):
        context_ids = generated_ids[
            -model_condif.context_size:
        ]

        model_inputs = mx.array(
            [context_ids],
            dtype=mx.int32,
        )

        logits = model(model_inputs)

        # Only the last position predicts the next token.
        next_token_logits = (
            logits[0, -1]
            / temperature
        )

        next_token_id = mx.random.categorical(
            next_token_logits
        ).item()

        if next_token_id == tokenizer.end_token_id:
            print("Reached the end token.")
            break

        generated_ids.append(next_token_id)

    return tokenizer.decode(generated_ids)

def main() -> None:
    
    arguments = parse_arguments()

    generated_text = generate(
        checkpoint_name=arguments.checkpoint,
        prompt=arguments.prompt,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        random_seed=arguments.seed,
    )

    print()
    print(generated_text)

if __name__ == "__main__":
    main()
