"""
Dump the nanochat tokenizer vocabulary to a text file.

Example:
    python -m scripts.dump_vocab --output /workspace/vocabulary.txt
"""

import argparse
import os

from nanochat.tokenizer import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Dump nanochat tokenizer vocabulary.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output text file path.",
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for token_id in range(vocab_size):
            token_bytes = tokenizer.decode_single_token_bytes(token_id)
            token_text = token_bytes.decode("utf-8", errors="replace")

            # repr() makes spaces, tabs, newlines, etc. visible.
            f.write(f"{token_id:5d}\t{token_text!r}\n")

    print(f"Wrote {vocab_size:,} tokens to:")
    print(output_path)


if __name__ == "__main__":
    main()