import argparse
import numpy as np
from typing import Literal

idx_to_char = {
    0: "{",
    1: "(",
    2: "[",
    3: "<",
    4: "}",
    5: ")",
    6: "]",
    7: ">",
    8: "SOS ",
    9: " EOS",
    10: "_",
}
char_to_idx = {char: idx for idx, char in idx_to_char.items()}

parser = argparse.ArgumentParser("Bracket Dataset Generator")
parser.add_argument("--seq_len", default=62, type=int, help="Length of the sequence")
parser.add_argument(
    "--num_seq", default=60_000, type=int, help="Number of sequences to generate"
)
parser.add_argument(
    "--output", default="brackets.txt", type=str, help="Output file path"
)
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument(
    "--mode",
    default="int",
    type=str,
    help="Output mode",
    choices=["string", "int"],
)

args = parser.parse_args()

np.random.seed(args.seed)

for _ in range(args.num_seq):
    # len = np.random.randint(args.seq_len // 4, args.seq_len // 2)
    len = args.seq_len // 2
    start_seq = np.random.randint(0, 4, (len,)).tolist()
    full_seq = [8] + start_seq + list(map(lambda x: x + 4, reversed(start_seq))) + [9] + [10] * (args.seq_len - 2 * len - 2)

    seq = (
        "".join(list(map(lambda x: idx_to_char[x], full_seq)))
        if args.mode == "string"
        else " ".join(list(map(str, full_seq)))
    )

    with open(args.output, mode="a") as f:
        f.write(seq + "\n")
