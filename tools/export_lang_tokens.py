#!/usr/bin/env python

from __future__ import annotations

import argparse
import struct
from pathlib import Path

DEFAULT_TOKENIZER = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "artifacts" / "lang_tokens.bin"
MAX_LENGTH = 48  # tokenizer_max_length, 与 checkpoint config 一致


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", type=str, default="pick up the blue block")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # SmolVLANewLineProcessor: 末尾无换行则补一个
    task = args.instruction
    if not task.endswith("\n"):
        task += "\n"

    enc = tokenizer(
        task,
        max_length=MAX_LENGTH,
        padding="max_length",
        padding_side="right",
        truncation=True,
        return_tensors="np",
    )
    ids = enc["input_ids"][0].astype("int64")
    mask = enc["attention_mask"][0].astype("uint8")
    assert ids.shape[0] == MAX_LENGTH and mask.shape[0] == MAX_LENGTH

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(struct.pack(f"<{MAX_LENGTH}q", *ids.tolist()))
        f.write(struct.pack(f"<{MAX_LENGTH}B", *mask.tolist()))

    n_valid = int(mask.sum())
    print(f"已写出 {args.output}")
    print(f"指令: {task!r}")
    print(f"有效 token 数: {n_valid}/{MAX_LENGTH}")
    print(f"tokens: {ids[:n_valid].tolist()}")


if __name__ == "__main__":
    main()
