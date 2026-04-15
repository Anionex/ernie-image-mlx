"""Tokenizer wrapper using HuggingFace tokenizers library."""
from tokenizers import Tokenizer
from typing import List


class ErnieTokenizer:
    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def __call__(self, text: str, **kwargs) -> dict:
        encoding = self.tokenizer.encode(text)
        return {"input_ids": encoding.ids, "attention_mask": [1] * len(encoding.ids)}
