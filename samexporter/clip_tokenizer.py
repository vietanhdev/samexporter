"""Small NumPy CLIP tokenizer used by the SAM3 language encoder.

Adapted from OpenAI CLIP's MIT-licensed ``simple_tokenizer.py`` and
``tokenize`` helper. See ``third_party_licenses/OPENAI_CLIP_LICENSE``.
"""

import gzip
import html
from functools import lru_cache
from pathlib import Path

import ftfy
import numpy as np
import regex


@lru_cache
def bytes_to_unicode() -> dict[int, str]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, characters)))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:]))


class CLIPTokenizer:
    def __init__(self, bpe_path: str | Path | None = None) -> None:
        if bpe_path is None:
            bpe_path = Path(__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        self.byte_encoder = bytes_to_unicode()
        with gzip.open(bpe_path, "rt", encoding="utf-8") as bpe_file:
            merges = bpe_file.read().splitlines()[1 : 49152 - 256 - 2 + 1]
        merge_pairs = [tuple(merge.split()) for merge in merges]
        vocab = list(self.byte_encoder.values())
        vocab += [value + "</w>" for value in vocab]
        vocab += ["".join(merge) for merge in merge_pairs]
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merge_pairs, range(len(merge_pairs))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pattern = regex.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"
            r"[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            first, second = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if (first, second) not in self.bpe_ranks:
                break
            merged = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        encoded = " ".join(word)
        self.cache[token] = encoded
        return encoded

    def encode(self, text: str) -> list[int]:
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = regex.sub(r"\s+", " ", text).strip().lower()
        result = []
        for token in regex.findall(self.pattern, text):
            encoded = "".join(self.byte_encoder[value] for value in token.encode())
            result.extend(
                self.encoder[piece] for piece in self._bpe(encoded).split(" ")
            )
        return result

    def tokenize(self, texts: str | list[str], context_length: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        start = self.encoder["<|startoftext|>"]
        end = self.encoder["<|endoftext|>"]
        result = np.zeros((len(texts), context_length), dtype=np.int64)
        for index, text in enumerate(texts):
            tokens = [start, *self.encode(text), end]
            if len(tokens) > context_length:
                raise ValueError(
                    f"Text prompt is too long for SAM3's {context_length}-token limit"
                )
            result[index, : len(tokens)] = tokens
        return result


@lru_cache
def _default_tokenizer() -> CLIPTokenizer:
    return CLIPTokenizer()


def tokenize(texts: str | list[str], context_length: int = 32) -> np.ndarray:
    return _default_tokenizer().tokenize(texts, context_length=context_length)
