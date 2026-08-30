import numpy as np
import pytest

from samexporter.clip_tokenizer import tokenize


def test_clip_tokenizer_matches_sam3_tokens():
    tokens = tokenize(["plant", "a person wearing a red hat"], context_length=32)
    assert tokens.dtype == np.int64
    assert tokens.shape == (2, 32)
    assert tokens[0, :3].tolist() == [49406, 3912, 49407]
    assert tokens[1, :8].tolist() == [
        49406,
        320,
        2533,
        3309,
        320,
        736,
        3801,
        49407,
    ]


def test_clip_tokenizer_rejects_prompt_over_context_limit():
    with pytest.raises(ValueError, match="32-token limit"):
        tokenize(" ".join(["plant"] * 40), context_length=32)
