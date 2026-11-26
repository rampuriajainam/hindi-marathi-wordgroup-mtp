import pytest

from mtp.data.grouping.align import label_tokens
from mtp.data.grouping.base import REGISTRY, get_grouper
from mtp.data.grouping.token_random import TokenRandomGrouper

SENT = "मैं कल बाजार जा रहा था और वह घर से आया"


def make(tok, **kw):
    return TokenRandomGrouper(lang="mr", tokenizer=tok, histogram={1: 0.5, 2: 0.3, 3: 0.2}, **kw)


def test_registered():
    assert REGISTRY["random_tokens"] == "mtp.data.grouping.token_random:TokenRandomGrouper"


def test_starts_are_token_starts_and_deterministic(tiny_tok):
    g = make(tiny_tok)
    starts = g.group_start_chars(SENT)
    assert starts == sorted(starts) and starts[0] == g.token_starts(SENT)[0]
    assert set(starts) <= set(g.token_starts(SENT))
    assert starts == make(tiny_tok).group_start_chars(SENT)
    assert any(make(tiny_tok, seed=s).group_start_chars(SENT) != starts for s in range(1, 6))


def test_group_lengths_follow_histogram(tiny_tok):
    g = TokenRandomGrouper(lang="mr", tokenizer=tiny_tok, histogram={2: 1.0})
    toks = g.token_starts(SENT)
    assert g.group_start_chars(SENT) == toks[::2]


def test_align_uses_group_start_chars(tiny_tok):
    g = TokenRandomGrouper(lang="mr", tokenizer=tiny_tok, histogram={2: 1.0})
    lab = label_tokens(SENT, tiny_tok, g)
    real = [i for i in lab["group_id"] if i >= 0]
    assert real == sorted(real) and all(real.count(i) <= 2 for i in set(real))
    assert sum(lab["group_start"]) == len(set(real))


def test_no_word_partition(tiny_tok):
    with pytest.raises(NotImplementedError):
        make(tiny_tok).group_words(SENT)


def test_default_histogram_file_loads():
    g = get_grouper("random_tokens", "mr")
    assert abs(sum(g.weights) - 1) < 1e-9 and g.lengths[0] == 1
