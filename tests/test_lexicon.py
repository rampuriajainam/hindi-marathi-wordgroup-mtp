"""Lexicon chains: tree decoding stays identical to greedy; chains complete the word in progress."""

import random

from mtp.eval import tree_decode as td
from mtp.eval.lexicon import LexiconLookup
from mtp.eval.prompt_lookup import PromptLookup
from mtp.eval.spec_decode import greedy_generate
from mtp.eval.tree_policy import StaticTree
from tests.test_spec_decode import _tiny_model


def test_completions_and_graft_at_node():
    flags = [True, False, False, False, True]
    lex = LexiconLookup(StaticTree([]), {(0, 1): [[2, 3], [3]], (4,): [[1]]}, flags, chains=2, ext=1, ext_nodes=4)
    assert lex.completions([4, 0, 1], 2) == [[2, 3], [3]]
    assert lex.completions([0, 1, 2], 2) == []                       # unknown prefix
    assert lex.extra_chains([9 % 5, 0, 1]) == [[2, 3], [3]]
    tokens, parents, depths = [0, 4], [-1, 0], [0, 1]
    ext = lex.extend([3], tokens, parents, depths, 2)                 # row 1 = token 4 starts a word -> [1]
    assert ext == [(1, [[1]])]
    td._graft(tokens, parents, depths, ext[0][1], 5, at=1)
    assert tokens == [0, 4, 1] and parents == [-1, 0, 1] and depths == [0, 1, 2]
    assert td._static(lex)


def test_phrase_table_spans_word_starts():
    # tokens 0 and 4 start words; entry "0 1 | 4 2" spans two words
    flags = [True, False, False, False, True]
    trie = {(0, 1, 4): [[2]], (4,): [[3]], (0, 1, 4, 2): [[1]]}
    one = LexiconLookup(StaticTree([]), trie, flags, chains=4)
    two = LexiconLookup(StaticTree([]), trie, flags, chains=4, max_words=2, tag="grp")
    assert one.completions([0, 1, 4], 4) == [[3]]                    # max_words=1: only the current word
    assert two.completions([0, 1, 4], 4) == [[2], [3]]               # longest prefix (from the earlier word) first
    assert two.completions([3, 4], 4) == [[3]]                       # no word start before: same as one word
    assert two.name.endswith("+grp4e2x12") and one.name.endswith("+lex4e2x12")


def test_lexicon_tree_identical_to_greedy(tiny_model_dir, tmp_path):
    for head_type in ("resblock", "seq"):
        model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type)
        rng = random.Random(5)
        v = len(tok)
        flags = [rng.random() < 0.4 for _ in range(v)]
        trie = {}
        for _ in range(4000):                                         # random lexicon: must never change the output
            p = tuple(rng.randrange(4, v) for _ in range(rng.randint(1, 2)))
            trie[p] = [[rng.randrange(4, v) for _ in range(rng.randint(1, 3))] for _ in range(3)]
        for _ in range(5):
            prompt = [rng.randrange(4, v) for _ in range(rng.randint(3, 9))]
            ref, _ = greedy_generate(model, tok, prompt, 16)
            for base in (StaticTree.from_widths((1, 1, 1)), StaticTree.from_widths((3, 2, 1))):
                for inner in (base, PromptLookup(base, chains=4, length=6)):
                    ids, _ = td.generate_tree(model, tok, prompt, 16, LexiconLookup(inner, trie, flags))
                    assert ids == ref, (head_type, prompt)
