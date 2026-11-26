"""karthika grouper (Karthika et al. 2025, Table 4 rules) on hand-built HDTB-style parses: no parser needed."""

import re

import pytest

from mtp.data.grouping.base import REGISTRY, get_grouper
from mtp.data.grouping.karthika_grouper import KarthikaGrouper, karthika_partition


def parse(text, rows):
    """rows: (xpos, upos, deprel, head) per whitespace word, 1-based heads."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    return [[{"id": i + 1, "xpos": x, "upos": u, "deprel": d, "head": h, "dspan": spans[i]}
             for i, (x, u, d, h) in enumerate(rows)]]


def test_paper_example():
    # the paper's examples in one sentence (parse as stanza gives it)
    text = "श्री ए पी जे अब्दुल कलाम ने अरुणाचल प्रदेश में विमान सेवा को देखते ही बंद कर दिया गया था ।"
    rows = [("NNC", "NOUN", "compound", 6), ("NNPC", "PROPN", "compound", 6), ("NNPC", "PROPN", "compound", 6),
            ("NNPC", "PROPN", "compound", 6), ("NNPC", "PROPN", "compound", 6), ("NNP", "PROPN", "nsubj", 17),
            ("PSP", "ADP", "case", 6), ("NNPC", "PROPN", "compound", 9), ("NNP", "PROPN", "obl", 17),
            ("PSP", "ADP", "case", 9), ("NNC", "NOUN", "compound", 12), ("NN", "NOUN", "obj", 14),
            ("PSP", "ADP", "case", 12), ("VM", "VERB", "advcl", 17), ("RP", "PART", "dep", 14),
            ("JJ", "ADJ", "compound", 17), ("VM", "VERB", "root", 0), ("VAUX", "VERB", "compound", 17),
            ("VAUX", "AUX", "aux:pass", 17), ("VAUX", "AUX", "aux:pass", 17), ("SYM", "PUNCT", "punct", 17)]
    groups, types = karthika_partition(text, parse(text, rows))
    assert [" ".join(g) for g in groups] == ["श्री ए पी जे अब्दुल कलाम ने", "अरुणाचल प्रदेश में", "विमान सेवा को",
                                              "देखते ही", "बंद कर दिया गया था", "।"]
    assert types == ["postposition", "postposition", "postposition", "other", "light_verb", "single"]


def test_only_consecutive_relations_join():
    # के attaches to the following word, not to the previous word or its head: it stays alone
    text = "राम और श्याम के घर"
    rows = [("NNP", "PROPN", "nmod", 5), ("CC", "CCONJ", "cc", 3), ("NNP", "PROPN", "conj", 1),
            ("PSP", "ADP", "case", 5), ("NN", "NOUN", "root", 0)]
    groups, _ = karthika_partition(text, parse(text, rows))
    assert [" ".join(g) for g in groups] == ["राम", "और", "श्याम", "के", "घर"]


def test_registered_and_lazy():
    assert REGISTRY["karthika"].endswith("KarthikaGrouper")
    assert REGISTRY["random_karthika"].endswith("KarthikaRandomGrouper")
    g = KarthikaGrouper()               # no parser loaded until a sentence is parsed
    assert g.pipeline is None and g.name == "karthika"
    with pytest.raises(ValueError):
        KarthikaGrouper(lang="mr")


def test_long_sentence_tail_is_single_words():
    g = KarthikaGrouper()
    words = ["क"] * (g.MAX_WORDS + 5)
    text = " ".join(words)
    rows = [("NN", "NOUN", "dep", 0)] * g.MAX_WORDS
    head = " ".join(words[: g.MAX_WORDS])

    class Doc:   # what doc_words reads: sentences -> tokens -> words
        pass

    doc = Doc()
    sent = Doc()
    sent.tokens = []
    for i, (x, u, d, h) in enumerate(rows):
        w = Doc()
        w.id, w.head, w.deprel, w.upos, w.xpos = i + 1, h, d, u, x
        t = Doc()
        t.words = [w]
        sent.tokens.append(t)
    doc.sentences = [sent]
    groups, types = g._finish(text, doc)
    assert sum(len(x) for x in groups) == len(words) and groups[-5:] == [["क"]] * 5
    assert head  # sanity


def test_random_karthika_uses_fitted_histogram():
    g = get_grouper("random_karthika", "hi")
    s = "यह एक बहुत लंबा वाक्य है जिसमें कई शब्द हैं"
    assert [w for x in g.group_words(s) for w in x] == s.split()
    assert max(g.lengths) >= 4      # karthika groups are longer than the v0 default (max 5 words but rare)
