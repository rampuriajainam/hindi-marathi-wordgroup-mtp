"""KT-2 helpers (scripts/tokenizer_fertility.py, scripts/accept_by_token_type.py): pure functions, no network."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fert = _load("tokenizer_fertility")
acc = _load("accept_by_token_type")


# ---- span -> (accepted, first miss, censored) ----

def test_step_slots_basic():
    # k=4 (3 drafts), long budget. Step 0 at s=0: span 2 -> slot1 accepted, slot2 first miss, slot3 censored.
    # Step 1 at s=2: span 4 -> all three accepted. Step 2 at s=6: span 1 -> slot1 miss, slots 2-3 censored.
    slots, cens = acc.step_slots([2, 4, 1, 1], n_ids=8, k=4, max_new_tokens=100)
    assert (1, 1, True) in slots and (2, 2, False) in slots
    assert [(i, d, a) for i, d, a in slots if 3 <= i <= 5] == [(3, 1, True), (4, 2, True), (5, 3, True)]
    assert (7, 1, False) in slots
    # last step at s=7: slot 1 targets index 8, which does not exist -> not evaluated
    assert all(i < 8 for i, _, _ in slots)
    assert cens == 1 + 2 + 3


def test_step_slots_end_of_budget():
    # max_new_tokens=6: at s=4 only min(3, 6-4-1)=1 draft is proposed; the final step (s=5) proposes none
    slots, cens = acc.step_slots([4, 1, 1], n_ids=6, k=4, max_new_tokens=6)
    assert slots == [(1, 1, True), (2, 2, True), (3, 3, True), (5, 1, False)]
    assert cens == 0


def test_step_slots_eos_proposes_nothing():
    slots, cens = acc.step_slots([1, 1], n_ids=2, k=4, max_new_tokens=64, eos_at=[1])
    assert slots == [(1, 1, False)]       # step 0 missed slot 1; step 1 is EOS: no drafts
    assert cens == 2


def test_recomputed_accept_rate_matches_rule():
    entry = {"per_prompt": {"ids": [[5, 5, 5, 5, 5, 5, 5, 5]], "spans": [[2, 4, 1, 1]]}}
    rate = acc.recomputed_accept_rate(entry, k=4, max_new_tokens=100, eos=None)
    # proposed: 4 steps x 3 drafts; accepted per head: d1 2 (steps 0,1), d2 1, d3 1
    assert rate == [2 / 4, 1 / 4, 1 / 4]


# ---- piece classes and word position ----

def test_classify_piece():
    assert acc.classify_piece("▁हम") == "initial"
    assert acc.classify_piece("न्हें") == "internal"
    assert acc.classify_piece("▁है।") == "initial"          # danda glued to a word stays with the word
    assert acc.classify_piece("।") == "punct"
    assert acc.classify_piece("▁,") == "punct"
    assert acc.classify_piece("२०२४") == "digit"
    assert acc.classify_piece("▁12") == "digit"
    assert acc.classify_piece("<0x0A>") == "other"
    assert acc.classify_piece("▁") == "other"
    assert acc.classify_piece("</s>", special=True) == "other"


def test_word_position():
    pieces = ["▁क्ष", "त्र", "िय", "▁हम"]
    assert [acc.word_position(pieces, i) for i in range(4)] == [1, 2, 3, 1]
    assert acc.word_position(["न", "▁a"], 0) is None


# ---- word-initial from offsets, tokens per word ----

def test_assign_tokens_sentencepiece_offsets_include_space():
    text = "हम उन्हें"
    # ganga-style: the offsets of "▁उ" include the leading space
    offsets = [(0, 2), (2, 4), (4, 9)]
    word_of, initial = fert.assign_tokens(text, offsets)
    assert word_of == [0, 1, 1] and initial == [True, True, False]
    assert fert.tokens_per_word(word_of, 2) == [1, 2]


def test_assign_tokens_byte_level_shared_offsets():
    text = "ab cd"
    # two byte pieces of one character share offsets (2,4)/(3,4): only the first is word-initial
    offsets = [(0, 2), (2, 4), (3, 4), (4, 5)]
    word_of, initial = fert.assign_tokens(text, offsets)
    assert word_of == [0, 1, 1, 1] and initial == [True, True, False, False]
    assert fert.shared_offsets(offsets) == 1


def test_assign_tokens_lone_space_token_opens_next_word():
    text = "ab cd"
    offsets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]   # Mistral-style: "a" "b" "▁" "c" "d"
    word_of, initial = fert.assign_tokens(text, offsets)
    assert word_of == [0, 0, 1, 1, 1] and initial == [True, False, True, False, False]
    assert fert.tokens_per_word(word_of, 2) == [2, 3]


def test_summarize_toy_tokenizer():
    class Toy:
        """Character tokenizer: every character is one token, offsets are exact."""

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
            return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

        def convert_ids_to_tokens(self, ids):
            return [None] * len(ids)

    r = fert.summarize(["ab c"], Toy())
    assert r["words"] == 2 and r["tokens"] == 4 and r["tokens_per_word"] == 2.0
    # the space token opens word "c": words get 2 ("ab") and 2 (" c") tokens
    assert r["tokens_per_word_dist"]["2"] == 1.0
    assert r["word_initial_token_share"] == 0.5
