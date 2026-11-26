"""
Random groups that ignore word boundaries: the control for word-boundary groups (`words`) in Marathi (R13 / R13a).

`random` keeps every word whole, so it can only test which words go together. This grouper places group boundaries
at random TOKEN positions, so cuts fall inside words as often as chance allows. Group lengths (in tokens) are drawn
from a histogram measured from the `words` grouper with the run's tokenizer (histograms_tokens_words_{lang}.json),
so both runs see the same number and sizes of groups; only where the boundaries fall differs.

Groups are not whole words, so this grouper has no group_words(): it implements group_start_chars() (INTERFACES §2,
optional), which align.py uses. Deterministic per (seed, sentence) through SHA-256, like RandomGrouper.
The tokenizer is loaded on first use (no download at import).
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

TOKENIZERS = {"mr": "smallstepai/Misal-1B-instruct-v0.1", "hi": "LingoIITGN/ganga-1b"}
HISTOGRAM_FILE = Path(__file__).with_name("histograms_tokens_words_{lang}.json")


class TokenRandomGrouper:
    name = "random_tokens"

    def __init__(self, lang="mr", tokenizer=None, histogram=None, seed=0, **kw):
        if lang not in TOKENIZERS:
            raise ValueError(f"supported languages: {sorted(TOKENIZERS)}")
        if histogram is None:
            raw = json.loads(Path(str(HISTOGRAM_FILE).format(lang=lang)).read_text(encoding="utf-8"))
            histogram = {int(k): v for k, v in raw["histogram"].items()}
        if not histogram or any(int(k) < 1 or v < 0 for k, v in histogram.items()) or sum(histogram.values()) <= 0:
            raise ValueError(f"histogram must map lengths >= 1 to non-negative weights, got {histogram!r}")
        total = sum(histogram.values())
        self.lang = lang
        self.seed = seed
        self.lengths = sorted(int(k) for k in histogram)
        self.weights = [histogram[k] / total for k in self.lengths]
        self._tok = tokenizer if tokenizer is not None and not isinstance(tokenizer, str) else None
        self._tok_name = tokenizer if isinstance(tokenizer, str) else TOKENIZERS[lang]

    def _tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self._tok_name)
        return self._tok

    def _rng(self, sentence):
        digest = hashlib.sha256(f"{self.seed}\x00{sentence}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def token_starts(self, sentence):
        """Start character of every non-special token of `sentence` (the tokenization training uses)."""
        enc = self._tokenizer()(sentence, return_offsets_mapping=True, add_special_tokens=False)
        return [a for a, b in enc["offset_mapping"] if b > a]

    def group_start_chars(self, sentence):
        starts = self.token_starts(sentence)
        rng = self._rng(sentence)
        out, i = [], 0
        while i < len(starts):
            out.append(starts[i])
            i += rng.choices(self.lengths, self.weights)[0]
        return out

    def group_words(self, sentence):
        raise NotImplementedError("random_tokens groups cut inside words; use group_start_chars() (align.py)")

    def group_types(self, sentence):
        raise NotImplementedError("random_tokens groups have no linguistic type")


def fit_histogram(tokenizer, sentences):
    """Tokens per whitespace word under `tokenizer` (the group sizes of the `words` grouper), as {length: share}."""
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.words import WordGrouper

    counts = Counter()
    for row in label_batch(sentences, tokenizer, WordGrouper(lang="mr"), max_length=128):
        ids = [g for g in row["group_id"] if g >= 0]
        counts.update(Counter(ids).values())
    n = sum(counts.values())
    return {k: counts[k] / n for k in sorted(counts)}
