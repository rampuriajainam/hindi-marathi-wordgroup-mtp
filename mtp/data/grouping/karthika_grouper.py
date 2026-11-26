"""Word grouping of Karthika et al. (2025), "Semantically Cohesive Word Grouping in Indian Languages"
(arXiv:2501.03988, Table 4), on the stanza Hindi parser (HDTB tags, the tag set the paper's rules use).

The paper parses with trankit and joins a dependent to its head when:
  lwg_psp        dependent PSP (postposition)                     -> noun/pronoun/verb head   (हॉल में, जाने के लिए)
  lwg_vaux(_cont) dependent VAUX (auxiliary)                       -> VM/VAUX head            (जाता था, बसाया गया)
  lwg_rp         dependent RP (particle: ही, भी, सा, नहीं)          -> any head                (देखते ही, एक ही)
  pof            noun/adjective in a compound (conjunct verb) arc  -> verb head               (बंद रहता, आनंद उठा)
  pof_cn         NNC/NNPC (compound noun / multi-word name part)  -> any head                (विमान सेवा, अरुणाचल प्रदेश)
Here: PSP, VAUX and RP dependents, NNC/NNPC dependents and every compound*/flat* arc join their head; the joined
words are then split into contiguous runs (a word group never skips a word), as in `partition_dependencies`.
Parsing differs from the paper (stanza instead of trankit), so groups can differ where the parsers disagree.
"""

from .base import check_partition
from .trankit_grouper import StanzaGrouper

JOIN_XPOS = {"PSP": "postposition", "VAUX": "aux_chain", "RP": "other", "NNC": "other", "NNPC": "other"}
RANK = {"single": 0, "other": 1, "postposition": 2, "aux_chain": 3, "light_verb": 4}


def karthika_partition(text, words):
    """words: parser words with id, head, deprel, xpos, upos and dspan (char span of the whitespace token)."""
    import re

    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    tokens = text.split()
    parent = list(range(len(tokens)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    kinds = {}
    for sent in words:
        mapped = {}
        for w in sent:
            a, b = w["dspan"]
            hit = [i for i, (s, e) in enumerate(spans) if a < e and b > s]
            if len(hit) != 1:
                raise ValueError(f"parser span {(a, b)} does not map to one whitespace word")
            mapped[w["id"]] = hit[0]
        # The paper's rules are relations between *consecutive* words. Consecutive words x, y join when
        #   y is a PSP / VAUX / RP whose head is x or x's head (chains: जाने के लिए, कर दिया गया था), or
        #   x is a compound / flat / NNC / NNPC dependent whose head is y or y's head (names, compound nouns,
        #   conjunct verbs: अरुणाचल प्रदेश, श्री ... कलाम, बंद कर).
        for x, y in zip(sent, sent[1:]):
            if "PUNCT" in (x.get("upos"), y.get("upos")):
                continue
            kind = None
            if y.get("xpos") in JOIN_XPOS and y.get("xpos") not in {"NNC", "NNPC"} and y.get("head") in (x["id"], x.get("head")):
                kind = JOIN_XPOS[y["xpos"]]
            rel = x.get("deprel") or ""
            if kind is None and (x.get("xpos") in {"NNC", "NNPC"} or rel.startswith(("compound", "flat"))) \
                    and x.get("head") in (y["id"], y.get("head")):
                kind = "light_verb" if x.get("upos") in {"NOUN", "ADJ"} and y.get("upos") == "VERB" else "other"
            if kind is None:
                continue
            a, b = mapped[x["id"]], mapped[y["id"]]
            parent[find(b)] = find(a)
            kinds.setdefault(a, set()).add(kind)
            kinds.setdefault(b, set()).add(kind)
    groups, types, i = [], [], 0
    while i < len(tokens):
        j = i + 1
        while j < len(tokens) and find(j) == find(i):
            j += 1
        groups.append(tokens[i:j])
        used = set().union(*(kinds.get(k, set()) for k in range(i, j)))
        types.append(max(used, key=lambda k: RANK[k]) if j - i > 1 and used else "single")
        i = j
    check_partition(text, groups)
    return groups, types


class KarthikaGrouper(StanzaGrouper):
    name = "karthika"

    def __init__(self, lang="hi", **kwargs):
        if lang != "hi":
            raise ValueError("karthika grouping rules are for Hindi")
        super().__init__(lang=lang, **kwargs)

    def _load(self):
        if self.pipeline is None:
            import stanza

            kw = {"lang": "hi", "processors": "tokenize,pos,lemma,depparse", "use_gpu": self.gpu,
                  "download_method": stanza.DownloadMethod.REUSE_RESOURCES, "verbose": False,
                  "tokenize_pretokenized": True}
            if self.cache_dir is not None:
                kw["dir"] = self.cache_dir
            self.pipeline = stanza.Pipeline(**kw)
        return self.pipeline

    @staticmethod
    def doc_words(text, document):
        """Parser words with whitespace spans (pretokenized: stanza token i = whitespace word i)."""
        import re

        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        out, k = [], 0
        for sent in document.sentences:
            ws = []
            for token in sent.tokens:
                for word in token.words:
                    ws.append({"id": word.id, "head": word.head, "deprel": word.deprel, "upos": word.upos,
                               "xpos": word.xpos, "dspan": spans[k]})
                k += 1
            out.append(ws)
        return out

    MAX_WORDS = 160  # parse only this many words: training reads <= 128 tokens (>= 1 token per word), and the
                     # parser's tree decoding needs O(n^2) memory (a 1,800-word IndicCorp line fails)

    def _finish(self, sentence, doc):
        words = sentence.split()
        head = " ".join(words[: self.MAX_WORDS])
        groups, types = karthika_partition(head, self.doc_words(head, doc))
        rest = words[self.MAX_WORDS:]
        groups, types = groups + [[w] for w in rest], types + ["single"] * len(rest)
        check_partition(sentence, groups)
        return groups, types

    def _parse(self, sentence):
        if not sentence.strip():
            return [], []
        memo = getattr(self, "_memo", None)
        if memo is not None and sentence in memo:
            return memo[sentence]
        return self._finish(sentence, self._load()([sentence.split()[: self.MAX_WORDS]]))

    def prefetch(self, sentences):
        """Bulk-parse `sentences` on the parser (much faster on GPU) and keep their groups for later
        group_words / group_types calls (e.g. by build_cache, which asks per sentence)."""
        import stanza

        if not hasattr(self, "_memo"):
            self._memo = {}
        todo = list(dict.fromkeys(s for s in sentences if s.strip() and s not in self._memo))
        if todo:
            docs = self._load().bulk_process([stanza.Document([], text=[s.split()[: self.MAX_WORDS]]) for s in todo])
            for s, d in zip(todo, docs):
                self._memo[s] = self._finish(s, d)

    def batch_group_words(self, sentences):
        self.prefetch(sentences)
        return [self.group_words(s) for s in sentences]
