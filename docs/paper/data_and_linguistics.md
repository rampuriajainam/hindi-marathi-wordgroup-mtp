# Data, word groups and linguistic analysis

Owner: Jai. Numbers below are from committed measurements, not newly executed
model experiments. This section complements [Om's evaluation section](setup_and_evaluation.md).
Bibliography keys are in [references.bib](../references.bib).

## Corpora and operational group definition

We use the Hindi (`hin_Deva`) and Marathi (`mar_Deva`) splits of IndicCorpV2
([Doddapaneni et al., 2023](https://aclanthology.org/2023.acl-long.693/)) and
FLORES-200 devtest
([NLLB Team, 2022](https://arxiv.org/abs/2207.04672)). IndicCorp splits follow raw
row indices: evaluation uses rows 0–999 and training starts at row 1000; blank
separator rows are removed after splitting. FLORES is read from the original
public archive rather than replaced with a different benchmark.

A word group is a contiguous span of whitespace words, such as `जा रहा था`,
`घर से`, `घर के बारे में` or `कर दिया`. Every word belongs to exactly one group.
Suffixes and punctuation attached to words remain intact. The annotation scheme
distinguishes auxiliary chains, simple and compound postpositions, light verbs,
singletons and `other`. Negation and वाला constructions receive `other` with
notes in human annotations; rule outputs use their documented attaching category.
Attachment scores are independent of that type convention. Adjective–noun
sequences, coordination and number–noun sequences are not merged merely because
they form a phrase. The complete decisions are in [the annotation guide](../annotation_guide.md).

Hindi v0 uses short function-word lists. Hindi v1 expands those lists, matches
compound postpositions longest first and adds documented heuristics for light
verbs and negation. Marathi v1 preserves fused case suffixes and attaches only
separately written auxiliaries, particles and case expressions. Its low word-level
attachment rate is a property of this operational grouper, not a direct measure
of the language's syntactic complexity. The separately named `stanza` adapter
uses dependency attachments and splits noncontiguous components; the toolkit is
described by [Qi et al. (2020)](https://aclanthology.org/2020.acl-demos.14/).

## Annotation design and current evidence

**Gold status (2026-10-22).** All 200 Hindi and 150 Marathi sentences are
human-reviewed gold: each was first annotated by a fresh-context AI pass and checked by
a separate AI review using the same guide, then reviewed by Jai, who confirmed the
annotations (`annotation_status: human_reviewed`, `reviewed_by: jai`). The AI origin of
every record is kept. Agreement (κ) is not yet available: it needs Om's independent
50-sentence Hindi annotation. See [the annotation provenance](../../data/gold/README.md)
and `results/grouping/annotation_gold_review_2026-10-22.json`.

The planned Hindi set has 200 sentences (100 FLORES, 100 IndicCorp); Marathi has
150 (75 from each). Candidates contain 6–25 whitespace words, are unique by text,
and retain source row IDs. Sampling uses seed 42. Om's independent, text-only
Hindi packet contains 50 selected IDs, balanced 25/25 across sources. Agreement
must be computed before adjudication, using every between-word position as an
attachment/nonattachment decision.

*Historical (before the 2026-10-22 gold):* at the 2026-10-19 repository audit, the
saved Marathi file contained **55 explicitly reviewed records**, attributed to its
recorded annotator, Ruchir Padhye (47 FLORES, 8 IndicCorp); they are preserved
unchanged. The partial scores below are from that source-skewed subset, not from the
full 150-sentence gold.

The historical partial evaluation on those 55 reviewed sentences is:

| Grouper | Attachment precision | Recall | F1 | Exact gold-span recall |
|---|---:|---:|---:|---:|
| Marathi rules v1 | 0.3607 | 0.4583 | 0.4037 | 0.8796 |
| Stanza | 0.3263 | 0.6458 | 0.4336 | 0.8323 |
| Length-matched random v1 | 0.0000 | 0.0000 | 0.0000 | 0.8689 |

Source: `results/grouping/mr_scores_partial55.json`; these are existing results,
not a parser rerun. Attachment F1 counts within-group word edges. The existing
`exact_group_accuracy` field is matched gold spans divided by all gold spans,
so singleton-heavy data can give random grouping a high value. Per-type scores
require both the span and type to match. The older 90-row report mixes reviewed
and pending rows and is explicitly excluded from linguistic accuracy claims.

## C3: the reach of grouping across tokenizers

Each language uses 5,000 training sentences; Hindi and Marathi have separate
samples. Every tokenizer/grouper pair within a language sees identical texts.
Statistics use full texts without truncation, exclude special tokens, and count
only valid token pairs within a sentence. For lookahead d, reach is the fraction
of positions with both t and t+d present whose group IDs agree. Sample hashes and
raw denominators are retained in `results/grouping/stats_{hi,mr}.json`.

Hindi v1 groups average 1.4100 words, compared with 1.3284 for v0. Under ganga's
tokenizer they average 1.5817 tokens, compared with 1.4901 for v0. The measured
Hindi v1 same-group rates are:

| Tokenizer | d=1 | d=2 | d=3 | d=4 |
|---|---:|---:|---:|---:|
| ganga-1b | 37.37% | 10.47% | 3.05% | 0.96% |
| Misal-1B | 61.85% | 33.32% | 16.27% | 6.95% |
| mT5-small | 65.77% | 34.71% | 15.44% | 6.08% |
| BLOOM-560m | 47.16% | 19.07% | 6.77% | 1.98% |
| XLM-R | 51.42% | 23.83% | 9.49% | 3.10% |

Marathi v1 averages 1.0569 words/group, 1.6399 tokens/group with Misal and 2.5947
with ganga. Its Misal reach is 39.64/15.08/4.85/1.68%; under ganga it is
62.07/34.19/15.77/6.30%. Tokenization therefore changes the number of positions
covered by a group substantially even when the word grouping is fixed. This
supports interpreting structural supervision relative to tokenizer granularity;
it does not establish that a finer tokenizer improves prediction or decoding.
Ganga's Hindi same-group signal is sparse at long lookaheads, consistent with
the limited scope of the structural hypothesis. This is an interpretation of
the reach measurement, not a causal explanation of the training results.

Include `results/figures/reach_hi.pdf`, `reach_mr.pdf` and
`group_lengths_hi_mr.pdf`. The older 128-token cache statistic (1.5740 Hindi
tokens/group) uses truncation and should not replace the full-text 1.5817 value.

## C5 and token-level error analysis: protocol ready, results deferred

The C5 runner freezes ganga-1b and extracts every hidden layer at position t to
predict the boundary status of token t+1. It trains CPU logistic probes with
one streaming SGD epoch on up to 64 sampled positions per training sentence,
using 2,000 requested training sentences and the held-out evaluation split.
Exact duplicate evaluation texts are removed from training. All label sets
share one model forward per sentence. Controls include word starts, legacy
random groups and the v1 length-matched random groups. Boundary prevalence,
majority baselines, boundary F1, attachment F1 and macro F1 must be interpreted
together: the random control's boundary F1 need not be numerically near zero.
No preferred layer or full-model probe result is claimed before Jainam's run.

The error-analysis runner compares matched per-token dumps from R2, R3 and R6a
(and R5/R6 if those variants are actually run). It assigns target group type
and within-group position using one common tokenizer/grouper, checks keys and
texts across every run, and exports a table plus up to ten improved and five
worse far-head examples per comparison. Aggregate evaluation JSONs cannot
recover these examples. The required dumps are absent from this checkout.

## Relation to the completed model results

The completed experiments in Om's section show that linguistic structure does
not reliably outperform random-group controls on prediction accuracy or decoding
speed. Parser supervision has stronger evidence for boundary alignment under an
independent grouping. The C3 measurements describe the short span of the available
signal; annotation and probing address label quality and representation, respectively.
They must not be presented as evidence that structure accelerated decoding.
