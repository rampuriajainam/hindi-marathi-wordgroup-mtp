# What the drafter learns from word groups, and why it does not pay (draft)

Jainam's section. Setup, metrics and the main speed / word-group tables are in `setup_and_evaluation.md` (Om); the
groupers, annotation and C3 are in `data_and_linguistics.md` (Jai). Every number below is read from the file named
next to it. Hindi, ganga-1b, parallel resblock heads (k = 4: head 0 verifies, heads 1-3 draft), unless stated.

## 1. Question and controls

The proposal expected that teaching the draft heads Hindi word groups (a noun with its postpositions, a verb with its
auxiliary chain) would make drafting faster or more accurate. Every comparison here is **linguistic groups vs random
groups with the same size distribution** (words kept whole), trained with the identical recipe and seed, and evaluated
on the same 200 IndicCorp and 200 FLORES prompts (Kaggle T4, paired bootstrap over prompts or sentences, 2,000
resamples, 95% CI). Pairs:

| pair | grouping | group losses | seeds |
|---|---|---|---|
| R3 / R6a | ours (`hi_rules_v0`) | S2 + S3 (×1) | 42, 43 |
| R11 / R11a | Karthika et al. (2025) (Table 4 rules on the stanza HDTB parse; `karthika`) | ×1 | 42, 43 |
| R12 / R12a | ours | S2 and S3 terms ×4 (λ_S2 0.4, λ_S3 1.0) | 42 |
| R4 / R4a | ours | ×1 + SupCon contrastive | 42 |

The Karthika et al. grouping (our reimplementation) gives 1.60 words per group on FLORES devtest vs 1.35 in their paper and 1.34
for `hi_rules_v0`: it is about 15% coarser, so it is a second, independent grouping, not a copy of ours.
Human gold for grouper scoring: 200 Hindi and 150 Marathi sentences, AI-drafted and reviewed by Jai
(`data/gold/README.md`); no inter-annotator agreement yet.

## 2. The structure is learned

**Group Integrity (GI)**: of the decoding steps that emit ≥ 2 tokens, the share whose chunk ends on a group boundary.
Linguistic − random, points (IndicCorp / FLORES; * = CI excludes 0):

| pair | GI scored with our rules | GI scored with the Karthika et al. grouping | tokens/step |
|---|---|---|---|
| R3 − R6a, seed 42 | (Om, Table 3) | +2.5* / +2.2* | n.s. |
| R3 − R6a, seed 43 | (Om, Table 3) | +1.7 / +1.4 | n.s. |
| R11 − R11a, seed 42 | +3.3* / +0.9 | +2.6* / +3.0* | −0.01 / −0.02* |
| R11 − R11a, seed 43 | +2.3* / +1.4 | +2.5* / +1.4 | +0.01 / −0.01 |
| R12 − R12a (×4) | **+6.6* / +5.1*** | | −0.02 / −0.03* |

Sources: `results/grouping/gi_karthika_rescore.json`, `r11_official_compare.json`, `r12_official_compare.json`.
GI rises with the dose of the group losses (×1: +1 to +3, ×4: +5 to +7) and shows up whichever grouping trains or
scores it. Top-1 accuracy of heads 1-3 moves by ≤ 0.3 points in every pair.

**Representations.** Centred cosine similarity of adjacent cross-word token pairs, within − across groups
(`results/grouping/coherence_hi.json`, 500 IndicCorp sentences). The base model already separates groups at layer 12:
+0.063 (rules), +0.069 (stanza), +0.075 (Karthika et al.), random groups ≈ 0. Training changes this only with the contrastive
loss: R4 − R4a +0.0085 [+0.0083, +0.0087] at layer 16; the S2/S3 losses do not (R3 − R6a −0.003).

**Shuffle test** (her paper's test, inside our LM; `results/grouping/shuffle_test_hi.json`, 1,012 FLORES sentences).
Shuffling whole groups keeps the sentence closer to the original than shuffling words: layer-12 similarity gap +0.119
(Karthika et al. groups) vs +0.082 (size-matched random), +0.073 (our rules) vs +0.055 (random); NLL gap 1.56 / 0.90 / 0.91 /
0.57 nats. Training changes the similarity by ≤ 0.001.

**Probing (C5)** (`results/probing/hi_*.json`): a linear probe recovers group boundaries from the hidden state with
macro-F1 0.84 (`hi_rules_v0`) and 0.83 (`hi_rules_v1`) at the best layer, vs 0.59-0.61 for random groups and 0.87
for plain word boundaries.

So the linguistic groups are real structure in ganga-1b, the heads pick them up, and the effect grows with the loss.

## 3. Why it does not make drafting faster

**The drafter trades group starts for group continuations.** Under speculative sampling a draft from q is accepted
with probability Σ min(p, q) (p = verifier). Teacher-forced estimate per head, split by whether the target token
continues a group (`in`) or starts one (`start`; `results/drafter/sampling_acceptance.json`, T = 1, head 1, points,
IndicCorp / FLORES, every cell * ):

| pair | in-group | group start | all |
|---|---|---|---|
| R12 − R12a (×4) | +5.3 / +4.2 | −2.8 / −2.2 | −0.2 / +0.2 |
| R3 − R6a | +2.4 / +1.8 | −0.9 / −0.7 | +0.2 / +0.3 |
| R11 − R11a | +1.6 / +1.3 | −0.6 / −0.5 | +0.1 / +0.2 |
| R11 − R11a, seed 43 | +1.4 / +1.1 | −0.5 / −0.5 | +0.1 / +0.1 |

The same sign pattern holds for heads 2 and 3, at T = 0.7, and in every pair; the trade doubles from ×1 to ×4. About
a third of targets continue a group, so the two effects cancel and accepted length does not move.

**What the heads propose** (`results/drafter/draft_text.json`, 300 sentences per dataset, every third position; the
draft block [t0, d1, d2, d3] is judged on its own, not by matching):

| linguistic − random, head 1 | R12 − R12a | R3 − R6a | R11 − R11a |
|---|---|---|---|
| share of drafts that are one of the 50 most frequent tokens (के, में, है, ...) | +11.4* / +7.9* | +5.4* / +3.4* | +4.4* / +3.3* |
| draft NLL under plain ganga (nats; + = less fluent) | +0.35* / +0.29* | +0.13* / +0.05* | +0.06* / +0.09* |
| wrong drafts that are content words | −9.1* / −7.3* | −4.8* / −3.0* | −3.5* / −3.3* |
| wrong drafts that are inflection siblings | ≈ 0 | ≈ 0 | ≈ 0 |

Group training teaches a word-class prior: inside a group the next token is usually a postposition or auxiliary, so
the heads bet on them. Every model already over-drafts these tokens (55-76% of drafts vs 32% of references, R2
included); group training makes it stronger. The drafts do not become better Hindi: they are slightly less fluent,
and the errors move from content words to function words and repeats.

**Parallel heads stutter.** In 60-69% of draft blocks two adjacent drafts are the same token ("के के"), in every
model: parallel heads guess without seeing each other's guesses. Structural losses add 4-5 points over R2. Sequential
heads (`Rsd_seq`, Om's §3.1) remove this by construction.

**Where the gain could come from, and does not.**
- *Stop at the group end*: fixed-length drafting beats stopping at the predicted group end for every model and group
  size; the group-end stop beats a random stop by ≤ 0.014 tokens/step (`results/grouping/stop_oracle_*.json`).
- *Sub-word structure*: 90% of target tokens start a word (ganga's 30k vocabulary, ~1.12 tokens/word); there is little
  inside-word structure for groups to add (`token_structure_*.json`).
- *Verb anticipation* (Hindi is SOV): main verbs are 10.6% of targets and ~11% of far-head errors, and a probe beats
  the majority baseline (26.7%) only one token before the verb (40%) (`verb_anticipation.json`).
- *Inflection siblings* (जाता / जाते) are 1-5% of wrong drafts (`results/drafter/inflection_*.json`); a phrase table on
  top of the heads moves tokens/step 2.633 → 2.650 (`phrase_table_Rsd_seq.json`).

## 4. Can the learned structure be used at decoding time?

**Lossy (typical) acceptance.** Medusa's rule: accept a draft x if p(x) ≥ min(ε, √ε·exp(−H(p))), so plausible
alternatives pass where the verifier is unsure (`mtp/eval/typical.py`). This is the one setting where the drafter's
own choices reach the output. Sweep ε ∈ {0.5, 0.2, 0.1, 0.05, 0.02} on R12/R12a, R3/R6a (2 seeds), R11/R11a
(2 seeds), R2: Kaggle T4, 200 prompts × 64 tokens per dataset; quality = perplexity of the output under plain
ganga-1b (`results/lossy/`, `results/drafter/lossy_compare.json`). Going lossy works as expected (R12a, IndicCorp:
1.38 → 1.97 tokens/step, ppl 2.6 → 6.9).

At the same ε the linguistic models look better (lower ppl, e.g. R12 − R12a −0.5 to −0.6 at ε 0.5-0.1), but they also
accept fewer tokens (−0.02 to −0.07 tokens/step) and repeat more (4-gram repeats up to +0.08), both of which lower
perplexity for free. Compared fairly, **at matched tokens/step and on tokens that do not close a repeated 4-gram**, over
45 testable (pair, dataset, ε) cells: linguistic better in 1, worse in 11, no difference in 33. No benefit.

**Down-weighting function words** in the drafts at inference, everywhere or only where the head's own boundary probe
predicts a group start (`scripts/function_penalty.py`, `results/drafter/function_penalty.json`; logit penalty
β ∈ {0.5, 1, 2, 3} on the 50 most frequent tokens, R12 / R12a / R3 / R6a, 300 sentences per dataset): no setting
raises the tokens/step proxy of any model; the mildest (gated, β 0.5) is at best neutral (R12: −0.001 / +0.001, n.s.)
and costs up to −0.011 on the others, and larger β costs up to −0.12. The gated penalty hurts the linguistic runs less than their random twins (their boundary probes are
better), but never beats no penalty. At a group start the drafter is uncertain, not miscalibrated: pushing it off
के / है does not make it guess the content word.

**Routing between drafters by group position** (`scripts/route_drafters.py`, `results/drafter/route_drafters.json`).
If the drafter knew whether its target starts a group, it could keep the in-group gain without the start loss. On the
frozen-backbone line the heads of different runs share the same verifier (head 0 = ganga-1b; 0 mismatched verifier
tokens), so they can be swapped: group-trained heads (Rsd_struct) where the target continues a group, random-group
(Rsd_struct_rand) or plain self-distilled (Rsd) heads where it starts one. Ceiling, with the true boundaries:
+0.005 to +0.008 tokens/step on IndicCorp (CI excludes 0), +0.001 to +0.002 on FLORES (n.s.). Switching on the S2
probe (68-72% accurate on group starts): −0.005 to +0.000. Every drafter is right only 17-20% of the time at a group
start, so telling it that a group starts does not tell it which word comes; on this line the start loss is also small
(−0.3 to −0.6 pts). Not pursued.

## 5. Marathi: morphemes and word boundaries

Phrase groups do nothing in Marathi (R10 − R10a: GI +0.3 to +2.2, none significant; Om's §3.2). Two checks on why.

**Morpheme groups are not reachable through Misal's tokens** (`scripts/morph_feasibility.py`,
`results/grouping/morph_feasibility_mr.json`). MorphTok's segmentations (karthika95/morphTok, 61,737 Marathi words;
Brahma et al. 2025) cover 33-37% of word occurrences in our eval text; 55% of covered words have two or more
morphemes. Matching suffixes right to left places 96% of their boundaries in the surface word despite sandhi
(वाघिणीसह = वाघीण+सह), but only **10-13% of those boundaries fall on a Misal token boundary** (11.5-13.4% allowing a
vowel-sign shift): frequent inflected words are single tokens (▁होतो, ▁वापरले, ▁करणे). That leaves 2.0-2.6 usable
morpheme boundaries per 100 words, against ~25 group joins per 100 words for Hindi `hi_rules_v0`; for ganga-1b and
Hindi it is 0.4. The tokenizer has already absorbed most of Marathi's word-internal morphology, so token-level groups
have little left to mark.

**Word-boundary groups: the Hindi trade-off again, net negative** (R13 / R13a, two seeds). 31% of Marathi tokens
are word-internal (9% in Hindi), and every earlier random control kept words whole, so word boundaries themselves were
never tested. R13 = R10's recipe with groups = whole words; R13a = `random_tokens`: the same tokens-per-group histogram
(1.58 vs 1.54 tokens per group) with boundaries at random token positions, so its group starts fall on a word start at
chance level (63.8% vs 63.3%) and the two groupings disagree on 45% of boundary decisions
(`scripts/fit_token_histogram.py`). Official eval: Kaggle T4, groups = words, 200 prompts, fp32 re-check 20/20
(`results/grouping/r13_official_compare.json`); R13 − R13a, IndicCorp / FLORES:

| | seed 42 | seed 43 |
|---|---|---|
| tokens/step | −0.03* / −0.02* | −0.02* / −0.01 |
| top-1, heads 1-3 | within ±0.4 pts | within ±0.4 pts |
| head-1 top-1 on word-internal targets | +0.7 / +1.0 | +1.2 / +1.4 |
| GI, words / `mr_rules_v1` | +2.9 / +2.3, +2.7 / +1.5 (n.s.) | +1.5 / +0.7, +1.4 / +1.9 (n.s.) |

Sampling acceptance split by word position (`results/drafter/sampling_acceptance_mr_words.json`, 500 sentences per
dataset, head 1, T = 1, every cell *): continuing a word +1.4 / +1.2 (s43 +1.6 / +1.3), starting a word −1.6 / −1.3
(s43 −1.7 / −1.5), overall −0.4 / −0.3 (s43 −0.5 / −0.4). Heads 2-3 are also negative overall. The mechanism of §3
replicates with a different structure in a second language, and here the loss at word starts outweighs the gain, so
word-boundary supervision makes Marathi drafting slightly slower.

## 6. Claims and limits

1. Linguistic word groups are real structure in ganga-1b (probe, shuffle test, coherence), and MTP heads trained with
   group losses absorb them: GI +1 to +3 points at the normal loss, +5 to +7 at ×4, under both our and the Karthika et al. grouping.
2. They absorb it as a word-class prior that helps inside groups and hurts at group starts by about the same amount,
   so accepted length, tokens/step and accuracy do not change (all pairs, both seeds, both datasets).
3. The prior does not make drafts better Hindi, and gives no quality gain under lossy decoding at matched speed.
4. Parallel heads produce repeated drafts in most blocks; sequential heads, not linguistic supervision, are what
   raised speed in this project (Om's §3.1).
5. In Marathi, the tokenizer already absorbs most morphology (87-90% of MorphTok morpheme boundaries fall inside a
   token), and word-boundary supervision shows the same continue-vs-start trade-off, net negative (−0.01 to −0.03
   tokens/step, two seeds).

Limits: one base model per language; groups are short in tokens (~1.5); the Karthika et al. grouping is our reimplementation of
their rules (15% coarser than their reported counts); MorphTok covers only a third of Marathi word occurrences and our
boundary placement under sandhi is heuristic; the sampling-acceptance and draft analyses are teacher-forced
estimates; the lossy quality judge is a perplexity, which is blind to some agreement errors; only ×1 and ×4 loss
weights in Hindi, ×1 in Marathi.
