# Experimental setup and evaluation (draft)

*Owner: Om. All numbers come from `results/*/eval_*.json` via `scripts/make_tables.py` (tables in `results/tables/`,
figures in `results/figures/`). Regenerate the tables before final edits; numbers below match the committed JSONs.*

## 1. Experimental setup

### 1.1 Models and data

**Base models.** Hindi experiments use **ganga-1b** (LingoIITGN; a Mistral-architecture 1B model with a 30k
SentencePiece vocabulary, about 1.12 tokens per word, no BOS/EOS added). Marathi experiments use
**Misal-1B-instruct-v0.1** (smallstepai; Llama architecture, 47,104-token vocabulary, BOS added, about 1.61 tokens per word).

**Training text.** IndicCorpV2 (`hin_Deva`, `mar_Deva`), raw rows ≥ 1000 (blank separator rows dropped). For the
self-distilled runs (§1.3), the training text is the base model's own greedy continuation (64 tokens) of the first
8-16 words of IndicCorp training sentences (100k texts for Hindi).

**Evaluation sets.** (i) IndicCorp eval: raw rows 0-999, about 500 sentences, disjoint from training. (ii) FLORES-200
devtest (1,012 sentences per language), out of domain. FLORES lines are stripped of outer whitespace, as in the
Hugging Face loader. A local copy built from the public FLORES-200 archive reproduces every token and in-group count
of the Hub copy.

### 1.2 Word groups

A *word group* is a content word together with the function words that attach to it (auxiliary chains such as
जा रहा था, postpositions, particles). Three groupings are used: **rules** (`hi_rules_v0`, `mr_rules_v1`; hand-written
morpho-syntactic rules), **parser** (stanza dependency parses partitioned into groups) and **random** (boundaries
drawn at random with the same group-length distribution). The random grouping is the control for every
linguistic run. Token-level labels mark, for every token, whether it starts a new group and which group it belongs to.

### 1.3 Multi-token prediction heads and training

**Heads.** A model with k heads predicts tokens t+1 … t+k from position t. Head 0 is the LM head, the verifier
at decoding time. Extra heads are either *parallel* (Medusa-style: a zero-initialised residual block on the last
hidden state, sharing the frozen LM head; every head sees only position t) or *sequential* (each head d updates a
state from head d−1's state and the embedding of the previous token, so a draft is conditioned on the drafts before
it, as in DeepSeek-V3's MTP modules, EAGLE and Hydra). Unless stated otherwise k = 4.

**Fine-tuning.** LoRA (r = 8, α = 16, dropout 0.05) on the query and value projections; AdamW, learning rate
2·10⁻⁴, weight decay 0.01, batch 8 × 128 tokens, gradient clipping 1.0, 12,500 steps (25,000 for Rsd_long);
head losses weighted 0.8^d. The extra heads pass 10% of their gradient into the shared LoRA (α_bb = 0.1, ablated in
Table 5), which keeps head 0 at next-token quality (head-0 loss within 0.01 of the NTP run R0).

**Word-group losses (R3 line).** On top of the per-head cross-entropy: (S2) a boundary probe per head, trained
with BCE to predict whether the head's target starts a new group (λ = 0.1); (S3) in-group self-consistency, a KL term
pulling head d at t towards head d−1 at t+1 on targets that lie in the same group as the context that head d−1 saw
(λ = 0.25); plus the same KL without the in-group mask (λ = 0.25). The final variant (S23_mix) was selected in
2k-step pilots. R3 trains it with the rules grouping, R7s with the parser grouping, and R6a with random groups. R4 adds a
supervised contrastive loss over groups. All of these have a second seed (s43, with shuffled data order).

**Self-distillation (Rsd line).** The base model is frozen (no LoRA update), so head 0 is the base model exactly
and decoding is guaranteed to reproduce its output. Only the extra heads are trained, on the self-distilled text
(§1.1). Rsd_seq uses sequential heads; Rsd_k6 has 6 heads. Rsd_soft uses soft labels (KL to head 0) and Rsd_long
uses twice the data and steps. For Marathi, Rsd_r8 freezes the NTP-fine-tuned R8 (so head 0 = R8 exactly).

**Hardware and precision.** Kaggle, 2 × NVIDIA T4 (16 GB). The T4 has no native bf16, so all runs use fp32
weights with fp16 autocast (and a gradient scaler during training). At decoding time, rows marked *+fast* compute every linear layer from
an fp16 copy of its weight made once (`fast_linear`, §3.4); outputs are bit-identical, and the greedy baseline uses it
too, so those speed-ups are against a faster greedy.

### 1.4 Decoding

**Self-speculative decoding.** At each step, head 0's argmax t₀ is always emitted. The extra heads draft
continuations, and **one forward pass** scores the drafts. A draft is accepted while it equals head 0's own argmax at
that position, so the output is **identical to greedy decoding with head 0**. The KV cache is rolled back to the
accepted tokens. We report the mean accepted length (tokens per step; greedy = 1) and wall-clock speed-up over
greedy decoding with the same model on the same prompts.

**Draft policies.** *FixedK*: propose all k−1 drafts. *ConfidenceCut*: stop drafting at the first head whose max
probability is below τ = 0.5. *Trees*: several candidates per position, arranged as a tree and verified in one pass
with a tree attention mask (each node attends to the prefix and its own ancestors; positions follow tree depth).
The longest path that matches head 0's argmax is accepted, and the cache keeps only that path. Tree shapes are fitted on
300 IndicCorp training prompts as Medusa sparse trees (*static*) or one tree per head-entropy cell (*entropy*),
with a node budget n ∈ {25, 40, 64}. For sequential heads, a node's token is head d's rank-r candidate given its
parent's state and token, computed batched per depth. *Prompt lookup*: chains copied from earlier occurrences of the
last 3 (else 2) tokens in the context, grafted onto the tree.

**Protocol.** 200 prompts per dataset (the first 8-16 words of evaluation sentences, identical across runs), 64
new tokens, batch size 1. For the Marathi instruct model with a frozen backbone, EOS is masked in both greedy decoding
and the verifier, because that model otherwise stops after one sentence. **Lossless check:** under fp16 autocast
81-96% of prompts match greedy decoding exactly. Every divergence occurs at a near-tie (greedy top-2 logit margin
≤ 0.016, one fp16 step). A re-run of the first 20 prompts with autocast off matches greedy **20/20 for every run and
policy** (one R2 case is 19/20 at an exact fp32 tie, margin 0.0).

## 2. Evaluation metrics

- **Per-head loss / top-1** on the evaluation text, teacher-forced. Head d at position t predicts token t+d+1.
- **In-group / at-boundary top-1:** top-1 restricted to targets that continue the source token's word group vs
  targets that start a new group (Figure `fig_ingroup_bars`).
- **Mean accepted length** and **speed-up** (§1.4).
- **Group Integrity (GI):** of the accepted multi-token spans (≥ 2 tokens), the share that end on a word-group
  boundary. GI measures the proposal's *phrase coherence*: whether the drafted spans respect word groups. Because
  GI is computed from stored decoding outputs, it is re-scored with a grouping the model was **not** trained on
  (rules vs parser), which separates genuine alignment from fitting one's own labels.
- **Statistics:** paired bootstrap over prompts (2,000 resamples, 95% CI). The prompts are identical across runs, so
  differences are paired.

## 3. Results

### 3.1 Speed (Tables 1, 4, 5; Figure `fig_speed_ladder`)

Plain multi-token heads give a modest speed-up: R2 (3 parallel heads) accepts 1.39 tokens/step, ×1.15 on T4
(FLORES ×1.12). **Self-distillation** is the first large gain: training the heads on the model's own text raises
acceptance to 1.67 (×1.36 / ×1.32) while head 0 stays exactly the base model, even though the heads become *less*
accurate on human text (head-1 top-1 13.9% vs 15.0%). **Sequential heads** raise it further to 2.00 tokens/step
(×1.61 / ×1.52; replicated with a second seed, ×1.60 / ×1.54). **Tree verification** adds the largest single
step: Rsd with a 40-node entropy tree reaches ×1.74 / ×1.68, and **Rsd_seq with a 40-node static tree reaches
×2.02 / ×1.94** (2.79 / 2.67 tokens/step). **Prompt-lookup chains** grafted onto the tree add another +0.6
tokens/step: **Rsd_seq + entropy n40 + lookup reaches 3.40 / 3.56 tokens/step, ×2.12 / ×2.22**, and Rsd + lookup is
equally fast (×2.12 / ×2.23), so once copied chains are available the sequential heads' advantage disappears. All of
these are lossless. Part of the lookup gain comes from the 1B model's greedy output repeating itself, so sampling
would gain less from it. **Engine:** autocast re-casts every frozen fp32 weight to fp16 on every call (72% of the draft chain's GPU
time, about a quarter of each forward pass); computing linear layers from fp16 weights cast once is bit-identical and
raises greedy from 27.6 to 36.5 tok/s. Against that faster greedy, **Rsd_seq + tree + lookup reaches ×2.64 / ×2.70
(96-99 tok/s), ×3.5 over the original greedy decoder with identical output**. A second seed of Rsd_seq reproduces it
(×2.65 / ×2.73).

The T4 at batch size 1 is memory-bound: scoring 64 positions in one pass costs only 1.12-1.30× a greedy pass
(`bench_verify`), which is why wide trees are affordable. Two costs bound the gains: each extra parallel head adds
about 2 ms per pass (Rsd_k6 is slower than Rsd as a chain and as a tree), and each sequential draft step adds about
2.3 ms (Rsd_seq_k6, ×1.53, is slower than Rsd_seq). For sequential heads, static trees beat entropy trees (×2.02 vs
×1.79 at n = 40) because entropy cells need the full greedy draft chain before the tree is built. Reducing per-step
engine overhead (batched device syncs) was worth about +0.1 speed-up on its own (Rsd_seq static n64: ×1.89 → ×2.01).
Real acceptance is 91-93% of the offline tree oracle. **ConfidenceCut is slower than FixedK on every model**: drafts
are nearly free to verify, so cutting them early wastes them.

Marathi follows the same pattern: R9 ×1.07 / ×1.05, Rsd_r8 ×1.37 / ×1.36, and Rsd_r8_seq ×1.44 / ×1.45 (Table 2).
Sequential heads add less in Marathi (+0.09 tokens/step) than in Hindi (+0.33). Trees lift Rsd_r8_seq to ×1.77 / ×1.75
and prompt lookup to ×2.35 / ×2.33 (3.87 / 3.77 tokens/step, Table 4); the lookup gain is larger than in Hindi
(+1.45 vs +0.6 tokens/step), consistent with more repetitive greedy output from the Marathi model. With the fp16 weight
cache (greedy 22.9 → 26.4 tok/s), **Rsd_r8_seq + static n64 tree + lookup reaches ×3.01 / ×2.94 (80 / 78 tok/s), ×3.5
over the original greedy decoder**, and its second seed reproduces it (×2.99 / ×2.96). With the same
tree (static n64 + lookup) the cache lifts the Marathi speed-up from ×2.33 / ×2.31 to ×3.01 / ×2.94, because it barely
speeds up Misal's greedy pass (43.6 → 38.8 ms) but cuts its 3-step draft chain from 10.4 to 3.3 ms (re-casting the
47k-token LM head was expensive) and its 64-position verify pass from 45.8 to 36.9 ms (`bench_verify_fast.json`).

### 3.2 Word groups (Table 3; Figures `fig_ingroup_bars`, `fig_gi_forest`)

**Accuracy.** The structural loss raises in-group top-1 of the far heads over R2 (R3: +2.4 / +5.1 / +4.7 points
at h1-h3 on IndicCorp), but **random groups give most or all of the same gain**. R3 − R6a is +0.6 / +2.5 / +2.3
on seed 42 and −1.5 / −0.2 / +1.8 on seed 43, and Marathi R10 = R10a on both seeds. The accuracy gain comes from the
extra loss terms, not from the linguistic grouping. Acceptance and speed are unchanged across the whole R3 line
(1.38-1.40 tokens/step).

**Boundary alignment.** Group Integrity tells a different story. Linguistic groups beat their random control in all
8 cells of the new unfrozen pairs (R7s, R7s_s43, R4, R10_s43 vs controls, both datasets). The decisive test re-scores
GI with a grouping the model never saw. **R7s, trained on parser groups, beats R6a under the rules grouping on
IndicCorp for both seeds: +4.3 [+2.4, +6.4] and +2.6 [+0.9, +4.3]**, and under its own parser grouping +4.1 and +3.5
(CIs exclude 0). Rules-trained runs (R3, R4) are significant only under their own rules and lose significance when
scored with the parser, so part of their effect is label fitting. Out of domain (FLORES) the direction holds in 9
of 10 estimates, but only the seed-42 rules-scored cells are significant. The same holds on the frozen self-distilled line: **Rsd_struct beats its random control by +1.9 [+0.8, +2.9]
(IndicCorp) and +1.4 [+0.4, +2.4] (FLORES)**, at identical speed. On the frozen sequential-head model, the group
loss changes neither speed nor GI (Rsd_seq_struct = its random control).

**By group type (Table 6, `scripts/decode_by_group_type.py`).** Filing each decoding step under the `hi_rules_v0` type
of the group its head-0 token falls in (FixedK, R3 line, both seeds): steps in single-word groups (about 52% of
steps) accept least (1.27-1.31 tokens/step on IndicCorp) and their spans end on a boundary least often (GI 58-63%);
auxiliary chains accept most (1.63-1.70, GI 84-89%) and postposition groups sit between (1.43-1.45, GI 85-90%). Steps
that start inside a group accept more than steps that start a new one (1.50-1.57 vs 1.33-1.35): continuing a group
is the easy part of drafting. The linguistic runs' GI advantage over random groups sits in the multi-word groups:
on IndicCorp, auxiliary chains 86-89% vs 84% (both seeds) and postpositions 88-90% vs 85-87%, while acceptance per
type is the same within ±0.07. These per-type cells are descriptive (no CIs); the tested GI comparisons are above.
(hi_rules_v0 assigns only these three types; the per-prompt outputs needed for this table exist for the R3 line, not
for the older R2 / Rsd evals.)

**Summary.** Supervising MTP heads with word groups does not make them more accurate than random groups do, and it
does not speed up decoding. But training on **syntactic (dependency-parser) groups makes the accepted draft spans end
on word-group boundaries more often, under both its own and an independent grouping**, at no cost in accuracy or speed.

### 3.3 Ablations (Table 5)

α_bb = 1 (heads train the shared LoRA fully) gives better heads (+0.05 tokens/step, ×1.18 vs ×1.15) but a worse
verifier (+0.14 nats head-0 loss); α_bb = 0.1 keeps head 0 within 0.01 of α_bb = 0. Soft labels do not help (Rsd_soft
= Rsd, Hindi and Marathi), and neither do contrastive learning (R4 = R3) or adaptive loss weighting (pilots flat when
head 0 is protected). Twice the data and steps give a small gain (Rsd_long ×1.41 vs ×1.36).

### 3.4 Drafter design: profiling, and what did not pay (`docs/runs.md`, drafter sections)

After trees and lookup, we asked whether a better or cheaper drafter could still raise the speed-up. A decoding step
costs one verify pass plus the draft, so speed-up ≈ tokens/step ÷ ((verify + draft) / greedy). Every candidate was
first tested as a 2k-step pilot against an Rsd_seq pilot trained on the same 20k self-distilled texts with the same
seed, scored by paired bootstrap on tokens/step and kept only if it raised the speed-up by ≥ 0.05× with an fp32
re-check of 20/20.

**Profiling found the real cost.** A draft step was dominated not by the drafter but by autocast re-casting the frozen
fp32 LM head on every call (2.01 ms vs 0.54 ms with a weight cast once; casts were 72% of the draft chain's GPU time).
Caching fp16 copies of all linear weights (`fast_linear`) shrank the 3-step draft chain from 7.0 to 2.4 ms and the
4-position verify pass from 36.3 to 26.7 ms, and is the source of the ×2.64 / ×2.70 result in §3.1.

**Cheaper word predictors do not help after that.** Restricting the draft LM head to the 16k most frequent tokens keeps
98.5-99% agreement with the full head but saves at most 0.25 ms per step. A low-rank (SVD) LM head agrees with the full
head on only 11-37% of drafts at ranks 128-512: the output layer is far from low-rank.

**Richer drafters guess slightly better but cost more than they gain.**
- *Context attention* (`seq_ctx`): each draft slot additionally attends over the whole context (4 heads, width 256,
  zero-initialised output). Acceptance rises by +0.03 to +0.06 tokens/step on chain, tree and tree + lookup, on both
  datasets (all CIs exclude 0), but the draft chain grows from 2.59 to 3.07 ms and the speed-up changes by +0.01 to −0.09.
- *Wide working state* (`seq_wide`): the drafter keeps a state of D > H numbers, z₀ = [h_t; SiLU(U h_t)], reads one
  word out per step through a learned D → H projection and folds the word back in. Offline, only 15-29% of the variance
  of the next three hidden states is predictable from h_t, and that part fits in 400-1000 dimensions (95%), fewer than
  H = 2048. In pilots (D ∈ {H, 1.5H, 2H, 3H}, per-step or shared update) the best (1.5H) gains +0.044 tokens/step
  (+0.024, +0.064) for +0.33 ms per chain, a +0.014 change in speed-up. The gain *shrinks* as D grows (3H: +0.017, n.s.),
  and with a linear expansion (z₀ = [h_t; U h_t]) it vanishes, so the benefit is one extra non-linear feature of h_t,
  not working space.

**Lesson.** At batch size 1 on a T4 the draft chain is a fixed per-step cost on top of an almost free verify pass. With
a decoding step of ~30 ms (verify ~27 ms + draft ~2.6 ms), a drafter change must add about 1% tokens/step for every
0.3 ms it costs just to break even, and about 3% more to clear the 0.05× threshold. Both richer drafters improved
acceptance by 2-3.5% while costing 0.3-0.5 ms, which is real but not enough; removing engine overhead
(`fast_linear`) was worth far more than any change to the drafter itself.

## 4. Figures and tables to include

| item | file | content |
|---|---|---|
| Table 1 | `results/tables/table1_main_hi.tex` | Hindi main results |
| Table 2 | `results/tables/table2_main_mr.tex` | Marathi main results |
| Table 3 | `results/tables/table3_controls.tex` | linguistic vs random groups, GI under two groupings with CIs |
| Table 4 | `results/tables/table4_decoding.tex` | draft policies and trees |
| Table 5 | `results/tables/table5_ablations.tex` | ablations |
| Table 6 | `results/tables/table6_group_types.md` | decoding by word-group type (OM-10) |
| Figure 1 | `results/figures/fig_speed_ladder.pdf` | speed-up from greedy to sequential heads + trees |
| Figure 2 | `results/figures/fig_ingroup_bars.pdf` | per-head top-1, in-group vs boundary |
| Figure 3 | `results/figures/fig_gi_forest.pdf` | GI difference linguistic − random, 95% CIs, two groupings |
| Figure 4 | `results/figures/fig_lookahead.pdf` | lookahead heatmap of one sentence (R7s) |

All tables and figures are final as of the committed results (regenerate with `python scripts/make_tables.py`).
