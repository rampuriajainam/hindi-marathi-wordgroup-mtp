# Word Groups and Self-Speculative Decoding for Hindi and Marathi

**A controlled study, and where the speed comes from.**
CS769, IIT Bombay · Jainam Rampuria · Jai Sheth · Om Dadoo · TA: N J Karthika

We made 1B-parameter Hindi and Marathi models **about 3× faster on one T4 GPU with exactly the same output**, and tested whether linguistic word groups help the model guess further ahead. They don't.

## The problem

A language model writes one token per forward pass. For a 1B model serving one request, the GPU mostly waits on memory, so checking 64 positions in one pass costs only **1.1–1.3×** one position. **Speculative decoding** uses that idle capacity: guess several tokens cheaply, check them all in one pass, keep the guesses the model agrees with. A guess is kept only if it is exactly what the model would have picked, so the output is identical to greedy decoding.

**Our three questions**
1. Can 1B Hindi and Marathi models generate faster on one T4, with output identical to greedy?
2. Do Hindi word groups help the drafter guess further ahead? (our starting hypothesis)
3. Does the speed-up survive sampling, less repetitive text and other tokenizers?

## Our drafter

Small extra **heads** on top of the frozen model: head 0 is the model's own output layer (the checker), heads 1–3 guess the 2nd, 3rd and 4th token ahead. **Sequential** heads also see the previous head's guess.

**Word groups.** Hindi writes postpositions and auxiliaries as separate words: `[राम ने] [बाज़ार से] [फल] [खरीदे थे]`. Inside a group the next word is predictable (after जा रहा comes था), so heads that know about groups might guess further. Three groupers: our rules, a dependency parser (Stanza), and Karthika et al. (2025). Every linguistic run has a **random twin**: random groups of the same sizes, trained identically.

**Training objective.** The usual per-head loss (weighted 0.8^d), plus three group losses: *boundary* (does a new group start here?), *agree-in-group* (inside a group, the far head learns from the nearer head) and *agree-everywhere* (the same on every token, no groups: a control). Loss weights were chosen with 2,000-step pilots.

## Setup

| | Hindi | Marathi |
|---|---|---|
| Base model | ganga-1b (1B) | Misal-1B + our LoRA |
| Vocabulary · tokens per word | 30k · 1.12 | 47k · 1.59 |
| Training text | IndicCorpV2 | IndicCorpV2 |
| Test text | in-domain: held-out IndicCorp; out-of-domain: FLORES-200 | same |
| Hand-checked groups | 200 sentences | 150 sentences |

One T4 (Kaggle), one request at a time, 200 prompts × 64 new tokens. Heads trained 12,500 steps, 8 texts per step (≤ 128 tokens). Every exact result re-checked in fp32. Differences reported with 95% paired-bootstrap ranges.

## Results

**1. Word groups are real structure inside the model.** Reordering a sentence by whole groups keeps the model's layer-12 representation closer to the original than reordering random spans of the same size (Karthika et al. groups +0.119 vs +0.082 random). A probe finds group boundaries (F1 0.84), though 0.75 already comes from the words alone. Our groupers match the hand-checked groups at F1 0.79–0.91 (random 0.28).

**2. But training heads on them does not make guesses go further.** Across 20 Hindi linguistic-vs-random comparisons, none is significantly better (every upper bound ≤ +0.03 tokens per pass) and 7 are slightly worse. The heads do learn the groups: accepted guesses end on a group boundary 2–4 points more often.

**Why:** accuracy moves from group starts to group insides. Only a third of tokens continue a group, so a small loss at group starts cancels a bigger gain inside groups. The real bottleneck is the next content word: every drafter gets group starts right only 17–20% of the time.

**3. Four changes take us from 1.4 to 3.5 tokens per pass** (Hindi, exact):

| change | tokens per pass |
|---|---|
| LoRA-tuned model, parallel heads | 1.39 |
| + frozen model, heads learn its own output (self-distillation) | 1.67 |
| + sequential heads | 2.00 |
| + tree of 64 guesses (2nd and 3rd choices, one pass) | 2.89 |
| + prompt lookup (copy what followed the same tokens earlier) | 3.49 |

**4. About 3× faster on one T4, with identical output.**

| decoding | Hindi | Marathi |
|---|---|---|
| heads, one guess per position | 1.82× | 1.67× |
| prompt lookup only | 1.49× | 2.28× |
| heads, tree of 64 guesses | 2.38× | 2.12× |
| **tree + lookup (exact)** | **2.9–3.0×** | **3.2–3.3×** |
| tree + lookup, lossy | 3.2–3.5× | 3.7–3.8× |

**5. Much of the speed comes from repetitive greedy text.** 35% (Hindi) and 62% (Marathi) of greedy 4-token sequences repeat earlier text (human text: 0.3% / 1.2%). Sampling at T = 0.7 leaves 1.4–1.8×; a repetition penalty removes a third (Hindi) to 60% (Marathi) of the gain. Low perplexity here means repetitive, not better, so we always report it next to the repetition rate.

**6. Human ratings** (blind, 50 Hindi / 30 Marathi prompts): sampling reads most fluent; exact greedy is ranked best in Hindi; lossy decoding is slightly worse than greedy.

**7. The tokenizer decides what a speed-up means.** Qwen2.5 splits a Hindi word into 4.7 tokens: 4.2× in tokens, but only 1.2 words per pass against 3.3 for ganga-1b. A word-completion list (no training) drafts the rest of split words for it.

## Repository

```
mtp/            the package: data and groupers, draft heads, losses, evaluation and speculative decoding
scripts/        training (train.py), evaluation (evaluate.py), analyses and figures
configs/        one YAML per run (pilot_*.yaml are 2k-step pilots)
results/        all measured numbers (JSON) and figures
data/gold/      hand-checked Hindi and Marathi word groups
notebooks/      Kaggle training notebook
docs/           design notes and the full run log (docs/runs.md)
tests/          CPU-only tests
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install torch                                       # the CUDA build for your machine
pip install -r requirements.txt && pip install -e .
python -m pytest -q                                     # CPU-only
python scripts/train.py --config configs/R2.yaml --set optim.max_steps=100
```

Full training runs: `notebooks/kaggle_train.ipynb` (Kaggle T4 ×2, a 12.5k-step run takes ~75 min).

## Run identifiers

| setup | linguistic | random twin |
|---|---|---|
| Baselines: NTP / MTP-1 / MTP-3 (Hindi) | R0 / R1 / R2 | – |
| Rules | R3 | R6a |
| Parser | R7s | R6a |
| Karthika et al. | R11 | R11a |
| Rules, 4× loss weight | R12 | R12a |
| Frozen parallel / sequential heads with groups | Rsd_struct / Rsd_seq_struct | …_rand |
| Marathi rules / whole words | R10 / R13 | R10a / R13a |
| Main speed models (frozen, sequential, no groups) | Rsd_seq (Hindi), Rsd_r8_seq (Marathi) | – |

Seed 43 runs carry the suffix `_s43`.

## References

Leviathan et al. 2023 (speculative decoding) · Gloeckle et al. 2024 (multi-token prediction) · Cai et al. 2024 (Medusa) · Ankner et al. 2024 (Hydra) · Li et al. 2024 (EAGLE-2) · Miao et al. 2024 (SpecInfer) · Saxena 2023 (prompt lookup) · Karthika et al. 2025 (semantically cohesive word grouping, arXiv:2501.03988) · Qi et al. 2020 (Stanza) · Doddapaneni et al. 2023 (IndicCorpV2) · NLLB Team 2022 (FLORES-200).
Models: LingoIITGN ganga-1b, smallstep.ai Misal-1B, Sarvam AI sarvam-1, Qwen2.5-1.5B.
