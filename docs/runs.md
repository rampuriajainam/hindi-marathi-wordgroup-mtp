# Run log

One row per run. Owner: Jainam (training); Om adds the evaluated numbers from `results/`.
Eval sets: `eval_small` = IndicCorp raw rows 0-99 (~50 sentences), `eval` = rows 0-999 (~500). "h{d}" = head d (predicts token t+d+1); loss = mean CE; % = top-1.

## Laptop (RTX 4060, bf16), before the package

| run | what | result |
|---|---|---|
| NTP baseline | LoRA r=8, 100k raw rows (~50k sentences), legacy script | eval_small h0 **3.05**, flat (lr/rank sweeps did not help) |
| MTP k=2 linear | legacy `MedusaWrapper`, 12,500 batches, heads summed | eval_small h0 **3.09**, h1 8.10 → **5.84** |

## `scripts/train.py` checks (laptop, bf16)

| run | steps | eval set | h0 | h1 | h2 | h3 | notes |
|---|---|---|---|---|---|---|---|
| R0 | 100 | eval_small | 3.054 | | | | matches laptop NTP (3.069 @100) |
| R1 | 100 | eval_small | 3.096 | 7.90 | | | matches laptop MTP (3.102 / 8.10 @100); same trainable-param count |
| R0 | 2000 | eval | 2.836 (42.5%) | | | | reference for the α pilot |
| R2 α=0.0 | 2000 | eval | 2.836 (42.5%) | 5.688 (12.5%) | 6.598 (6.4%) | 6.996 (4.1%) | heads detached: LM identical to R0 |
| R2 α=0.1 | 2000 | eval | 2.847 (42.3%) | 5.595 (12.7%) | 6.487 (6.7%) | 6.878 (4.4%) | **chosen** (`head_backbone_grad: 0.1`) |
| R2 α=1.0 | 2000 | eval | 3.034 (40.1%) | 5.422 (13.2%) | 6.263 (7.1%) | 6.634 (4.9%) | heads best, verifier 0.20 worse |

α = `head_backbone_grad`, the share of the extra heads' gradient that reaches LoRA. Zero-init resblock heads start as copies of head 0, so at init h1/h2/h3 = 9.37 / 10.59 / 10.95 (random linear heads: ~10.5).

## Kaggle (T4 x2, fp32 weights + fp16 autocast, ~0.34 s/step)

Step 12500, eval_small (50 sentences, `data.eval_n: 100` raw rows). Cell: loss (top-1). In-group / at-boundary top-1 from `train.py`'s reference eval with `hi_rules_v0`.

| ID | commit | config | Kaggle dataset | h0 | h1 | h2 | h3 | notes |
|---|---|---|---|---|---|---|---|---|
| R0 | fe87853 | R0.yaml | `jainamrampuria/mtp-run-R0` | 3.049 (39.8%) | | | | 2026-10-09; 0.29 s/step; h0 in-group / boundary 57.7 / 31.4% |
| R1 | fe87853 | R1.yaml | `jainamrampuria/mtp-run-R1` | 3.068 (39.3%) | 5.570 (13.9%) | | | 2026-10-09; 0.36 s/step; in-group / boundary h0 56.0 / 31.5%, h1 25.2 / 13.0% |
| R2 | fe87853 | R2.yaml (α=0.1) | `jainamrampuria/mtp-run-R2` | 3.052 (39.6%) | 5.533 (13.8%) | 6.402 (7.8%) | 6.801 (4.4%) | 2026-10-09; 0.38 s/step; in-group / boundary h0 56.5 / 31.7%, h1 24.5 / 13.0%, h2 19.3 / 7.5%, h3 13.3 / 4.4% |

Datasets hold `config.yaml`, `metrics.jsonl`, `train.log`, `step_12000/`, `step_12500/` (private; shared with Om).

**Gate B: passed.** R0 h0 3.049 vs laptop 3.05; R1 h0 3.068 vs 3.09 (both within 0.05). R1 h1 5.57 vs 5.84 is 0.27 *better*, which fits the known differences: the legacy script summed head losses, the laptop trained on raw rows 100+ with half of them blank, and bf16 vs fp16 autocast. Head 0 of the MTP runs stays within 0.02 of R0, so `head_backbone_grad: 0.1` holds the verifier at NTP quality over the full 12.5k steps.

On eval_small the h0 loss is flat from step 250 (3.052) to 12500 (3.049), so the gains are in heads 1-3. The full 500-sentence eval set and FLORES come from Om's `evaluate.py` (OM-4/OM-7).

### Method runs (Kaggle T4 x2, account `jainam2142`)

Pilots that led here: `docs/design_structural_loss.md` §5. Step 12500. Logged eval (`train.py`): eval_small for the R3 line, R8-R10a and A1; `eval` (~500 sentences) for the Rsd line. Logged in-group / boundary uses each run's **own** grouper, so for runs trained with random groups (R6a, R6a_s43, R10a, Rsd_mr) it is not comparable; the comparisons below fix the labels.

| ID | commit | config | Kaggle dataset | h0 | h1 | h2 | h3 | notes |
|---|---|---|---|---|---|---|---|---|
| R3 | 75b1db5 | R3.yaml (S23_mix, λ 0.25 / 0.25, chain) | `jainam2142/mtp-run-R3` | 3.056 (39.5%) | 5.497 (15.0%) | 6.382 (8.1%) | 6.773 (4.9%) | 2026-10-10; eval_small; h1 in-group / boundary 28.8 / 14.0% |
| Rsd | a4f1ff4 | Rsd.yaml (frozen backbone, self-distillation, 100k texts) | `jainam2142/mtp-run-Rsd` | 2.865 (42.2%) | 5.967 (13.9%) | 6.869 (7.1%) | 7.250 (4.5%) | 2026-10-10; `eval`; head 0 = base model exactly |
| Rsd_s43 | a4f1ff4 | Rsd_s43.yaml (= Rsd, seed 43 + `data.shuffle`) | `jainam2142/mtp-run-Rsd-s43` | 2.865 (42.2%) | 5.969 (14.0%) | 6.878 (7.2%) | 7.257 (4.3%) | replicate of Rsd |
| R6a | 1124204 | R6a.yaml (= R3, `grouper: random`) | `jainam2142/mtp-run-R6a` | 3.057 (39.5%) | 5.474 (14.8%) | 6.342 (8.0%) | 6.743 (5.4%) | 2026-10-10; eval_small; compare with R3 under `hi_rules_v0` (below) |
| R3_s43 | #44 | R3_s43.yaml (= R3, seed 43 + `data.shuffle`) | `jainam2142/mtp-run-R3-s43` | 3.057 (39.5%) | 5.494 (14.4%) | 6.387 (7.6%) | 6.786 (4.6%) | 2026-10-10; laptop RTX 4060 (bf16); in-group h1 / h2 / h3 25.2 / 24.6 / 20.0% |
| R6a_s43 | #44 | R6a_s43.yaml (= R6a, seed 43 + `data.shuffle`) | `jainam2142/mtp-run-R6a-s43` | 3.055 (39.5%) | 5.461 (14.4%) | 6.354 (7.2%) | 6.751 (4.9%) | 2026-10-10; account `jainamrampuria` |
| A1_alpha0 | #35 | A1_alpha0.yaml (= R2, α = 0) | `jainamrampuria/mtp-run-A1-alpha0` | 3.051 (39.7%) | 5.596 (13.5%) | 6.481 (7.8%) | 6.880 (4.3%) | official eval below |
| A1_alpha1 | #35 | A1_alpha1.yaml (= R2, α = 1) | `jainamrampuria/mtp-run-A1-alpha1` | 3.175 (38.1%) | 5.392 (14.5%) | 6.230 (8.4%) | 6.611 (4.9%) | official eval below |
| R8 | — | R8.yaml (Misal-1B, NTP) | `jainam2142/mtp-run-R8` | 4.127 (31.9%) | | | | eval_small (mr) |
| R9 | — | R9.yaml (Misal-1B, = R2) | `jainam2142/mtp-run-R9` | 4.132 (31.6%) | 6.576 (13.5%) | 7.373 (8.7%) | 7.716 (7.2%) | eval_small (mr) |
| R10 | #43 | R10.yaml (= R3 recipe, `mr_rules_v1`) | `jainam2142/mtp-run-R10` | 4.159 (31.2%) | 6.494 (13.7%) | 7.296 (9.3%) | 7.635 (7.1%) | eval_small (mr) |
| R10a | #43 | R10a.yaml (= R10, `random_mr_v1`) | `jainam2142/mtp-run-R10a` | 4.157 (31.3%) | 6.490 (13.8%) | 7.282 (9.0%) | 7.628 (7.2%) | eval_small (mr) |
| Rsd_soft | #42 | Rsd_soft.yaml (= Rsd + soft labels, S3_all λ 1, teacher h0) | `jainam2142/mtp-run-Rsd-soft` | 2.865 (42.2%) | 5.693 (14.3%) | 6.638 (7.4%) | 7.025 (4.9%) | `eval` |
| Rsd_k6 | #42 | Rsd_k6.yaml (= Rsd, 6 heads) | `jainam2142/mtp-run-Rsd-k6` | 2.865 (42.2%) | 5.967 (13.8%) | 6.869 (7.1%) | 7.249 (4.5%) | `eval`; h4 7.447 (3.5%), h5 7.573 (2.9%) |
| Rsd_mr | #42 | Rsd_mr.yaml (frozen Misal, self-distillation, `min_new_tokens 64`) | `jainam2142/mtp-run-Rsd-mr` | 4.890 (27.8%) | 7.833 (9.3%) | 8.552 (5.7%) | 8.840 (4.3%) | `eval` (mr); head 0 = Misal-instruct as is |
| Rsd_mr_soft | #42 | Rsd_mr_soft.yaml (= Rsd_mr + soft labels) | `jainam2142/mtp-run-Rsd-mr-soft` | 4.890 (27.8%) | 7.666 (9.3%) | 8.394 (5.6%) | 8.683 (4.3%) | `eval` (mr) |
| Rsd_long, Rsd_data2x | #42 | 2× data and 2× steps / 2× data, same steps | — | | | | | running 2026-10-10 |

`mtp-run-R3-s43`, `-R6a-s43`, `-Rsd-mr`, `-Rsd-mr-soft` are public; the others are private and shared with Om.

**Acceptance (Jainam's check, not the official eval):** FixedK on 100 IndicCorp-eval prompts (`make_prompts(..., seed=1)`), RTX 4060, paired bootstrap vs R2.

| run | tokens/step | h1 / h2 / h3 acceptance | h1 in-group top-1 (`eval`, real text) | speed-up (fp32, 20 flores_hi prompts) |
|---|---|---|---|---|
| R2 | 1.370 | 30.1 / 6.1 / 1.4% | 23.4% | 1.24× |
| R3 | 1.380 (n.s.) | 30.9 / 6.3 / 1.5% | **+2.3 ✓** | — |
| Rsd | **1.649 (+0.28 ✓)** | **41.0 / 17.1 / 8.2%** | −3.4 | **1.53×** |
| Rsd_s43 | 1.664 (+0.29 ✓) | 41.6 / 18.0 / 8.3% | −3.3 | — |

## Official evaluation (OM-7 `scripts/evaluate.py`, OM-8)

Step 12500, grouper `hi_rules_v0`, Kaggle T4. JSONs in `results/{run}/eval_{dataset}.json` (INTERFACES §10).
Cell: loss (top-1). In-group / boundary = top-1 when the target is / is not in the source token's word group.

**IndicCorp eval (raw rows 0-999, ~500 sentences)**

| run | h0 | h1 | h2 | h3 | in-group / boundary top-1 (h0 · h1 · h2 · h3) |
|---|---|---|---|---|---|
| R0 | 2.835 (42.5%) |  |  |  | 58.4% / 34.7% |
| R1 | 2.866 (42.0%) | 5.468 (15.4%) |  |  | 58.2% / 34.2% · 28.7% / 14.3% |
| R2 | 2.841 (42.4%) | 5.367 (15.0%) | 6.290 (8.1%) | 6.707 (5.3%) | 58.3% / 34.6% · 23.4% / 14.3% · 19.0% / 7.8% · 17.5% / 5.2% |

Targets per head (R2): h0 26,563 (in-group 8,729), h1 26,063 (in-group 1,886), h2 25,563 (in-group 557), h3 25,063 (in-group 171).

**FLORES-200 devtest hin_Deva (1,012 sentences)**

| run | h0 | h1 | h2 | h3 | in-group / boundary top-1 (h0 · h1 · h2 · h3) |
|---|---|---|---|---|---|
| R0 | 3.956 (30.2%) |  |  |  | 44.1% / 21.7% |
| R1 | 3.957 (30.1%) | 6.377 (10.4%) |  |  | 43.9% / 21.7% · 15.0% / 9.8% |
| R2 | 3.946 (30.3%) | 6.358 (10.3%) | 7.070 (5.9%) | 7.341 (4.4%) | 44.1% / 21.9% · 12.5% / 10.0% · 12.1% / 5.7% · 12.1% / 4.3% |

Targets per head (R2): h0 29,135 (in-group 11,025), h1 28,123 (in-group 3,116), h2 27,111 (in-group 1,009), h3 26,099 (in-group 298).

Checks and observations:

- R0 h0 on `eval` = 2.835, matching the laptop R0 at 2k steps (2.836). Head 0 of R2 is within +0.006 of R0 (R1, α = 1: +0.031).
- In-group targets are easier than boundary targets for every head. Ratio of top-1 (R2) on `eval`: 1.7× / 1.6× / 2.4× / 3.4× for h0-h3; on FLORES 2.0× / 1.25× / 2.1× / 2.8×. FLORES h1 has the smallest gap (12.5% vs 10.0%).
- R1's linear head 1 beats R2's resblock head 1 on in-group top-1 (28.7% vs 23.4% on `eval`) at similar overall top-1 (15.4% vs 15.0%). Worth a look before R3 is compared against R2.
- In-group counts for h3 are small (171 on `eval`), so h3 in-group numbers are noisy.
- FLORES is harder for every head (h0 ~3.95 vs ~2.84): a different domain from the IndicCorp training text.

## Self-speculative decoding (OM-6, OM-8)

Step 12500, Kaggle T4 (fp32 weights + fp16 autocast), batch size 1. Prompts: first 8-16 words of 200 sentences per dataset, 64 new tokens.
Accepted length = tokens per step (greedy = 1.0). Speed-up = tokens/s vs greedy decoding with the base LM on the same prompts.

| run | dataset | policy | accepted len | accept rate h1 / h2 / h3 | speed-up | identical to greedy (fp16) | fp32 re-check (20) | Group Integrity |
|---|---|---|---|---|---|---|---|---|
| R1 | indiccorp_eval | fixed_k | 1.31 | 31.0% | ×1.25 | 87.5% of prompts | 20/20 | 68.0% |
| R1 | indiccorp_eval | confidence_cut | 1.08 | 80.8% | ×1.06 | 88.5% of prompts | 20/20 | 55.0% |
| R1 | flores_hi | fixed_k | 1.28 | 28.7% | ×1.23 | 81.5% of prompts | 20/20 | 77.3% |
| R1 | flores_hi | confidence_cut | 1.05 | 79.9% | ×1.04 | 81.0% of prompts | 20/20 | 70.7% |
| R2 | indiccorp_eval | fixed_k | 1.39 | 30.1% / 7.2% / 2.4% | ×1.14 | 87.5% of prompts | 19/20 | 75.7% |
| R2 | indiccorp_eval | confidence_cut | 1.10 | 77.0% / 73.2% / 84.8% | ×0.94 | 87.5% of prompts | 20/20 | 67.1% |
| R2 | flores_hi | fixed_k | 1.36 | 29.5% / 6.2% / 1.3% | ×1.12 | 89.0% of prompts | 20/20 | 79.0% |
| R2 | flores_hi | confidence_cut | 1.07 | 75.9% / 63.6% / 65.0% | ×0.91 | 89.0% of prompts | 20/20 | 74.2% |

- **Speed-up:** FixedK gives ×1.12-1.25. ConfidenceCut (τ = 0.5) proposes few drafts (~1.05-1.10 tokens/step) and is slower than greedy for R2. The far heads of R2 are rarely accepted (h2 6-7%, h3 1-2%), so R2's extra heads add cost without adding accepted tokens; R1's single linear head gives the best speed-up.
- **Correctness:** in fp16, 81-89% of prompts are identical to greedy. Every divergence happens at a near-tie: greedy's top-2 logit margin there is at most 0.0156 = 2⁻⁶, one fp16 step at these logit sizes (some exactly 0, true ties). The fp32 re-check of the same engine (autocast off) is 20/20 identical in 11 of 12 cases.
- **Resolved:** R2 / indiccorp_eval / fixed_k is 19/20 in the fp32 re-check. The rerun (2026-10-10, same numbers: ×1.147 vs ×1.143) records the fp32 divergence margin: **0.0**, an exact tie between greedy's top-2 logits, so argmax tie-breaking differs and the engine is correct.

## R3 vs R2 (OM-8, official eval)

R3 = `R3_hi_k4_struct` (S23_mix, λ_S3 0.25, λ_S3_all 0.25, chain teacher), `jainam2142/mtp-run-R3`, step 12500. Same eval as R0-R2: Kaggle T4, grouper `hi_rules_v0`, 200 spec-decode prompts.

| dataset | head | loss R2 → R3 | top-1 R2 → R3 | in-group top-1 R2 → R3 (n targets) | boundary top-1 R2 → R3 |
|---|---|---|---|---|---|
| indiccorp_eval | h0 | 2.841 → 2.849 (+0.008) | 42.4% → 42.3% | 58.3% → 58.3% (8,729) | 34.6% → 34.6% |
| indiccorp_eval | h1 | 5.367 → 5.320 (-0.047) | 15.0% → 15.5% | 23.4% → 25.8% (1,886) | 14.3% → 14.7% |
| indiccorp_eval | h2 | 6.290 → 6.256 (-0.033) | 8.1% → 8.4% | 19.0% → 24.1% (557) | 7.8% → 8.0% |
| indiccorp_eval | h3 | 6.707 → 6.678 (-0.030) | 5.3% → 5.5% | 17.5% → 22.2% (171) | 5.2% → 5.3% |
| flores_hi | h0 | 3.946 → 3.944 (-0.002) | 30.3% → 30.2% | 44.1% → 44.1% (11,025) | 21.9% → 21.7% |
| flores_hi | h1 | 6.358 → 6.299 (-0.059) | 10.3% → 10.6% | 12.5% → 14.3% (3,116) | 10.0% → 10.1% |
| flores_hi | h2 | 7.070 → 7.016 (-0.053) | 5.9% → 6.2% | 12.1% → 14.0% (1,009) | 5.7% → 5.9% |
| flores_hi | h3 | 7.341 → 7.296 (-0.045) | 4.4% → 4.5% | 12.1% → 13.1% (298) | 4.3% → 4.4% |

| dataset | policy | accepted len R2 → R3 | speed-up R2 → R3 | R3 identical to greedy (fp16) | R3 fp32 re-check |
|---|---|---|---|---|---|
| indiccorp_eval | fixed_k | 1.39 → 1.39 | ×1.14 → ×1.15 | 84.5% of prompts | 20/20 |
| indiccorp_eval | confidence_cut | 1.10 → 1.11 | ×0.94 → ×0.94 | 84.5% of prompts | 20/20 |
| flores_hi | fixed_k | 1.36 → 1.35 | ×1.12 → ×1.12 | 87.5% of prompts | 20/20 |
| flores_hi | confidence_cut | 1.07 → 1.07 | ×0.91 → ×0.91 | 88.0% of prompts | 20/20 |

- **In-group accuracy of the far heads goes up, on both datasets.** IndicCorp h1 / h2 / h3: +2.4 / +5.1 / +4.7 points; FLORES (out of domain): +1.8 / +1.9 / +1.0. Boundary top-1 barely moves (≤ +0.4), so the gain is where the loss targets it. Loss improves for every extra head (−0.03 to −0.06).
- **Head 0 is held:** +0.008 on IndicCorp eval, −0.002 on FLORES.
- **Acceptance and speed are unchanged** (1.39 vs 1.39 tokens/step, ×1.15 vs ×1.14 FixedK on IndicCorp), matching Jainam's 100-prompt check (design note §5). The structural loss improves *what* the heads predict inside groups, not how often head 0 agrees with them.
- **Correctness:** fp32 re-check 20/20 for every policy and dataset; fp16 divergences again only at near-ties (margin ≤ 0.0156).
- h3 in-group counts are small (171 IndicCorp, 298 FLORES): read h3's in-group numbers as noisy.

## Rsd: frozen backbone + self-distillation (OM-8, official eval)

`Rsd_hi_k4_frozen_sd` (`jainam2142/mtp-run-rsd`) and its replicate `Rsd_hi_k4_frozen_sd_s43` (`jainam2142/mtp-run-rsd-s43`, seed 43 + shuffled data), step 12500. Same eval as R0-R3: Kaggle T4, grouper `hi_rules_v0`, 200 spec-decode prompts. Head 0 is ganga-1b with no adapter, so it trails R0 (LoRA NTP) by 0.03 loss on IndicCorp eval and is ahead by 0.03 on FLORES.

| run | dataset | h0 | h1 | h2 | h3 | in-group / boundary top-1 (h1 · h2 · h3) |
|---|---|---|---|---|---|---|
| R2 | indiccorp_eval | 2.841 (42.4%) | 5.367 (15.0%) | 6.290 (8.1%) | 6.707 (5.3%) | 23.4 / 14.3 · 19.0 / 7.8 · 17.5 / 5.2 |
| Rsd | indiccorp_eval | 2.865 (42.2%) | 5.967 (13.9%) | 6.869 (7.1%) | 7.250 (4.5%) | 19.9 / 13.4 · 13.1 / 7.0 · 13.5 / 4.5 |
| Rsd_s43 | indiccorp_eval | 2.865 (42.2%) | 5.969 (14.0%) | 6.878 (7.2%) | 7.257 (4.3%) | 20.0 / 13.5 · 12.6 / 7.1 · 12.9 / 4.3 |
| R2 | flores_hi | 3.946 (30.3%) | 6.358 (10.3%) | 7.070 (5.9%) | 7.341 (4.4%) | 12.5 / 10.0 · 12.1 / 5.7 · 12.1 / 4.3 |
| Rsd | flores_hi | 3.984 (29.9%) | 6.909 (8.8%) | 7.577 (4.6%) | 7.822 (3.3%) | 9.5 / 8.7 · 6.9 / 4.5 · 8.7 / 3.3 |
| Rsd_s43 | flores_hi | 3.984 (29.9%) | 6.908 (8.5%) | 7.580 (4.4%) | 7.822 (2.9%) | 9.1 / 8.5 · 5.9 / 4.4 · 5.0 / 2.9 |

Per-head numbers are against the *real* next tokens; Rsd's heads were trained on the base model's greedy text, so they are worse here and better where it matters, on head 0's own continuations:

| run | dataset | policy | accepted len | accept rate h1 / h2 / h3 | speed-up (tok/s) | identical to greedy (fp16) | fp32 re-check (20) | Group Integrity |
|---|---|---|---|---|---|---|---|---|
| R2 | indiccorp_eval | fixed_k | 1.39 | 30.1 / 7.2 / 2.4% | ×1.15 (31.8) | 87.5% | 19/20 (exact tie) | 75.7% |
| Rsd | indiccorp_eval | fixed_k | **1.67** | 41.7 / 17.4 / 9.1% | **×1.36** (37.5) | 87.5% | 20/20 | 74.2% |
| Rsd_s43 | indiccorp_eval | fixed_k | 1.66 | 40.7 / 17.6 / 8.9% | ×1.35 (37.4) | 87.5% | 20/20 | 74.4% |
| Rsd | indiccorp_eval | confidence_cut | 1.36 | 70.5 / 64.3 / 67.8% | ×1.14 | 87.0% | 20/20 | 72.2% |
| Rsd_s43 | indiccorp_eval | confidence_cut | 1.37 | 70.5 / 65.9 / 67.8% | ×1.14 | 88.0% | 20/20 | 71.5% |
| R2 | flores_hi | fixed_k | 1.36 | 29.5 / 6.2 / 1.3% | ×1.12 (31.1) | 89.0% | 20/20 | 79.0% |
| Rsd | flores_hi | fixed_k | **1.60** | 39.0 / 15.5 / 7.1% | **×1.32** (36.4) | 82.5% | 20/20 | 80.2% |
| Rsd_s43 | flores_hi | fixed_k | 1.62 | 39.6 / 15.7 / 7.6% | ×1.33 (36.8) | 82.5% | 20/20 | 78.5% |
| Rsd | flores_hi | confidence_cut | 1.30 | 72.6 / 64.9 / 65.8% | ×1.09 | 83.0% | 20/20 | 76.4% |
| Rsd_s43 | flores_hi | confidence_cut | 1.31 | 71.3 / 67.6 / 72.6% | ×1.10 | 82.5% | 20/20 | 76.6% |

Greedy baseline: 27.6-27.8 tok/s on every run.

- **Rsd is the fastest run: ×1.32-1.36 on T4** (vs ×1.12-1.15 for R2/R3), 1.60-1.67 tokens/step. It matches Jainam's 100-prompt check (1.649 tokens/step); his 1.53× was fp32 on an RTX 4060, where the verify pass is relatively cheaper.
- **Replicate agrees:** Rsd_s43 is within 0.01-0.02 tokens/step and ±0.01 speed-up of Rsd on both sets.
- **Far heads are where it gains:** h2 / h3 acceptance 17 / 9% vs R2's 7 / 2%.
- **Correctness:** fp32 re-check 20/20 for every Rsd policy and dataset; fp16 divergences only at near-ties (margin ≤ 0.0156).
- ConfidenceCut is now faster than greedy (×1.09-1.14) but still behind FixedK, consistent with dropping stop-early policies (#33).

## Verify-pass cost on T4 (`bench_verify`, issue #33)

One forward pass of the MTP model (all heads) scoring N positions after a 64-token prompt, KV cache rolled back after each pass. Median of 20 repeats; fp32 weights + fp16 autocast. Greedy = base LM scoring 1 position.

| positions | 1 | 4 | 8 | 16 | 25 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| R2 (ms) | 41.9 | 42.9 | 43.1 | 44.1 | 45.6 | 45.8 | 47.1 |
| Rsd (ms) | 41.6 | 42.6 | 43.0 | 43.8 | 45.3 | 45.5 | 46.5 |
| Rsd_s43 (ms) | 41.4 | 42.5 | 42.8 | 43.7 | 45.1 | 45.2 | 46.2 |
| × greedy pass (Rsd) | 1.17 | 1.20 | 1.21 | 1.23 | 1.27 | 1.27 | 1.30 |

Greedy pass: 35.4-35.7 ms.

- **At batch 1 the T4 is memory-bound:** scoring 25 positions costs 9% more than scoring 1, and 64 positions costs 12% more. Tree size is not the constraint up to 64 nodes. The fixed cost is the extra heads: 1 position is already ×1.17 greedy.
- **The model predicts the measured chain speed-up:** Rsd chain = 1.67 tokens/step ÷ 1.20 (4 positions) = ×1.39, against ×1.36 measured (the rest is drafting overhead). On the same model, Jainam's offline 25-node tree (2.51 tokens/step, #33) ÷ 1.27 gives a **×1.98 ceiling** for `generate_tree()` on T4.

## R6a: random-group control for R3 (OM-8, official eval)

`R6a_hi_k4_struct_random` (`jainam2142/mtp-run-R6a`) = R3 with `data.grouper: random` (JI-4), step 12500. Evaluated with `--grouper hi_rules_v0`, so in-group / boundary use the **linguistic** groups for every run. The target counts match R2/R3 exactly (h1 in-group 1,886 on IndicCorp, 3,116 on FLORES).

| dataset | head | top-1 R2 / R3 / R6a | in-group top-1 R2 / R3 / R6a (n) | boundary top-1 R2 / R3 / R6a |
|---|---|---|---|---|
| indiccorp_eval | h0 | 42.4 / 42.3 / 42.3 | 58.3 / 58.3 / 58.3 (8,729) | 34.6 / 34.6 / 34.5 |
| indiccorp_eval | h1 | 15.0 / 15.5 / 15.4 | 23.4 / **25.8** / **25.2** (1,886) | 14.3 / 14.7 / 14.7 |
| indiccorp_eval | h2 | 8.1 / 8.4 / 8.4 | 19.0 / **24.1** / 21.5 (557) | 7.8 / 8.0 / 8.1 |
| indiccorp_eval | h3 | 5.3 / 5.5 / 5.4 | 17.5 / **22.2** / 19.9 (171) | 5.2 / 5.3 / 5.3 |
| flores_hi | h0 | 30.3 / 30.2 / 30.2 | 44.1 / 44.1 / 44.2 (11,025) | 21.9 / 21.7 / 21.8 |
| flores_hi | h1 | 10.3 / 10.6 / 10.6 | 12.5 / **14.3** / 13.7 (3,116) | 10.0 / 10.1 / 10.2 |
| flores_hi | h2 | 5.9 / 6.2 / 6.2 | 12.1 / **14.0** / 12.9 (1,009) | 5.7 / 5.9 / 6.0 |
| flores_hi | h3 | 4.4 / 4.5 / 4.5 | 12.1 / **13.1** / 11.1 (298) | 4.3 / 4.4 / 4.5 |

| dataset | FixedK tokens/step R2 / R3 / R6a | speed-up R2 / R3 / R6a | R6a fp32 re-check |
|---|---|---|---|
| indiccorp_eval | 1.39 / 1.39 / 1.40 | ×1.15 / ×1.15 / ×1.14 | 20/20 (both policies) |
| flores_hi | 1.36 / 1.35 / 1.37 | ×1.12 / ×1.12 / ×1.12 | 20/20 (both policies) |

- **Head 1: random groups give most of R3's in-group gain.** R6a is at +1.8 vs R2 on IndicCorp (R3 +2.4) and +1.2 on FLORES (R3 +1.8). It does **not** reproduce the −11.5 point drop of the laptop check in #39.
- **Heads 2-3: R3 stays ahead of R6a** by 2.3 to 2.6 points on IndicCorp and 1.1 to 2.0 on FLORES, about half of R3's gain over R2. These counts are small (557 / 171 targets on IndicCorp), so the binomial SE is about 1.7 / 3.1 points. A paired bootstrap needs token dumps (`--dump_tokens`).
- Head 0, acceptance and speed: same as R2/R3.

## α ablation A1 (JN-10, OM-8, official eval)

`A1_alpha0` / `A1_alpha1` (`jainamrampuria/mtp-run-A1-alpha{0,1}`) = R2 with `head_backbone_grad` 0 / 1; R2 is α = 0.1. Step 12500, `hi_rules_v0`, 200 prompts.

| α | dataset | h0 loss (top-1) | h1 / h2 / h3 top-1 | FixedK tokens/step | h1 / h2 / h3 accept | speed-up | fp32 re-check |
|---|---|---|---|---|---|---|---|
| 0 | indiccorp_eval | 2.834 (42.5%) | 14.7 / 7.9 / 5.2 | 1.38 | 29.7 / 6.8 / 2.3% | ×1.14 | 20/20 |
| 0.1 (R2) | indiccorp_eval | 2.841 (42.4%) | 15.0 / 8.1 / 5.3 | 1.39 | 30.1 / 7.2 / 2.4% | ×1.15 | 19/20 (exact tie) |
| 1 | indiccorp_eval | **2.981 (40.6%)** | 15.6 / 8.7 / 5.7 | **1.44** | 34.1 / 8.3 / 2.0% | **×1.18** | 20/20 |
| 0 | flores_hi | 3.955 (30.2%) | 10.0 / 5.8 / 4.3 | 1.32 | 27.2 / 5.0 / 0.8% | ×1.09 | 20/20 |
| 0.1 (R2) | flores_hi | 3.946 (30.3%) | 10.3 / 5.9 / 4.4 | 1.36 | 29.5 / 6.2 / 1.3% | ×1.12 | 20/20 |
| 1 | flores_hi | **4.019 (29.5%)** | 10.7 / 6.3 / 4.7 | **1.40** | 32.5 / 7.0 / 1.1% | **×1.16** | 20/20 |

- **The trade-off holds on T4:** α = 1 gives +0.04-0.05 tokens/step and +0.03-0.04 speed-up over R2, but head 0 (the model being sped up) is worse by +0.14 nats / −1.8 points top-1 on IndicCorp and +0.07 / −0.8 on FLORES.
- **α = 0.1 is the right default:** its verifier is within 0.01 of α = 0, and its heads are slightly better.
- This matches Jainam's laptop check: +0.049 tokens/step for α = 1.

## Marathi: R8 / R9 (Misal-1B, OM-8, official eval)

`R8_mr_ntp` / `R9_mr_mtp_k4_resblock` (`jainam2142/mtp-run-R8`, `-R9`), base `smallstepai/Misal-1B-instruct-v0.1` (adds BOS; 1.61 tokens/word). Step 12500, grouper `mr_rules_v1`, IndicCorp mr eval (raw rows 0-999) + FLORES devtest mar_Deva, 200 prompts.

| run | dataset | h0 | h1 | h2 | h3 | in-group / boundary top-1 (h0 · h1 · h2 · h3) |
|---|---|---|---|---|---|---|
| R8 | indiccorp_eval (mr) | 4.252 (30.2%) | | | | 46.2 / 20.1 |
| R9 | indiccorp_eval (mr) | 4.258 (30.1%) | 6.824 (11.5%) | 7.621 (7.6%) | 7.969 (6.0%) | 46.4 / 19.7 · 28.3 / 8.6 · 17.4 / 7.2 · 11.7 / 6.0 |
| R8 | flores_mr | 4.903 (23.7%) | | | | 39.8 / 12.6 |
| R9 | flores_mr | 4.905 (23.6%) | 7.520 (7.1%) | 8.122 (5.0%) | 8.318 (4.5%) | 39.5 / 12.6 · 16.2 / 5.4 · 8.2 / 4.9 · 8.1 / 4.5 |

In-group targets per head (R9, IndicCorp): h0 10,214, h1 3,931, h2 1,256, h3 444. That's about twice Hindi's share at h1 (3,931 vs 1,886 for a similar number of targets).

| run | dataset | policy | tokens/step | accept h1 / h2 / h3 | speed-up (tok/s) | identical (fp16) | fp32 re-check | Group Integrity |
|---|---|---|---|---|---|---|---|---|
| R9 | indiccorp_eval | fixed_k | 1.31 | 24.4 / 6.0 / 1.2% | ×1.07 (24.4 vs 22.9) | 85.0% | 20/20 | 77.3% |
| R9 | indiccorp_eval | confidence_cut | 1.07 | 58.4 / 36.5 / 14.1% | ×0.89 | 85.5% | 20/20 | 84.9% |
| R9 | flores_mr | fixed_k | 1.30 | 23.8 / 5.5 / 1.0% | ×1.05 (24.1 vs 23.0) | 90.0% | 20/20 | 79.4% |
| R9 | flores_mr | confidence_cut | 1.06 | 51.3 / 22.5 / 8.2% | ×0.88 | 88.5% | 20/20 | 78.1% |

- **Head 0 is held:** R9 − R8 = +0.006 (IndicCorp) / +0.002 (FLORES).
- **Marathi speed-up is smaller than Hindi:** ×1.05-1.07 vs ×1.12-1.15 for R2, because head 1 is accepted less often (24% vs 30%). Greedy is also slower (23 vs 27.7 tok/s; Misal is a Llama with a 47k vocabulary).
- **In-group vs boundary gap is much larger than in Hindi:** h1 28.3% in-group vs 8.6% at boundaries (3.3×; Hindi R2 1.6×). That's the word-internal structure Jainam's tree oracle picks up (#33).
- **EOS is not an issue for these prompts:** greedy took ~2.8 s per prompt at 23 tok/s ≈ 64 tokens, i.e. almost no prompt stopped early. Prompts are cut mid-sentence, unlike the train prompts in #39.

## JN-7b: loss-weighting pilots (Table 5, #36)

4 pilots × 2k steps on the R3 recipe (`configs/pilot_w_*.yaml`), each differing from `R3.yaml` only in `max_steps: 2000` and the `weighting` line. Paired bootstrap vs `pilot_w_fixed` (2000 resamples; 500 IndicCorp eval sentences for top-1, 100 prompts for tokens/step; labels `hi_rules_v0`; 95% CI). Points.

| pilot | weighting | h0 eval loss | h1 in-group top-1 | mean top-1 h1-3 | FixedK tokens/step |
|---|---|---|---|---|---|
| `pilot_w_fixed` | fixed 0.8^d (= R3) | 3.057 | ref | ref | ref |
| `pilot_w_unc` | uncertainty, head 0 fixed | 3.054 | +0.6 [−0.2, +1.4] | −0.12 [−0.26, +0.01] | −0.012 [−0.035, +0.014] |
| `pilot_w_unc_free` | uncertainty, head 0 free | **3.198 (+0.14)** | **+1.8 [+0.8, +2.8]** | +0.24 [+0.09, +0.40] | +0.014 [−0.023, +0.047] |
| `pilot_w_dwa` | DWA | 3.057 | +0.05 [−0.3, +0.4] | −0.03 [−0.07, +0.01] | −0.011 [−0.030, +0.002] |

- **Negative row:** with head 0 protected (`unc`, DWA), adaptive weighting is flat on every metric. `unc_free` gains in-group top-1 only by down-weighting head 0, so the verifier loses 0.14 nats: the same trade-off as α = 1 (A1). So R5 / R6 are not run (#36).
- **On a frozen backbone (Rsd), weighting is a no-op:** each head has its own parameters, and AdamW divides out a per-parameter loss scale. Rsd config, 200 steps, only `head_decay` changed: h1 eval loss 6.4244 (0.8^d) / 6.4248 (equal) / 6.4263 (0.3^d).

## R3 vs R6a: paired bootstrap (Jainam's check, #39)

Same token-level evaluation as above (500 IndicCorp eval sentences, labels `hi_rules_v0` for **every** run, so the in-group targets are the same: 1,886 / 557 / 171 for h1 / h2 / h3; matches OM-8). Difference vs R2 in points, 95% CI.

| vs R2 | h1 in-group | h2 in-group | h3 in-group | accepted len (100 prompts) |
|---|---|---|---|---|
| R3 | +2.4 [+1.5, +3.4] | +5.0 [+2.6, +7.7] | +4.7 [+0.5, +9.5] | −0.003 [−0.024, +0.020] |
| R6a (random groups) | +1.8 [+1.0, +2.6] | +2.5 [+0.4, +4.9] | +2.3 [−0.7, +5.6] | −0.002 [−0.024, +0.021] |

- At h1 the extra loss itself gives most of the gain (1.8 of 2.4 points). At h2 / h3 about half of R3's gain needs the linguistic groups.
- The direct R3 − R6a CI and the seed-43 pair (R3_s43 / R6a_s43) follow in #39.

## Marathi: R10 / R10a (structural loss, Jainam's check)

R10 = R3 recipe on Misal-1B with `mr_rules_v1`; R10a = the same with `random_mr_v1` (length-matched random groups). Labels `mr_rules_v1` for all three runs (in-group targets h1 / h2 / h3: 3,931 / 1,256 / 444). Difference vs R9 in points, 95% CI.

| vs R9 | h1 in-group | h2 in-group | h3 in-group | accepted len (100 prompts) |
|---|---|---|---|---|
| R10 (`mr_rules_v1`) | +2.2 [+1.5, +3.0] | +1.2 [+0.1, +2.3] | +1.6 [−0.5, +3.7] | +0.028 [−0.009, +0.068] |
| R10a (`random_mr_v1`) | +2.6 [+1.9, +3.4] | +1.4 [+0.3, +2.6] | +0.9 [−1.4, +3.3] | +0.027 [−0.007, +0.066] |

- **In Marathi the structural loss gains the same with random groups** (R10 ≈ R10a at every head). The linguistic groups still matter at decoding time: on R9's draft trees, groups beat random and word starts (#33).

## Rsd line: variants (Jainam's check)

FixedK chain on 100 IndicCorp eval prompts (RTX 4060 or T4), paired bootstrap vs Rsd. Offline tree oracle (`scripts/fit_tree.py`, accepted drafts per step on 200 eval prompts, teacher-forced; +1 = tokens/step): the per-position oracle overstates real decoding (real / offline ≈ 0.86 for k = 4).

| run | chain tokens/step vs Rsd | tree n25 | tree n40 | tree n64 (entropy-shaped) |
|---|---|---|---|---|
| Rsd | ref (offline chain 0.908) | 1.554 | 1.656 | 1.752 |
| Rsd_soft (soft labels) | −0.001 | 1.543 | 1.641 | 1.741 |
| Rsd_k6 (6 heads) | +0.037 [+0.023, +0.055] | 1.695 | 1.818 | 1.933 |

- Soft labels: same as hard labels at full length (the 2k-step pilot's +0.019 did not hold). Negative row.
- k = 6: small chain gain; the offline tree gain (+0.14-0.18) is to be checked on T4 with `generate_tree()` (OM).

**Marathi Rsd (frozen Misal-1B-instruct).** Misal-instruct stops after one sentence (median 6 generated tokens), so the self-distilled text is generated with `--min_new_tokens 64`. At decode time head 0 also stops: 66% of the 100 eval prompts end within 10 tokens. With head 0's `</s>` logit masked (still lossless with respect to the masked model):

| run | tokens/step, stop at `</s>` (mean generated) | tokens/step, `</s>` masked | vs R9 (masked) |
|---|---|---|---|
| R9 | 1.306 (64.0) | 1.306 | ref |
| Rsd_mr | 1.356 (14.0) | **1.539** | **+0.233 [+0.174, +0.299]** |
| Rsd_mr_soft | 1.415 (14.0) | 1.526 | +0.220 [+0.160, +0.287] |

- Self-distillation transfers to Marathi (Hindi: Rsd − R2 = +0.28 on the same check). Soft = hard again.
- **Caveat:** Rsd_mr speeds up Misal-instruct as is, whose head 0 (4.89 on `eval`) is much weaker on IndicCorp than the LoRA-tuned R8/R9 (4.25 official). Evaluate it with `ignore_eos` (OM).

## Rsd_soft and Rsd_k6 (OM-8, official eval)

`Rsd_hi_k4_frozen_sd_soft` (soft-label KL) and `Rsd_hi_k6_frozen_sd` (6 heads), step 12500, Kaggle T4, `hi_rules_v0`, 200 prompts. Head 0 = ganga-1b in all three, so its loss is identical (2.865 / 3.984).

| run | dataset | FixedK tokens/step | accept h1 … h5 | speed-up (tok/s) | ConfidenceCut | fp32 re-check |
|---|---|---|---|---|---|---|
| Rsd | indiccorp_eval | 1.67 | 41.7 / 17.4 / 9.1% | **×1.36** (37.5) | 1.36, ×1.14 | 20/20 |
| Rsd_soft | indiccorp_eval | 1.67 | 41.9 / 17.8 / 8.7% | **×1.37** (37.8) | 1.26, ×1.07 | 20/20 |
| Rsd_k6 | indiccorp_eval | **1.71** | 40.8 / 16.2 / 8.1 / 4.6 / 3.1% | ×1.28 (35.2) | 1.38, ×1.05 | 20/20 |
| Rsd | flores_hi | 1.60 | 39.0 / 15.5 / 7.1% | ×1.32 (36.4) | 1.30, ×1.09 | 20/20 |
| Rsd_soft | flores_hi | 1.60 | 39.2 / 15.4 / 6.6% | ×1.31 (36.0) | 1.22, ×1.03 | 20/20 |
| Rsd_k6 | flores_hi | **1.63** | 38.4 / 15.0 / 6.6 / 2.8 / 1.6% | ×1.21 (33.5) | 1.31, ×1.00 | 20/20 |

Verify-pass cost (`bench_verify`, ms, × greedy pass of 35.3-35.4 ms):

| positions | 1 | 4 | 8 | 16 | 25 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Rsd (k4) | 41.6 (1.17) | 42.6 (1.20) | 43.0 (1.21) | 43.8 (1.23) | 45.3 (1.27) | 45.5 (1.27) | 46.5 (1.30) |
| Rsd_soft (k4) | 41.5 (1.18) | 42.6 (1.21) | 42.9 (1.22) | 43.8 (1.24) | 45.3 (1.28) | 45.5 (1.29) | 46.5 (1.32) |
| Rsd_k6 | 45.5 (1.29) | 46.7 (1.32) | 47.0 (1.33) | 48.4 (1.37) | 49.6 (1.40) | 49.7 (1.40) | 50.9 (1.44) |

- **Soft labels: no gain over Rsd** in acceptance or speed. That confirms Jainam's laptop check, so it goes in as a negative row. Its far heads are better on real text (h1 loss 5.69 vs 5.97), but that doesn't move acceptance.
- **k6: +0.03-0.04 tokens/step, but slower on T4.** Each extra head costs ~2 ms per pass (4 ms for two, at every size), so ×1.28 vs ×1.36 on IndicCorp. Heads 4-5 are accepted only 3-5% / 2-3% of the time, which doesn't pay for them.
- **Tree outlook:** with Jainam's real/offline correction (0.86 k4, 0.82 k6), the n64 trees give ≈ 2.38 tokens/step ÷ 1.30 ≈ ×1.83 for k4 and ≈ 2.41 ÷ 1.44 ≈ ×1.67 for k6. So **k4 is the better tree base on T4**, unless `generate_tree()` shows the deep nodes are accepted more than the oracle predicts.
- k6 h4/h5 in-group numbers rest on 58 / 26 targets: ignore them.

## Rsd line: sequential heads, longer training, structural loss (OM-8, official eval)

`Rsd_hi_k4_frozen_sd_seq` (sequential heads, #49), `Rsd_hi_k4_frozen_sd_long` (2× data, 2× steps: step 25000) and `Rsd_hi_k4_frozen_sd_struct` (+ R3's structural loss), all frozen ganga-1b (head 0 identical: 2.865 / 3.984). Kaggle T4, `hi_rules_v0`, 200 prompts, fp32 re-check 20/20 everywhere, 64 tokens per prompt. These are the first entries with per-prompt GI (#55).

| run | dataset | FixedK tokens/step | accept h1 / h2 / h3 | speed-up (tok/s) | ConfidenceCut | GI fixed_k |
|---|---|---|---|---|---|---|
| Rsd | indiccorp_eval | 1.67 | 41.7 / 17.4 / 9.1% | ×1.36 (37.5) | 1.36, ×1.14 | 74.2 |
| **Rsd_seq** | indiccorp_eval | **2.00** | **53.3 / 30.5 / 18.9%** | **×1.61** (44.3) | 1.60, ×1.32 | 76.4 |
| Rsd_long | indiccorp_eval | 1.72 | 43.9 / 19.1 / 10.1% | ×1.41 (38.3) | 1.42, ×1.20 | 74.6 |
| Rsd_struct | indiccorp_eval | 1.65 | 41.6 / 17.0 / 8.3% | ×1.35 (37.0) | 1.31, ×1.10 | 77.3 |
| Rsd | flores_hi | 1.60 | 39.0 / 15.5 / 7.1% | ×1.32 (36.4) | 1.30, ×1.09 | 80.2 |
| **Rsd_seq** | flores_hi | **1.89** | **49.1 / 26.5 / 15.9%** | **×1.52** (41.9) | 1.51, ×1.24 | 79.6 |
| Rsd_long | flores_hi | 1.66 | 41.4 / 17.9 / 7.8% | ×1.36 (37.1) | 1.34, ×1.13 | 78.6 |
| Rsd_struct | flores_hi | 1.59 | 39.0 / 15.0 / 6.0% | ×1.30 (35.6) | 1.26, ×1.06 | 81.9 |

Verify-pass cost (`bench_verify`, ms; × greedy pass of 35.3-35.9 ms):

| positions | 1 | 4 | 8 | 16 | 25 | 32 | 64 | + draft chain |
|---|---|---|---|---|---|---|---|---|
| Rsd (parallel heads) | 41.6 (1.17) | 42.6 (1.20) | 43.0 (1.21) | 43.8 (1.23) | 45.3 (1.27) | 45.5 (1.27) | 46.5 (1.30) | – |
| **Rsd_seq** (head 0 only) | 35.3 (0.99) | 36.3 (1.02) | 36.6 (1.03) | 37.0 (1.04) | 38.4 (1.08) | 38.5 (1.08) | 39.7 (1.12) | **7.0 ms** (3 steps) |

- **Sequential heads are the fastest model: ×1.61 / ×1.52 on T4**, +0.33 / +0.29 tokens/step over Rsd. That matches Jainam's +0.365 on 100 prompts. The far heads gain most: h3 acceptance doubles (9 → 19%).
- **Cost model checks out:** a step = verify pass (36.3 ms at 4 positions, head 0 only) + draft chain (7.0 ms) ≈ 1.22× greedy, so 2.00 ÷ 1.22 = ×1.64 predicted vs ×1.61 measured. The verify pass is *cheaper* than Rsd's (no extra heads), and the 3-step draft chain (~2.3 ms per step, dominated by the 30k-vocab `lm_head`) is the overhead.
- **Room left on the draft side:** with a ~3 ms chain (e.g. CUDA graphs, or top-k over a reduced vocab), the same acceptance would give ≈ ×1.75. Trees with seq heads need one chain per branch, so the draft cost grows with the tree width; the verify side stays cheap (64 nodes = 1.12×).
- **Rsd_long:** +0.05 tokens/step, ×1.41 vs ×1.36. That matches Jainam's +0.050, a small gain from 2× data and steps.
- **Rsd_struct:** same acceptance as Rsd (1.65 vs 1.67), but **GI is higher** (77.3 vs 74.2 on IndicCorp, 81.9 vs 80.2 on FLORES). The structural loss shapes *where* accepted spans end even on the frozen line. It needs a random-group control (an "Rsd_struct_random") before it can be read as linguistic.

## Sequential heads: seed replicate, 6 heads, Marathi fast line (OM-8, official eval)

Kaggle T4, step 12500, 200 prompts, fp32 re-check 20/20 everywhere, 64 tokens per prompt, per-prompt GI. Hindi uses `hi_rules_v0`. Marathi uses `mr_rules_v1` and `--ignore_eos on`, as in Jainam's checks; these runs don't stop early anyway. These are the first runs pushed with the Kaggle CLI. FLORES comes from the local copy (#62), and target counts are identical to R2 / R9.

| run | dataset | FixedK tokens/step | accept h1 / h2 / h3 (… h5) | speed-up (tok/s) | ConfidenceCut | GI fixed_k |
|---|---|---|---|---|---|---|
| Rsd_seq | indiccorp_eval | 2.00 | 53.3 / 30.5 / 18.9% | ×1.61 (44.3) | 1.60, ×1.32 | 76.4 |
| **Rsd_seq_s43** | indiccorp_eval | **1.99** | 52.6 / 29.8 / 18.6% | **×1.60** (43.7) | 1.59, ×1.30 | 78.0 |
| Rsd_seq_k6 | indiccorp_eval | 2.09 | 50.6 / 27.4 / 16.7 / 10.8 / 7.2% | ×1.53 (41.3) | 1.60, ×1.19 | 77.5 |
| Rsd_seq | flores_hi | 1.89 | 49.1 / 26.5 / 15.9% | ×1.52 (41.9) | 1.51, ×1.24 | 79.6 |
| **Rsd_seq_s43** | flores_hi | **1.91** | 49.5 / 27.4 / 16.4% | **×1.54** (42.7) | 1.51, ×1.24 | 78.7 |
| Rsd_seq_k6 | flores_hi | 1.97 | 46.7 / 24.6 / 14.6 / 8.8 / 5.4% | ×1.44 (39.7) | 1.51, ×1.12 | 80.5 |
| R9 (plain MTP) | indiccorp_eval (mr) | 1.31 | 24.4 / 6.0 / 1.2% | ×1.07 (24.4) | 1.07, ×0.89 | 77.3 |
| Rsd_mr (Misal-instruct head 0) | indiccorp_eval (mr) | 1.53 | 33.6 / 13.3 / 6.9% | ×1.25 (28.0) | 1.30, ×1.07 | 69.6 |
| Rsd_r8 | indiccorp_eval (mr) | 1.72 | 42.7 / 20.2 / 10.1% | ×1.37 (31.7) | 1.46, ×1.18 | 76.6 |
| **Rsd_r8_seq** | indiccorp_eval (mr) | **1.81** | 43.9 / 24.7 / 14.3% | **×1.44** (33.1) | 1.49, ×1.19 | 78.6 |
| Rsd_r8_seq_s43 | indiccorp_eval (mr) | 1.81 | 44.0 / 24.5 / 14.4% | ×1.44 (33.0) | 1.49, ×1.19 | 77.0 |
| Rsd_r8_seq_k6 | indiccorp_eval (mr) | 1.84 | 41.3 / 22.0 / 12.3 / 6.8 / 4.4% | ×1.29 (29.9) | 1.47, ×1.05 | 81.3 |
| R9 | flores_mr | 1.30 | 23.8 / 5.5 / 1.0% | ×1.05 (24.1) | 1.06, ×0.88 | 79.4 |
| Rsd_r8 | flores_mr | 1.68 | 41.3 / 18.9 / 9.2% | ×1.36 (30.8) | 1.42, ×1.16 | 77.3 |
| **Rsd_r8_seq** | flores_mr | **1.77** | 42.3 / 22.6 / 13.7% | **×1.45** (31.5) | 1.44, ×1.18 | 77.3 |
| Rsd_r8_seq_s43 | flores_mr | 1.79 | 43.7 / 23.5 / 14.0% | ×1.46 (31.6) | 1.44, ×1.17 | 79.9 |
| Rsd_r8_seq_k6 | flores_mr | 1.82 | 40.2 / 20.8 / 12.4 / 6.6 / 4.0% | ×1.31 (29.2) | 1.43, ×1.03 | 79.2 |

Greedy: 27.3-27.7 tok/s (ganga), 21.6-23.2 tok/s (Misal). Head 0 of the Rsd_r8 runs is R8 exactly (4.252 / 4.903).

Verify pass (head 0 only for seq) and draft chain, ms (median of 20):

| run | greedy pass | verify, 4 positions | verify, 64 positions | draft chain |
|---|---|---|---|---|
| Rsd_seq (hi, k4) | 35.5 | 36.3 | 39.7 | 7.0 (3 steps) |
| Rsd_seq_s43 (hi, k4) | 35.6 | 36.6 | 39.8 | 7.0 |
| Rsd_seq_k6 (hi) | 35.6 | 36.7 | 40.0 | 11.7 (5 steps) |
| Rsd_r8 (mr, parallel heads) | 44.1 | 51.7 | 56.2 | – |
| Rsd_r8_seq (mr, k4) | 43.6 | 41.8 | 45.8 | 10.4 (3 steps) |
| Rsd_r8_seq_k6 (mr) | 43.4 | 42.4 | 46.6 | 17.3 (5 steps) |

The two Marathi runs on one GPU at a time give noisier timings (Rsd_r8_seq_s43 has outliers at 1 / 8 positions, so it's left out of this table). The decoding speed-ups above are unaffected; they come from the decode loop, not from this benchmark.

- **Sequential heads replicate:** Rsd_seq_s43 gives ×1.60 / ×1.54 vs ×1.61 / ×1.52 for seed 42, and Rsd_r8_seq_s43 gives ×1.44 / ×1.46 vs ×1.44 / ×1.45.
- **Six sequential heads are slower in both languages**, despite more tokens per step (+0.08-0.10 Hindi, +0.03-0.05 Marathi). Each extra chain step costs ~2.3 ms (ganga) / ~3.5 ms (Misal's 47k vocabulary), and heads 4-5 are accepted only 3-11% of the time. So k4 is the setup for chain decoding. Whether k6 pays off with trees needs `generate_tree()`.
- **Marathi fast line:** Rsd_r8_seq gives ×1.44 / ×1.45 with head 0 = R8 exactly, vs R9 ×1.07 / ×1.05 and Rsd_mr ×1.25 / ×1.20 (Misal-instruct head 0). Sequential heads add less in Marathi (+0.09 tokens/step over Rsd_r8) than in Hindi (+0.33 over Rsd). That matches Jainam's laptop numbers (1.78 / 1.89 / 1.94 / 1.86).
- **Misal's draft chain costs more than ganga's** (10.4 vs 7.0 ms for 3 steps; bigger vocabulary and hidden size), so the draft-side overhead matters more for Marathi.

## Word-group claims, P1 (#39): R4 / R4a, R7s, R10_s43 / R10a_s43, Rsd_seq_struct (OM-8, official eval)

Kaggle T4, step 12500, 200 prompts, fp32 re-check 20/20 everywhere, per-prompt outputs stored (#64). In-group / GI measured with `hi_rules_v0` (Hindi) or `mr_rules_v1` (Marathi) for every run, including the random-group controls.

In-group top-1 h1 / h2 / h3 and Group Integrity (FixedK):

| run | groups trained on | IndicCorp in-group | FLORES in-group | GI IndicCorp | GI FLORES | FixedK tokens/step, speed-up (IndicCorp) |
|---|---|---|---|---|---|---|
| R2 | none | 23.4 / 19.0 / 17.5 | 12.5 / 12.1 / 12.1 | 75.7 | 79.0 | 1.39, ×1.15 |
| R3 | hi_rules_v0 | 25.8 / 24.1 / 22.2 | 14.3 / 14.0 / 13.1 | 77.8 | 82.4 | 1.39, ×1.15 |
| R6a | random | 25.2 / 21.5 / 19.9 | 13.7 / 12.9 / 11.1 | 74.9 | 79.5 | 1.40, ×1.14 |
| **R7s** | stanza parser | 25.5 / 22.8 / 19.9 | 14.2 / 14.2 / 13.4 | **79.2** | **82.4** | 1.39, ×1.14 |
| R3_s43 | hi_rules_v0 | 23.6 / 21.2 / 19.9 | 12.1 / 13.0 / 13.4 | 79.2 | 81.7 | 1.38, ×1.13 |
| R6a_s43 | random | 25.2 / 21.4 / 18.1 | 12.5 / 13.2 / 14.8 | 77.1 | 81.1 | 1.40, ×1.15 |
| **R7s_s43** | stanza parser | 24.9 / 23.2 / 18.1 | 12.8 / 13.4 / 15.8 | **79.7** | **82.0** | 1.38, ×1.14 |
| **R4** | hi_rules_v0 + contrastive | 26.1 / 24.1 / 24.0 | 14.1 / 14.0 / 13.1 | **79.4** | **83.8** | 1.40, ×1.14 |
| R4a | random + contrastive | 25.3 / 20.8 / 20.5 | 13.6 / 13.6 / 10.7 | 76.9 | 81.3 | 1.40, ×1.14 |
| R10_s43 (mr) | mr_rules_v1 | 31.2 / 18.9 / 15.5 | 18.7 / 10.2 / 10.1 | 81.0 | 78.6 | 1.33, ×1.09 |
| R10a_s43 (mr) | random_mr_v1 | 31.3 / 18.9 / 15.5 | 18.8 / 9.9 / 10.1 | 79.9 | 76.5 | 1.34, ×1.10 |
| Rsd_seq_struct | hi_rules_v0 (frozen, seq heads) | 25.3 / 24.8 / 26.3 | 13.5 / 12.8 / 13.4 | 77.0 | 80.8 | 2.00, ×1.60 |
| Rsd_seq_struct_random | random (frozen, seq heads) | 24.3 / 23.7 / 24.0 | 13.1 / 12.9 / 10.4 | 77.0 | 80.9 | 2.01, ×1.63 |

- **GI favours linguistic groups over their random control in all 8 run × dataset cells** of the four new unfrozen pairs (and in all 4 of the earlier R3 − R6a cells):
  - R7s − R6a: +4.3 / +2.9 (seed 42) and +2.6 / +0.9 (seed 43);
  - R4 − R4a: +2.5 / +2.5;
  - R10_s43 − R10a_s43: +1.1 / +2.1;
  - R3 − R6a: +2.9 / +2.9 and +2.1 / +0.6 (earlier rows).

  The parser groups (R7s) align at least as well as the rules (R3). These GIs are scored with `hi_rules_v0`; the same comparison scored with stanza groups follows (`scripts/rescore_gi.py`).
- **In-group accuracy again doesn't separate linguistic from random groups**, except at h2/h3 on some single seeds (R4 − R4a: +3.3 / +3.5 at h2/h3, R7s > R6a at h2 on both seeds). Acceptance and speed are the same across the R3 line (1.38-1.40 tokens/step).
- **Contrastive loss (R4) adds nothing over R3** on accuracy or acceptance. It's the same row as Jainam's bootstrap.
- **On the fast model, group loss changes nothing:** Rsd_seq_struct = its random control on speed (×1.60 / ×1.63) **and** GI (77.0 / 77.0, 80.8 / 80.9). Sequential heads already produce spans that align as well as the structurally trained ones. The structural loss's GI effect is confined to the parallel-head line.

## Second seed: R3_s43 vs R6a_s43 (OM-8, official eval, C1 control)

`R3_hi_k4_struct_s43` / `R6a_hi_k4_struct_random_s43` (seed 43 + shuffled data), step 12500, Kaggle T4, `--grouper hi_rules_v0` for both (same in-group target counts as R2/R3: 1,886 / 557 / 171 on IndicCorp; 3,116 / 1,009 / 298 on FLORES).

In-group top-1 (h1 / h2 / h3):

| dataset | R2 | R3 (s42) | R6a (s42) | R3_s43 | R6a_s43 | R3 mean | R6a mean |
|---|---|---|---|---|---|---|---|
| indiccorp_eval | 23.4 / 19.0 / 17.5 | 25.8 / 24.1 / 22.2 | 25.2 / 21.5 / 19.9 | 23.6 / 21.2 / 19.9 | 25.2 / 21.4 / 18.1 | 24.7 / 22.7 / 21.1 | 25.2 / 21.5 / 19.0 |
| flores_hi | 12.5 / 12.1 / 12.1 | 14.3 / 14.0 / 13.1 | 13.7 / 12.9 / 11.1 | 12.1 / 13.0 / 13.4 | 12.5 / 13.2 / 14.8 | 13.2 / 13.5 / 13.3 | 13.1 / 13.1 / 13.0 |

| run | IndicCorp FixedK tokens/step, speed-up | FLORES FixedK | h0 loss IndicCorp / FLORES |
|---|---|---|---|
| R3_s43 | 1.38, ×1.13 | 1.34, ×1.10 | 2.853 / 3.941 |
| R6a_s43 | 1.40, ×1.15 | 1.36, ×1.12 | 2.850 / 3.942 |

- **The R3 > R6a gap at h2/h3 does not replicate.** On seed 43, random groups match or beat linguistic groups at every head on both sets (h1 IndicCorp 25.2 vs 23.6). Averaged over the two seeds, R3 − R6a is −0.5 / +1.2 / +2.1 points on IndicCorp and +0.1 / +0.4 / +0.3 on FLORES, within seed-to-seed variation (R3 alone moves 2.2-2.9 points between seeds).
- **C1 reading:** the structural loss raises in-group accuracy over R2, and **random groups do the same.** The gain comes from the extra loss terms, not from the linguistic grouping. This agrees with Jainam's bootstrap on #39 and with Marathi (R10 = R10a).
- Head 0, acceptance and speed are unchanged vs R2 on both seeds; fp32 re-check 20/20, `mean_new_tokens` 64.

## Marathi: R10 / R10a / Rsd_mr (OM-8, official eval)

Step 12500, Kaggle T4, `--grouper mr_rules_v1` for all runs (R10a's config says `random_mr_v1`), IndicCorp mr eval + FLORES mar_Deva, 200 prompts. In-group target counts match R9 (h1 3,931 / 5,255). Rsd_mr and Rsd_mr_soft are frozen Misal-instruct, evaluated with `ignore_eos` (#46): 64.0 tokens per prompt. R9 and R10 average ~63.

In-group top-1, h1 / h2 / h3:

| run | IndicCorp mr | FLORES mr | h0 loss IndicCorp / FLORES |
|---|---|---|---|
| R9 (plain MTP) | 28.3 / 17.4 / 11.7 | 16.2 / 8.2 / 8.1 | 4.258 / 4.905 |
| R10 (mr_rules_v1) | 30.5 / 18.6 / 13.3 | 18.5 / 10.8 / 10.1 | 4.288 / 4.953 |
| R10a (random_mr_v1) | 30.9 / 18.8 / 12.6 | 18.5 / 10.4 / 10.9 | 4.287 / 4.953 |
| Rsd_mr | 22.2 / 11.7 / 8.1 | 12.6 / 5.5 / 5.4 | 4.890 / 5.555 (= Misal-instruct) |

Decoding (FixedK; ConfidenceCut in brackets for GI):

| run | dataset | tokens/step | accept h1 / h2 / h3 | speed-up (tok/s) | GI fixed_k (CC) | fp32 re-check |
|---|---|---|---|---|---|---|
| R9 | indiccorp_eval | 1.31 | 24.4 / 6.0 / 1.2% | ×1.07 (24.4) | 77.3 (84.9) | 20/20 |
| R10 | indiccorp_eval | 1.34 | 26.6 / 6.7 / 1.2% | ×1.10 (24.7) | 81.1 (86.4) | 20/20 |
| R10a | indiccorp_eval | 1.35 | 27.1 / 7.1 / 1.3% | ×1.08 (24.8) | 80.8 (83.9) | 20/20 |
| **Rsd_mr** | indiccorp_eval | **1.53** | 33.6 / 13.3 / 6.9% | **×1.25** (28.0) | 69.6 (67.3) | 20/20 |
| Rsd_mr_soft | indiccorp_eval | 1.52 | 34.0 / 12.4 / 6.3% | ×1.24 (27.5) | 67.8 (63.6) | 20/20 |
| R9 | flores_mr | 1.30 | 23.8 / 5.5 / 1.0% | ×1.05 (24.1) | 79.4 (78.1) | 20/20 |
| R10 | flores_mr | 1.31 | 25.0 / 5.8 / 0.7% | ×1.07 (24.1) | 79.1 (87.6) | 20/20 |
| R10a | flores_mr | 1.30 | 24.7 / 4.8 / 0.7% | ×1.05 (23.8) | 77.6 (85.4) | 20/20 |
| **Rsd_mr** | flores_mr | **1.46** | 30.8 / 11.5 / 5.1% | **×1.20** (26.8) | 69.6 (64.4) | 20/20 |
| Rsd_mr_soft | flores_mr | 1.46 | 31.2 / 10.8 / 4.6% | ×1.20 (26.4) | 67.0 (62.4) | 20/20 |

Greedy: 22.1-22.9 tok/s.

- **The structural loss raises in-group accuracy in Marathi too, and random groups give the same gain** (R10 = R10a within ±0.8 at every head). This matches Jainam's bootstrap on #39 and Hindi's two-seed result.
- **GI leans towards linguistic groups, but only slightly:** R10 − R10a = +0.3 / +1.5 (FixedK) and +2.5 / +2.2 (CC). That's one seed with no CI. It points the same way as Hindi (R3 > R6a in 8 of 8 cells), with smaller differences.
- **Self-distillation transfers to Marathi:** Rsd_mr reaches 1.53 / 1.46 tokens/step, ×1.25 / ×1.20, vs R9's ×1.07 / ×1.05. It matches Jainam's masked check (1.539). It speeds up Misal-instruct as is, whose head 0 is 0.63 nats weaker on IndicCorp than the LoRA-tuned R9.
- **Rsd_mr's GI is lower (~68-70%).** Its accepted spans come from Misal-instruct's own text and its heads were never trained on word groups, so its spans line up with `mr_rules_v1` boundaries less often. This is a different model from R9/R10, so the GI numbers aren't comparable across that line.
- **Soft labels = hard labels** in Marathi too (negative row).
- These were run before #55, so they have no per-prompt GI.

## Boundary alignment across groupers (P1-5, #39 / #53): GI scored with rules and with the stanza parser

Group Integrity (FixedK) of the R3 line, scored two ways from the same stored decoding outputs (#64, `scripts/rescore_gi.py`): with `hi_rules_v0` and with the stanza dependency-parser groups. R3 / R4 were trained on the rules, R7s on the parser groups, R6a / R4a on random groups. Kaggle T4, 200 prompts per dataset (identical prompts across runs), paired bootstrap over prompts (2,000 resamples), 95% CI; ✓ = CI excludes 0.

Absolute GI (%), IndicCorp / FLORES:

| run (trained on) | scored with rules | scored with stanza |
|---|---|---|
| R3 (rules) | 77.8 / 82.4 | 73.7 / 75.2 |
| R3_s43 (rules) | 79.2 / 81.7 | 75.2 / 75.0 |
| R7s (stanza) | 79.2 / 82.4 | 75.8 / 76.0 |
| R7s_s43 (stanza) | 79.7 / 82.0 | 77.2 / 76.3 |
| R4 (rules + contrastive) | 79.4 / 83.8 | 73.7 / 76.0 |
| R6a (random) | 74.9 / 79.5 | 71.7 / 74.4 |
| R6a_s43 (random) | 77.1 / 81.1 | 73.7 / 75.2 |
| R4a (random + contrastive) | 76.9 / 81.3 | 73.2 / 74.6 |

Linguistic − random (IndicCorp | FLORES):

| pair | scored with rules | scored with stanza |
|---|---|---|
| R3 − R6a (s42) | +2.9 [+1.0, +4.8] ✓ \| +2.8 [+1.3, +4.3] ✓ | +2.0 [−0.1, +4.1] \| +0.9 [−0.8, +2.5] |
| R3_s43 − R6a_s43 | +2.1 [−0.3, +4.4] \| +0.7 [−1.6, +2.9] | +1.5 [−0.8, +3.9] \| −0.2 [−2.7, +2.3] |
| R7s − R6a (s42) | +4.3 [+2.4, +6.4] ✓ \| +2.9 [+1.4, +4.3] ✓ | +4.1 [+2.1, +6.3] ✓ \| +1.7 [−0.1, +3.3] |
| R7s_s43 − R6a_s43 | +2.6 [+0.9, +4.3] ✓ \| +0.9 [−1.0, +2.8] | +3.5 [+1.8, +5.3] ✓ \| +1.1 [−1.1, +3.3] |
| R4 − R4a | +2.5 [+0.2, +4.7] ✓ \| +2.5 [+0.3, +4.7] ✓ | +0.5 [−1.8, +2.8] \| +1.4 [−0.9, +3.7] |

- **Parser-trained groups give the robust alignment gain.** R7s beats its random control on IndicCorp for both seeds and both scorings, including the rules grouping it never saw (+4.3 / +2.6). So R7s's drafted spans end on word-group boundaries more often under an independent definition of a group, not just under its own labels.
- **Rules-trained groups transfer less:** R3 and R4 are significant only when scored with the rules they were trained on, and not on R3's second seed. Under the parser scoring their gains shrink to +0.5 to +2.0 (n.s.). Part of the rules-scored gap is the model fitting the specific boundaries it was given.
- **Out of domain (FLORES)** the direction holds (9 of 10 point estimates positive), but only two cells are significant.
- **Overall:** 19 of 20 point estimates are positive. Acceptance and speed are identical across all of these runs (1.38-1.40 tokens/step), so the alignment difference is not a by-product of accepting more or fewer drafts.
- R3 / R3_s43 / R6a / R6a_s43 were re-decoded to store their outputs (`--no_heads`). GI and accepted length reproduced the committed values exactly, so only the `spec_decode` section of their eval JSONs changed.

## Tree decoding on T4 (`generate_tree`, #67)

Real tree verification of Jainam's fitted trees (#38 / #42, and `fit_tree.py` on Kaggle for Rsd_seq). Kaggle T4, step 12500, 200 prompts, 64 new tokens, `hi_rules_v0`. **fp32 re-check 20/20 for every tree**, so trees are lossless like the chain. Tokens/step (speed-up):

| run | dataset | chain (FixedK) | entropy n25 | entropy n40 | entropy n64 | static n25 | static n64 |
|---|---|---|---|---|---|---|---|
| Rsd (k4, parallel) | indiccorp_eval | 1.67 (×1.36) | 2.32 (×1.67) | **2.42 (×1.70)** | 2.53 (×1.67) | 2.28 (×1.65) | 2.50 (×1.66) |
| Rsd (k4, parallel) | flores_hi | 1.60 (×1.32) | 2.22 (×1.60) | **2.33 (×1.63)** | 2.42 (×1.61) | 2.19 (×1.59) | 2.39 (×1.59) |
| Rsd_k6 (parallel) | indiccorp_eval | 1.71 (×1.27) | 2.35 (×1.54) | **2.45 (×1.56)** | 2.57 (×1.55) | – | – |
| Rsd_k6 (parallel) | flores_hi | 1.63 (×1.21) | 2.23 (×1.47) | **2.34 (×1.50)** | 2.45 (×1.48) | – | – |
| **Rsd_seq (k4, sequential)** | indiccorp_eval | 2.00 (×1.61) | 2.70 (×1.69) | 2.82 (×1.72) | 2.93 (×1.69) | – | **2.90 (×1.89)** |
| **Rsd_seq (k4, sequential)** | flores_hi | 1.89 (×1.53) | 2.58 (×1.62) | 2.70 (×1.65) | 2.79 (×1.61) | – | **2.78 (×1.81)** |

Greedy: 27.4-27.7 tok/s. Mean nodes verified per step ≈ the budget (24.3 / 38.8 / 62.1).

- **Best overall: Rsd_seq with the static 64-node tree, ×1.89 / ×1.81**, up from ×1.61 / ×1.53 for its chain and ×1.36 / ×1.32 for the Rsd chain. Lossless.
- **Parallel heads (Rsd):** trees add +0.65-0.85 tokens/step and +0.30-0.34 speed-up over the chain. The best size is 40 nodes; 64 accepts more but costs more per pass.
- **Sequential heads: static beats entropy at the same size** (×1.89 vs ×1.69 at 64 nodes) although entropy accepts slightly more (2.93 vs 2.90). The entropy policy needs the greedy draft chain's logits to pick its cell, an extra ~7 ms chain per step. The static tree needs no logits before drafting.
- **Rsd_k6 stays behind Rsd with trees too** (×1.56 vs ×1.70 at n40): deeper trees add ~0.03 tokens/step, but the two extra heads cost ~4 ms on every pass.
- **Offline vs real:** Rsd entropy n40 gives 2.66 tokens/step offline vs 2.42 real (0.91); Rsd_seq static n64 gives 3.13 vs 2.90 (0.93). Better than the 0.86-0.87 Jainam assumed (#33).
- **Engine overhead:** per step, trees cost more than the bare verify pass (Rsd_k6 n40: ~1.57× vs ~1.40× greedy from `bench_verify`). #67's second commit cuts the per-step device syncs. These runs predate it, so the speed-ups above are a lower bound.
- **GI** of tree outputs is 74-77% (FixedK chain 74-80%). Trees accept longer spans that cross more group boundaries.

## Rsd_struct vs Rsd_struct_rand: word groups on the frozen line (OM-8, official eval)

`Rsd_hi_k4_frozen_sd_struct_random` (`jainam2142/mtp-run-Rsd-struct-rand`) = Rsd_struct with random groups. Kaggle T4, step 12500, `hi_rules_v0`, 200 prompts, fp32 re-check 20/20. Head 0 is ganga-1b in both runs.

| run | in-group top-1 h1 / h2 / h3 (IndicCorp) | FixedK tok/step, speed-up (IndicCorp / FLORES) | GI fixed_k (IndicCorp / FLORES) |
|---|---|---|---|
| Rsd_struct (rules groups) | 21.5 / 17.6 / 17.0 | 1.65, ×1.35 / 1.59, ×1.30 | 77.3 / 81.9 |
| Rsd_struct_rand (random groups) | 20.4 / 15.1 / 14.6 | 1.66, ×1.36 / 1.59, ×1.30 | 75.5 / 80.5 |

- **GI difference, paired bootstrap:** **+1.9 [+0.8, +2.9]** (IndicCorp) and **+1.4 [+0.4, +2.4]** (FLORES); both CIs exclude 0. This is a second, independent GI result on a different model line (frozen backbone, self-distilled heads), significant on both datasets including FLORES.
- **In-group accuracy** is higher too (+1.2 / +2.5 / +2.3). Speed is identical.
- **Caveat:** scored with the rules the run was trained on. Rsd_struct was evaluated before outputs were stored (#64), so it can't be re-scored with stanza. On the sequential-head model the same comparison is null (Rsd_seq_struct = random control).

## Final tree timings and prompt lookup (OM-8, official eval)

Re-timed with the final engine (#67's per-step overhead fix), plus prompt-lookup chains grafted onto the tree (#69:
up to 4 chains of 6 tokens copied from earlier occurrences of the last 3, else 2, tokens). Kaggle T4, 200 prompts,
fp32 re-check 20/20 for every entry. Tokens/step (speed-up):

| model | policy | IndicCorp | FLORES |
|---|---|---|---|
| Rsd | chain | 1.67 (×1.36) | 1.60 (×1.31) |
| Rsd | tree entropy n40 | 2.42 (×1.74) | 2.33 (×1.68) |
| Rsd | tree static n64 | 2.50 (×1.73) | 2.39 (×1.68) |
| Rsd | tree entropy n40 + lookup | 3.01 (**×2.12**) | 3.14 (**×2.23**) |
| Rsd | tree static n64 + lookup | 3.10 (×2.11) | 3.20 (×2.20) |
| Rsd_seq | chain | 2.00 (×1.61) | 1.89 (×1.53) |
| Rsd_seq | tree entropy n40 | 2.82 (×1.79) | 2.70 (×1.71) |
| Rsd_seq | tree static n40 | 2.79 (**×2.02**) | 2.67 (**×1.94**) |
| Rsd_seq | tree static n64 | 2.90 (×2.01) | 2.78 (×1.93) |
| Rsd_seq | tree entropy n40 + lookup | 3.40 (**×2.12**) | 3.56 (**×2.22**) |
| Rsd_seq | tree static n40 + lookup | 3.36 (×2.11) | 3.53 (×2.22) |

- **Overhead fix:** about +0.1 speed-up on the same tree (Rsd_seq static n64: ×1.89 → ×2.01; Rsd entropy n40: ×1.70 → ×1.74).
- **Prompt lookup adds +0.6 tokens/step,** matching Jainam's offline +0.37-0.6 (#33). Fastest lossless configuration:
  **×2.12 IndicCorp / ×2.22-2.23 FLORES**.
- **With lookup, Rsd = Rsd_seq** (×2.12 / ×2.23 vs ×2.12 / ×2.22). The copied chains supply the long correct drafts that
  sequential heads were providing, at no draft-chain cost for the parallel heads.
- **Caveat:** greedy output of a 1B model repeats itself, so in-context lookup gains are larger than they would be under sampling.

## Marathi trees and prompt lookup: Rsd_r8_seq (OM-8, official eval)

Trees fitted on Kaggle (`fit_tree.py`, 300 IndicCorp mr train prompts; offline accepted drafts/step: chain 1.10,
n40 static 1.75 / entropy 1.77, n64 1.84 / 1.85). Final engine, `--ignore_eos on`, 200 prompts, 64 tokens per
prompt, fp32 re-check 20/20 everywhere.

| Rsd_r8_seq (head 0 = R8) | IndicCorp mr tok/step (speed-up) | FLORES mr |
|---|---|---|
| chain | 1.81 (×1.44) | 1.77 (×1.44) |
| tree static n40 | 2.41 (**×1.77**) | 2.36 (×1.75) |
| tree entropy n40 | 2.43 (×1.51) | 2.38 (×1.50) |
| tree static n64 | 2.48 (×1.74) | 2.44 (×1.73) |
| tree static n40 + lookup | 3.87 (**×2.35**) | 3.77 (×2.33) |
| tree entropy n40 + lookup | 3.88 (×2.35) | 3.81 (×2.34) |

- **Trees help Marathi as in Hindi:** +0.6 tokens/step and ×1.44 → ×1.77 (static n40). Entropy trees are again slower for
  sequential heads (the extra chain to read the cells).
- **Prompt lookup adds +1.45 tokens/step in Marathi** (Hindi +0.6), to **×2.35 / ×2.33**, lossless. Misal's greedy
  continuations repeat more, so the sampling caveat applies even more strongly here.

## Drafter experiments, Phase A (DR-1 / DR-3 / DR-4, no training; #76)

Rsd_seq on a Kaggle T4, fp32 weights + fp16 autocast. Files: `results/Rsd_hi_k4_frozen_sd_seq/draft_profile.json`,
`draft_agreement.json`, `results/drafter/vocab_rank_hi.json` (100k IndicCorp train sentences, 6.5M tokens).

**DR-1, one draft step (median of 200, CUDA events):**

| part | ms |
|---|---|
| LM head, as decoding runs it (fp32 weight cast to fp16 by autocast on every call) | **2.01** |
| LM head with a weight cast to fp16 once | **0.54** |
| SeqBlock | 0.18 |
| embedding lookup / argmax / argmax + CPU sync | 0.03 / 0.03 / 0.05 |
| full 3-step `draft_chain` | 6.96 |

torch.profiler over 20 chains: `aten::copy_` (the weight casts) is **72%** of CUDA time, the matrix-vector products
23%. Autocast caches casts only for weights that require grad, and the backbone and LM head are frozen. The
per-step CPU sync and the vocabulary size, the two suspects in DRAFTER.md, are negligible.

**DR-3 / DR-4, offline agreement with the full LM head** (500 eval sentences, ~25k positions per head):

| predictor | matmul vs full | agreement h1 / h2 / h3 | fp16 ms per step |
|---|---|---|---|
| full | 1.00 | 100 / 100 / 100 | 0.54 |
| top-16k vocab (96.3% token coverage) | 0.53 | 98.5 / 98.9 / 99.0 | 0.29 |
| top-8k (89.7%) | 0.27 | 95.8 / 96.5 / 96.8 | 0.16 |
| top-4k (81.7%) | 0.13 | 91.8 / 93.1 / 93.6 | 0.11 |
| low-rank SVD r = 512 / 256 / 128 / 64 | 0.27-0.03 | 37 / 28 / 19 / 11 (h1) | 0.21-0.09 |

**Decisions:**
- **DR-2 (engine) is the win:** cast linear weights once. It should cut the draft chain from ~7.0 ms to about 2 ms, and
  every verify / greedy pass pays the same casts. Built as `fast_linear` (#80, bit-identical); T4 measurement running.
- **DR-4 low-rank: drop.** The LM head is far from low-rank (11-37% agreement), and a trained low-rank head (DR-6) has a weak prior.
- **DR-3 top-N: marginal after DR-2.** It saves at most ~0.25 ms per step (top-16k at ~1% agreement loss); revisit
  once DR-2's numbers are in.

## Drafter DR-2: cached fp16 linear weights (`fast_linear`, #80; official eval)

`evaluate.py --fast_linear`: every (untied) linear layer computes from an fp16 copy of its weight made once, instead
of autocast re-casting the frozen fp32 weight on every call. Outputs are bit-identical (tests), and the fp32 re-check
stays 20/20. The greedy baseline uses it too, so the speed-up ratios are against a faster greedy. Kaggle T4, 200 prompts.

| | without | **with fast_linear** |
|---|---|---|
| greedy, Rsd_seq (tok/s) | 27.6 | **36.5** |
| one verify pass, 4 positions (ms) | 36.3 | **26.7** |
| draft chain, 3 steps (ms) | 7.0 | **2.4** |

| model | policy | IndicCorp tok/step, speed-up | FLORES | tok/s (IndicCorp) |
|---|---|---|---|---|
| Rsd_seq | chain | 2.00, ×1.82 | 1.89, ×1.71 | 66 |
| Rsd_seq | tree static n64 | 2.89, ×2.39 | 2.78, ×2.27 | 87 |
| **Rsd_seq** | **tree static n64 + lookup** | 3.49, **×2.64** | 3.63, **×2.70** | **96** |
| Rsd | chain | 1.67, ×1.56 | 1.60, ×1.49 | 55 |
| Rsd | tree static n64 | 2.50, ×2.13 | 2.39, ×2.02 | 75 |
| Rsd | tree static n64 + lookup | 3.10, ×2.59 | 3.20, ×2.64 | 91 |

- **Greedy itself gets 32% faster** (27.6 → 36.5 tok/s): about a quarter of every forward pass was re-casting weights.
- **The speed-ups rise even against the faster greedy:** the verify pass now costs about one greedy pass (1.0-1.05×,
  `bench_verify_fast.json`) and the draft chain shrinks 3×.
- **End to end:** Rsd_seq + tree + lookup decodes at **96-99 tok/s vs the original 27.6 tok/s greedy, ×3.5 with
  identical output** (×1.32 from the cast fix × ×2.64-2.70 from speculative decoding).
- With the draft chain cheap, sequential heads (×2.64 / ×2.70) stay ahead of parallel heads (×2.59 / ×2.64), narrowly.


## Drafter DR-8 `seq_ctx`: context attention for the sequential heads (Phase B1 deciding test, #82; negative)

Does letting each future slot re-read the whole context (one attention read per slot, query from h_t, zero-init output)
make the drafts better *enough* to pay for itself? Two 2k-step pilots on the same 20k self-distilled texts, same seed:
Rsd_seq_p2k (reference) and Rsd_seq_ctx_p2k (ctx_dim 256, 4 heads). Kaggle T4, `--fast_linear`, 200 prompts, the
Rsd_seq static n40 tree, prompt lookup 4×6. Differences are ctx − seq, paired bootstrap over prompts (`compare_spec.py`).

| policy | IndicCorp tok/step: seq → ctx (Δ, 95% CI) | speed-up seq → ctx | FLORES tok/step (Δ, 95% CI) | speed-up |
|---|---|---|---|---|
| chain | 1.741 → 1.796 (+0.055 [+0.040, +0.073]) | ×1.588 → ×1.597 | 1.710 → 1.747 (+0.037 [+0.023, +0.051]) | ×1.573 → ×1.556 |
| tree static n40 | 2.481 → 2.543 (+0.063 [+0.042, +0.083]) | ×2.087 → ×2.071 | 2.411 → 2.469 (+0.058 [+0.039, +0.078]) | ×1.937 → ×1.887 |
| tree + lookup | 3.090 → 3.123 (+0.033 [+0.012, +0.055]) | ×2.264 → ×2.187 | 3.267 → 3.297 (+0.030 [+0.008, +0.054]) | ×2.410 → ×2.324 |

Teacher-forced at step 2000: head-1 top-1 17.5% → 18.1%, head 2 13.4% → 13.6%, head 3 12.1% → 11.9%. fp32 re-check
20/20 for every row. Draft chain 2.59 → 3.07 ms (`bench_verify.json`).

- **The context read helps acceptance, significantly but slightly:** +2-3.5% tokens/step on every policy and dataset.
- **It does not pay:** the extra attention read costs +0.48 ms per step, so the speed-up moves by +0.01 to −0.09.
  Fails the keep gate (≥ +0.05× with the CI excluding 0). To clear it, tree + lookup would need about +0.11
  tokens/step, three times the measured gain; longer training would have to triple it.
- **Decision:** seq_ctx dropped, and with it the Phase B2 ideas that build on the context read (DR-8 roles, DR-9).
  The pilots stay as a negative row; the Rsd_seq line remains the best drafter.

## TA's grouping, ×4 group losses, lossy decoding, drafter analysis (Jainam, 2026-10-22)

Write-up: `docs/paper/analysis.md`. Official evals: Kaggle T4, `hi_rules_v0`, 200 prompts, as OM-8; earlier RTX 4060
cross-checks of R11*/R12a are in `results/laptop_crosscheck/` (not comparable for speed).

| ID | config | Kaggle dataset | what | IndicCorp / FLORES vs its random twin |
|---|---|---|---|---|
| R11 / R11a | R11.yaml / R11a.yaml (= R3 recipe, `karthika` / `random_karthika`, cached labels) | `jainam2142/mtp-run-r11-all` | TA's grouping (Karthika et al. 2025 Table 4 rules on stanza HDTB) vs size-matched random | GI (TA's grouping) +2.6* / +3.0*; tok/step −0.01 / −0.02*; top-1 ≤ 0.3 pts |
| R11_s43 / R11a_s43 | seed 43 + shuffle | `jainam2142/mtp-run-r11-all`, `jainamrampuria/mtp-run-r11-s43` | replicate | GI +2.5* / +1.4; tok/step n.s. |
| R12 / R12a | R12.yaml / R12a.yaml (= R3 / R6a, λ_S2 0.4, λ_S3 1.0: group losses ×4) | kernel output `jainamrampuria/mtp-r12` | dose-response | GI (rules) +6.6* / +5.1*; tok/step −0.02 / −0.03*; head-1 loss 5.380 vs 5.291 |

- **Lossy (typical) acceptance**, ε ∈ {0.5, 0.2, 0.1, 0.05, 0.02} on R12/R12a, R3/R6a (+s43), R11/R11a (+s43), R2
  (`results/lossy/`, kernels `jainam2142/mtp-lossy-a`, `jainamrampuria/mtp-lossy-b`, code f162e95): at matched
  tokens/step, repeat-free ppl under plain ganga, linguistic better in 1 of 45 cells, worse in 11.
- **Sampling acceptance** (Σ min(p, q), teacher-forced): linguistic heads gain inside groups (+1.4 to +5.3 pts, head 1)
  and lose at group starts (−0.5 to −2.8), all pairs; `results/drafter/sampling_acceptance.json`.
- **Drafts as text:** group training raises the share of function-word drafts (+3 to +11 pts) and lowers judge
  fluency; 60-69% of parallel-head draft blocks repeat a token; `results/drafter/draft_text.json`.
- **Function-word penalty** at inference (global or boundary-gated): never raises tokens/step;
  `results/drafter/function_penalty.json`.

## Drafter idea `seq_wide`: offline sizing check (no training)

Idea: draft in a wider state z of D > H numbers (z_0 = [h_t; SiLU(U h_t)], one word read out per step through a learned
P: D → H, the word folded back with z ← z + F([z; e(word)])), so the drafter has room to lay out several future words
side by side. Rsd_seq is D = H. Before training, measure how much room the next three states need and how much of it
h_t can predict (`scripts/wide_sizing.py`): frozen ganga-1b (H = 2048), 60k positions of self-distilled text to fit,
25.6k positions of the 500 eval sentences to measure. "logit" = distances as the LM head sees them (each state
multiplied by (WᵀW)^½).

| | raw: PCs 95% / 99% | logit: PCs 95% / 99% |
|---|---|---|
| one future state h_{t+1} | 1587 / 1928 | 951 / 1633 |
| three stacked [h_{t+1}; h_{t+2}; h_{t+3}] | 4228 / 5537 | 2426 / 4506 |
| linear prediction from h_t | 1014 / 1619 | 433 / 1121 |
| MLP prediction from h_t | 1070 / 2121 | 411 / 1266 |

| variance explained (R²) | t+1 | t+2 | t+3 | all |
|---|---|---|---|---|
| linear, raw | 22.2% | 12.9% | 9.4% | 14.8% |
| MLP, raw | 20.3% | 11.9% | 8.5% | 13.6% |
| linear, logit | 37.8% | 26.4% | 21.8% | 28.7% |
| MLP, logit | 36.3% | 24.6% | 19.4% | 26.7% |

- Holding three future states would take room (2.5-2.7× one state), but **only 15-29% of that future can be predicted
  from h_t**, and the predictable part fits in 400-1000 dimensions at 95% (1100-1600 at 99%): **less than H**.
- **Non-linear features of h_t add nothing:** the MLP (hidden 2H, early stopping) does not beat the linear map.
  Caveat: it trains on standardised targets, so it is slightly disadvantaged on raw-variance R²; it is nowhere near
  ahead.
- **Verdict:** the wide state has no measurable room to fill: z_0 is built from h_t alone, and what h_t knows about
  the next three states already fits in H numbers. Implemented and tested (`mtp/model/wide_heads.py`,
  `head_type: seq_wide`, `configs/Rsd_seq_wide.yaml`) but not trained.

## Drafter idea `seq_wide`: 2k pilots (D sweep × per-step / shared F)

The sizing check above only measured z_0; the working-space argument is about later steps (h_t plus the drafted
words), so the idea was also trained. Eight 2k-step pilots on the same 20k self-distilled texts and seed as
Rsd_seq_p2k (D = H, per-step F, i.e. Rsd_seq). Screen: chain drafting on IndicCorp (200 prompts), `--fast_linear`,
`bench_verify`. Δ tokens/step vs Rsd_seq_p2k by paired bootstrap (`compare_spec.py`). fp32 re-check 20/20 for all.

| pilot | D | F | head 1-3 top-1 (%) | draft ms | tok/step (Δ, 95% CI) | speed-up (Δ) |
|---|---|---|---|---|---|---|
| Rsd_seq_p2k (reference) | H | per step | 17.5 / 13.4 / 12.1 | 2.59 | 1.741 | ×1.588 |
| Rsd_wide1s_p2k | H | shared | 17.7 / 13.6 / 11.3 | 2.55 | 1.723 (−0.018 [−0.031, −0.006]) | ×1.594 (+0.006) |
| **Rsd_wide15_p2k** | 1.5H | per step | 18.7 / 14.4 / 12.6 | 2.92 | **1.785 (+0.044 [+0.024, +0.064])** | ×1.603 (+0.014) |
| Rsd_wide15s_p2k | 1.5H | shared | 18.9 / 14.5 / 11.9 | 2.95 | 1.761 (+0.020 [+0.002, +0.040]) | ×1.589 (+0.000) |
| Rsd_wide2_p2k | 2H | per step | 18.7 / 14.3 / 12.3 | 3.25 | 1.772 (+0.031 [+0.011, +0.051]) | ×1.570 (−0.018) |
| Rsd_wide2lin_p2k | 2H, linear U | per step | 17.5 / 13.5 / 11.4 | 3.23 | 1.741 (−0.000 [−0.020, +0.020]) | ×1.591 (+0.003) |
| Rsd_wide2s_p2k | 2H | shared | 19.0 / 14.2 / 11.7 | 3.18 | 1.766 (+0.025 [+0.007, +0.044]) | ×1.581 (−0.007) |
| Rsd_wide3_p2k | 3H | per step | 18.1 / 13.7 / 11.6 | 3.95 | 1.758 (+0.017 [−0.003, +0.038]) | ×1.527 (−0.062) |
| Rsd_wide3s_p2k | 3H | shared | 18.8 / 13.9 / 11.1 | 3.96 | 1.742 (+0.001 [−0.020, +0.023]) | ×1.543 (−0.045) |

- **Small, real acceptance gains, no speed-up:** the best pilot (1.5H, per-step F) gains +0.044 tokens/step but
  costs +0.33 ms per chain, so the speed-up moves +0.014: fails the keep gate (≥ +0.05×), like seq_ctx.
- **No knee: the gain shrinks as D grows** (1.5H +0.044, 2H +0.031, 3H +0.017 n.s.) while the cost grows. More room
  does not help, consistent with the sizing check.
- **The SiLU expansion is what helps:** at 2H, z_0 = [h; U h] (linear) gains nothing (−0.000), z_0 = [h; SiLU(U h)]
  gains +0.031. The benefit is the extra non-linear feature of h_t at step 0, not working space.
- **Shared F is a little worse than per-step F** at every D (and −0.018 at D = H).
- **Decision:** seq_wide dropped. Not sent to the full tree / tree + lookup eval: on seq_ctx, gains of this size gave
  no speed-up there either. The drafter stays Rsd_seq.

## fast_linear on Marathi + seed replicates (official eval)

`--fast_linear`, 200 prompts, Kaggle T4. Entries are stored as `<policy>+fast` in the runs' eval JSONs, timings in
`bench_verify_fast.json`. fp32 re-check 20/20 everywhere.

| run | chain | tree static n64 | tree static n64 + lookup | tok/s (IndicCorp) |
|---|---|---|---|---|
| Rsd_seq s43 (hi) | ×1.80 / ×1.76 | ×2.39 / ×2.30 | **×2.65 / ×2.73** | 92 (greedy 34.8) |
| Rsd_r8_seq (mr) | ×1.66 / ×1.61 | ×2.11 / ×2.07 | **×3.01 / ×2.94** | 80 (greedy 26.4) |
| Rsd_r8_seq s43 (mr) | ×1.65 / ×1.64 | ×2.09 / ×2.04 | **×2.99 / ×2.96** | 80 (greedy 26.8) |

(IndicCorp / FLORES.) The Hindi headline (seed 42: ×2.64 / ×2.70) replicates. Marathi with the same tree goes from
×2.33 / ×2.31 to ×3.01 / ×2.94: Misal's greedy pass gains little (43.6 → 38.8 ms) while its draft chain drops from 10.4
to 3.3 ms (47k-token LM head) and the 64-position verify pass from 45.8 to 36.9 ms. End to end, 80 tok/s vs the
original 22.9 tok/s greedy, ×3.5, identical output.

## Marathi word boundaries R13 / R13a, MorphTok check, drafter routing (Jainam, 2026-10-23)

Write-up: `docs/paper/analysis.md` §4-5. Official eval: Kaggle T4, `--grouper words`, IndicCorp mr eval + FLORES
mar_Deva, 200 prompts, fp32 re-check 20/20; GI also re-scored with `mr_rules_v1`.

| ID | config | Kaggle | what | IndicCorp / FLORES vs its twin |
|---|---|---|---|---|
| R13 / R13a | R13.yaml / R13a.yaml (= R10, grouper `words` / `random_tokens`) | kernel output `jainam2142/mtp-r13` | word-boundary groups vs a size-matched control that cuts inside words | tok/step −0.03* / −0.02*; GI (words) +2.9 / +2.3 n.s.; top-1 ±0.4 pts |
| R13_s43 / R13a_s43 | seed 43 + shuffle | kernel output `jainamrampuria/mtp-r13` | replicate | tok/step −0.02* / −0.01; GI +1.5 / +0.7 n.s. |

- `random_tokens` (`mtp/data/grouping/token_random.py`): tokens-per-word histogram of `words` under Misal
  (`histograms_tokens_words_mr.json`, 5,000 train sentences, mean 1.55), boundaries at random token positions; on the
  eval split its group starts hit word starts at chance (63.8% vs 63.3%). Uses the new optional
  `group_start_chars` hook (INTERFACES §2).
- Sampling acceptance by word position (`results/drafter/sampling_acceptance_mr_words.json`): continuing a word
  +1.2 to +1.6 pts, starting a word −1.3 to −1.7, overall −0.3 to −0.5 (head 1, all *).
- MorphTok (`results/grouping/morph_feasibility_{mr,hi}.json`): 10-13% of located morpheme boundaries fall on a
  Misal token boundary (Hindi/ganga 3-4%); 2.0-2.6 usable boundaries per 100 Marathi words. Morpheme-group training
  not run.
- Routing between frozen drafters (`results/drafter/route_drafters.json`, Rsd_struct / Rsd_struct_rand / Rsd):
  ceiling +0.005 to +0.008 tok/step with true boundaries, ≈ 0 with the S2 probe. Not pursued.

## Frozen pairs completed: SD-seq seed 43, SD-par re-decode, acceptance split (2026-10-25)

Kaggle `jainam2142/mtp-rq43` (T4 ×2): `Rsd_seq_struct_s43` / `Rsd_seq_struct_rand_s43` (= Rsd_seq_struct recipe, seed 43 +
data.shuffle), official eval (200 prompts, outputs stored, fp32 re-check 20/20); `Rsd_struct` re-decoded on T4 with outputs
stored (1.65 / 1.59 tokens/step, GI 77 / 82: reproduces the original eval); `results/drafter/sampling_acceptance_frozen.json`.

| pair (IndicCorp / FLORES) | Δ tokens/step | Δ GI rules | Δ GI Karthika et al. | Δ GI parser |
|---|---|---|---|---|
| Rsd_struct − random (parallel) | −0.01 / +0.00 | **+1.9\* / +1.4\*** | **+1.3\* / +1.5\*** | **+1.4\* / +1.6\*** |
| Rsd_seq_struct − random (seq) | −0.02\* / −0.00 | −0.0 / −0.0 | −0.5 / +0.1 | −0.1 / +0.1 |
| Rsd_seq_struct_s43 − random_s43 | +0.01 / −0.00 | +0.9 / +0.6 | +0.7 / −0.6 | +0.9 / +0.6 |

- **Seed 43 replicates the sequential null:** no speed change, no significant GI gap.
- **Both head types learn the word-class prior** (speculative-sampling acceptance, head 1, in-group / group-start): parallel
  +2.3\* / −0.9\*, sequential +1.4\* / −0.8\* (seed 43 +1.3\* / −0.7\*). Only parallel heads turn it into a GI gap.
- **Greedy top-1 agreement with the verifier** shows the same in/start trade in every pair (paper Table 14).
- The first two smoke kernels hung on `git clone` (repo is now private): code comes from the `mtp-code-r11` dataset, FLORES
  from `mtp-flores-hi`.

## Lossy tree decoding, sampled tree decoding, perplexity matrix (Jainam, 2026-11-13; `docs/plan/NEXT_PHASE.md`)

Kaggle T4 x2 (fp32 weights + fp16 autocast, `--fast_linear`), 200 prompts x 64 new tokens, the paper's prompts and
static n64 trees; fp32 re-check 10 prompts. Kernels `jainam2142/mtp-lossy-tree-a` (Hindi SD-seq) and
`jainam2142/mtp-lossy-tree-b` (Marathi SD-seq, Hindi SD-par). Outputs `results/lossy_tree/<run>/`: `eval_*.json`,
`quality_*.json` (self judge ganga-1b / Misal-1B, fp32), `quality_*.sarvam-1.json` (independent judge, text level,
per word). Full tables: `python scripts/lossy_table.py --results results/lossy_tree/<run> --judges sarvam-1 Misal-1B-instruct-v0.1`.
`typ<eps>` = Medusa typical acceptance at tree nodes; `samp<T>p<top_p>` = lossless speculative sampling (speed-up
and match vs plain sampling with the same seeds). `human` = rest of the source sentence (shorter: 15-41 tokens).

Speed-up vs greedy, IndicCorp / FLORES (sarvam-1 ppl per word, IndicCorp; rep4 IndicCorp):

| run | tree + lookup, exact | + typ0.5 | + typ0.02 | chain typical_0.02 | tree + lookup, T=0.7 p=0.9 | T=1.0 |
|---|---|---|---|---|---|---|
| Hindi SD-seq | x2.71 / x2.82 (4.62; 0.353) | x3.13 / x3.26 | x3.19 / x3.36 (4.70; 0.363) | x2.96 / x3.19 (7.05; 0.228) | x1.70 / x1.68 (14.3; 0.041) | x1.25 / x1.21 |
| Hindi SD-par | x2.54 / x2.63 (4.62; 0.353) | x2.99 / x3.03 | x3.07 / x3.09 (4.63; 0.401) | x2.05 / x2.18 (7.67; 0.199) | x1.66 / x1.61 (14.2; 0.049) | x1.17 / x1.16 |
| Marathi SD-seq | x3.05 / x3.00 (4.93; 0.623) | x3.49 / x3.54 | x3.36 / x3.49 (6.03; 0.613) | x2.56 / x2.68 (10.7; 0.460) | x1.51 / x1.46 (51.7; 0.103) | x1.07 / x1.02 |

Reference rows, sarvam-1 ppl per word IndicCorp / FLORES: Hindi human 33.6 / 49.6, Hindi greedy 4.62 / 4.18;
Marathi human 174 / 289, greedy 4.93 / 6.07. Self judge, per token: Hindi human 14.4 / 33.9, greedy 2.64 / 2.81.

- Exact and sampled modes are correct: fp32 re-check 10/10 for every exact and T=0.7 entry (T=1.0 FLORES Hindi
  8/10, float noise). fp16 match vs plain sampling 0.53 (T=0.7) and ~0.01 (T=1.0): batched verify drifts the
  sequence numerically; the method is exact in distribution.
- Lossy at tree nodes adds +0.4 to +0.7 tok/step on top of the exact tree + lookup (x2.5-3.0 -> x3.1-3.6).
  10-20% of accepted drafts are lossy. With lookup the judged cost is small (sarvam ppl/word +0.0 to +1.1, rep4
  unchanged); lookup copies the prompt, which keeps outputs close to greedy. Without lookup lossy lowers rep4.
- Chain typical acceptance (the older lossy setting) is slower than lossy trees at the same eps and costs more
  quality (sarvam ppl/word 7-11 vs 4.6-6); only on Hindi SD-seq does it beat the exact tree + lookup.
- **Sampling kills most of the speed-up (KT-1):** x1.5-1.7 at T=0.7, x1.0-1.25 at T=1.0, and prompt lookup adds
  nothing. The x2.6-3.0 is a greedy-decoding result.
- **Perplexity is not a quality metric across decoding modes here:** greedy text (rep4 0.35-0.62) gets ppl 4-5 per
  word under sarvam-1, human text 34-290, T=1 samples 150-18000. Repetition and low entropy lower ppl. It is
  usable only within one mode (exact vs lossy at the same decoding), next to rep4. Misal-1B-instruct as a judge
  gives Marathi human text ppl/word 591 / 1896 (weak on raw text), so Marathi self-judge numbers are only relative.

### Temperature sweep (kernel `jainam2142/mtp-temp-sweep`, IndicCorp, top-p 1.0; `results/temp_sweep/<run>/`)

Tree n64 + lookup, speed-up vs greedy (sarvam-1 ppl per word):

| run | greedy | T=0.1 | T=0.3 | T=0.5 | T=0.7 |
|---|---|---|---|---|---|
| Hindi SD-seq | x2.67 (4.62) | x2.58 (4.79) | x2.27 (6.27) | x1.87 (10.5) | x1.62 (19.4) |
| Marathi SD-seq | x3.16 (4.93) | x2.97 (5.20) | x2.46 (9.42) | x1.88 (25.0) | x1.37 (138) |

fp32 re-check 10/10 for every entry. The speed-up decays smoothly with temperature: about x2.3-2.5 survives at
T=0.3, half of the gain is gone by T=0.5. Lookup helps only up to T~0.3.

## Sampling-aware trees (2026-11-14/15; jainam2142 `mtp-samp-trees-hi` / `-mr`, T4, `--fast_linear`)

`fit_tree.py --sample T,TOP_P` on 300 IndicCorp-train calib / 100 eval prompts; eval 200 x 64, fixed_k + greedy-fitted
n64 + n64_fitT0.7p0.9 + n64_fitT1p1, each also with `--tree_sample 0.7,0.9 1.0,1.0`. All fit/eval exits 0. Results:
`results/samp_trees/<run>/` (laptop fits with T=0.3/0.5 too: `results/samp_trees_laptop/`).

Offline (accepted drafts/step on sampled eval text, n64): greedy fit / sampled fit / ceiling (drafter = verifier):
Hindi T0.7 1.37 / 1.39 / 2.44 (greedy text 2.13); T1 0.73 / 0.79 / 1.49. Marathi T0.7 0.99 / 1.03 / 2.22
(greedy text 1.84); T1 0.36 / 0.42 / 1.00. Tree shape is not the bottleneck; sampling itself is.

T4, sampled tok/step greedy-fit -> best sampled fit (paired bootstrap, 200 prompts): +0.03..+0.07, CI excludes 0 in 15/16
(not: Hindi FLORES fitT1 at T0.7, +0.023 [-0.001, +0.047]); speed +0.5..5%. In greedy decoding the sampled fits lose
up to 0.06 tok/step. fp32: 10/10 greedy and T0.7; T1 7-10/10 (CDF-boundary float flips).

## Qwen word groups (K3; jainamrampuria `mtp-qwen-groups`)

Qwen SD-seq + structural losses, hi_rules_v0 vs random groups (6.15 / 6.22 tok/group), 3,000 steps, 100 prompts x 256.
Linguistic - random tok/step: IndicCorp chain +0.005 [-0.003, +0.013], tree +0.005 [-0.002, +0.013], tree+lookup -0.008
[-0.032, +0.013]; FLORES -0.002 / -0.004 / +0.004 (all n.s.). fp32 10/10. Null also with long groups.

## Lookup-only baseline + decoder fix (2026-11-15; jainam2142 `mtp-pld-only`, T4, `--fast_linear`)

Fix: `tree_decode._static` did not unwrap `PromptLookup`, so seq-head lookup variants ran `draft_chain` every step and
threw it away (outputs unchanged). Re-timed after the fix, one session, 200 x 64, fp32 10/10 on every exact entry
(`results/pld_only/`). tok/step, speed-up, IndicCorp / FLORES:
- Hindi SD-seq: chain 2.00 x1.82 / 1.89 x1.73; lookup only (`tree_static_n0+pld4x6`) 1.53 x1.49 / 1.69 x1.68;
  tree 2.89 x2.38 / 2.78 x2.28; tree+lookup 3.49 x2.85 / 3.63 x2.97 (was x2.71 / x2.82 before the fix);
  tree+lookup+typ0.5 4.08 x3.23 / 4.30 x3.48.
- Marathi SD-seq: chain 1.81 x1.67 / 1.77 x1.64; lookup only 2.36 x2.28 / 2.31 x2.30; tree 2.48 x2.12 / 2.44 x2.08;
  tree+lookup 3.96 x3.26 / 3.87 x3.23 (was x3.05 / x3.00); tree+lookup+typ0.5 4.55 x3.73 / 4.59 x3.82.
Lookup rows of the earlier lossy / temperature / sampling-tree sessions understate speed by ~5-7%.

## Repetition-penalty probe (2026-11-15, laptop, offline; `scripts/rep_probe.py`)

Hindi SD-seq, 100 IndicCorp eval prompts x 64, teacher-forced drafts accepted per step with the greedy-fitted n64 tree
(raw head ranks / with the rule applied to the head logits), sarvam-1 ppl/word: greedy rep4 .379, 2.15, 5.2;
rp1.1 .071, 1.84 / 1.85, 10.4; rp1.2 .015, 1.54 / 1.63, 16.4; no-repeat-4gram 0, 1.93, 9.6 (still loops
semantically). rp1.1 text reads naturally. Exact tree mode added: `mtp/eval/penalty.py`, `evaluate.py --tree_penalty`.

## External baseline: Hugging Face generate (T4, 2026-11-15)
Kernel `jainam2142/mtp-hf-base`, one session, IndicCorp, 200 x 64, SD-seq. Results in `results/hf_baseline/`.
tokens/s Hindi / Marathi: HF greedy autocast 28.4 / 24.6, HF greedy fp16 44.7 / 34.1, HF prompt lookup (10) autocast 42.7 / 57.9, fp16 63.9 / 77.4;
ours greedy 33.5 / 25.1, ours tree+lookup 95.3 / 83.4 (x2.84 / x3.32, fp32 10/10).
Ours vs fastest HF: x1.49 Hindi, x1.08 Marathi (noise). HF fp16 greedy is ~1/3 faster than our greedy baseline.

## Repetition-penalised tree decoding (T4, 2026-11-15/16; jainam2142 `mtp-rp-trees`, `--fast_linear`)
`--tree_penalty 1.1 1.2`: CTRL penalty on head 0, tree output identical to penalised greedy (fp32 10/10 in all 16 entries).
200 x 64, SD-seq, static n64. Results in `results/rp_trees/`. IndicCorp / FLORES; speed-up over penalised greedy; judge = sarvam-1 ppl/word.

| | Hindi tok/step, speed-up | rep-4 | judge | Marathi tok/step, speed-up | rep-4 | judge |
|---|---|---|---|---|---|---|
| greedy, tree + lookup | 3.49 x2.85 / 3.63 x2.87 | .35 / .42 | 4.6 / 4.3 | 3.96 x3.32 / 3.87 x3.19 | .62 / .62 | 4.9 / 6.1 |
| r = 1.1, tree | 2.56 x2.14 / 2.59 x2.20 | .07 / .10 | 8.6 / 8.5 | 2.20 x1.87 / 2.15 x1.82 | .11 / .10 | 21.5 / 29.0 |
| r = 1.1, tree + lookup | 2.66 x2.19 / 2.71 x2.25 | .07 / .10 | 8.5 / 8.5 | 2.33 x1.97 / 2.27 x1.93 | .11 / .10 | 21.6 / 29.0 |
| r = 1.2, tree | 2.29 x1.90 / 2.33 x1.95 | .01 / .01 | 13.8 / 14.1 | 1.88 x1.63 / 1.86 x1.59 | .00 / .00 | 43.3 / 53.2 |
| human continuation | | .00 | 33.6 / 49.6 | | .01 / .00 | 174 / 289 |

- Removing most repetition (r = 1.1) keeps x2.1-2.25 for Hindi and x1.8-2.0 for Marathi: more than sampling at T = 0.7 (x1.4-1.8), and deterministic.
- Lookup adds little once the output stops repeating (+0.1 tok/step at r = 1.1, nothing at 1.2).

## Cost of the speed-up: tree sizes, GPU power, plain fp16 weights (T4, 2026-11-16; jainam2142 `mtp-cost`)
One session, IndicCorp, 200 x 64, SD-seq, `--fast_linear`; `nvidia-smi` power every 0.5 s, aligned to each policy's
timed loop (`wall` / `greedy_wall`, `scripts/cost_table.py`). Board power of the GPU only. Results in `results/cost_curve/`.
pos/tok = verified positions per emitted token; J/tok = mean watts / tokens per second.

| Hindi | tok/step | speed-up | pos/tok | W | J/tok |
|---|---|---|---|---|---|
| greedy | 1.00 | x1.00 | 1.0 | 54.8 | 1.62 |
| chain | 2.00 | x1.87 | 2.0 | 66.6 | 1.05 |
| tree n25 / n40 / n64 | 2.68 / 2.79 / 2.89 | x2.36 / 2.43 / 2.46 | 9.4 / 14.3 / 21.7 | 67.7 / 67.8 / 67.0 | 0.85 / 0.83 / 0.81 |
| tree n25 / n40 / n64 + lookup | 3.25 / 3.36 / 3.49 | x2.79 / 2.88 / 2.92 | 8.2 / 12.3 / 18.6 | 67.9 / 67.8 / 67.1 | 0.72 / 0.70 / 0.68 |

| Marathi | tok/step | speed-up | pos/tok | W | J/tok |
|---|---|---|---|---|---|
| greedy | 1.00 | x1.00 | 1.0 | 41.1 | 1.64 |
| chain | 1.81 | x1.70 | 2.2 | 58.1 | 1.36 |
| tree n40 / n64 | 2.41 / 2.48 | x2.13 / 2.13 | 16.5 / 25.5 | 65.6 / 67.4 | 1.23 / 1.26 |
| tree n40 / n64 + lookup | 3.87 / 3.96 | x3.28 / 3.27 | 11.0 / 16.7 | 65.3 / 67.2 | 0.79 / 0.82 |

- The full stack verifies 17-19 positions per emitted token, but the GPU draws 67 W instead of 41-55 W for a third of the
  time: energy per token falls to x0.42 (Hindi) / x0.50 (Marathi) of greedy. fp32 10/10 everywhere.
- Tree size is nearly flat in speed: n25 gives x2.79 with lookup vs x2.92 at n64, at less than half the positions.
- `--half` (plain fp16 weights, no autocast): greedy 37.1 / 27.2 tok/s, full stack 105.0 / 91.3 tok/s (x2.83 / x3.36):
  +6% / +11% tokens per second over autocast + fast_linear, same ratios. HF fp16 greedy (44.7 / 34.1) is still faster than ours.

## Word-completion lexicon branches (Qwen2.5-1.5B, Hindi, T4)

Kernel `jainam2142/mtp-lex`, frozen SD-seq heads on Qwen2.5-1.5B (4.87 tokens per Hindi word), 100 prompts x 256
tokens, exact decoding (fp32 check 1.0 for every policy). Lexicon: trie of word-prefix tokens -> most frequent
completions from IndicCorp train (rows >= 1000), 4 root chains, 2 completions under each of the 12 most likely head
nodes (`+lex4e2x12`). Results in `results/lexicon/`.

| Policy | IndicCorp tok/step | speed-up | FLORES tok/step | speed-up | nodes |
|---|---|---|---|---|---|
| chain | 2.98 | x2.64 | 2.89 | x2.57 | 3 |
| tree n64 | 3.73 | x3.04 | 3.64 | x3.05 | 63 |
| tree n64 + lexicon | 4.55 | x3.67 | 4.44 | x3.59 | 98 |
| tree n64 + lookup | 5.67 | x4.66 | 5.70 | x4.68 | 68 |
| tree n64 + lookup + lexicon | 6.02 | x4.81 | 6.05 | x4.88 | 101 |

The lexicon adds +0.8 tok/step (x3.04 -> x3.67) to the tree alone and +0.35 (x4.66 -> x4.81) on top of prompt
lookup, which already covers repeated words. It needs no training and no model call. Not useful for ganga-1b
(1.12 tokens per word).
