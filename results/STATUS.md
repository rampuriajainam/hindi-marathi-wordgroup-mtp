# Evaluation status (OM-8)

Which runs are evaluated on which datasets. ⏳ = running, – = dropped (group-aware decoding stopped, #33).

| run | step | indiccorp_eval | flores_hi | spec: fixed_k | spec: confidence_cut | spec: group_aware | notes |
|---|---|---|---|---|---|---|---|
| R0 | 12500 | ✅ | ✅ | – | – | – | NTP, 1 head (no drafts) |
| R1 | 12500 | ✅ | ✅ | ✅ | ✅ | – | spec: fp16 81-89% identical (near-ties), fp32 re-check 20/20 |
| R2 | 12500 | ✅ | ✅ | ✅ | ✅ | – | spec: fp32 re-check 20/20 except indiccorp fixed_k 19/20 = exact tie (margin 0.0); `bench_verify` ✅ |
| R3 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R3`; fp32 re-check 20/20 everywhere; in-group h1-h3 up, acceptance unchanged vs R2 |
| Rsd | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-rsd`; frozen backbone (head 0 = base model), self-distillation; FixedK ×1.32-1.36, fp32 20/20; `bench_verify` ✅ |
| Rsd_s43 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-rsd-s43`; replicate of Rsd (within 0.02 tokens/step); fp32 20/20; `bench_verify` ✅ |
| R6a | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R6a`; `--grouper hi_rules_v0`; h1 in-group ≈ R3 (25.2 vs 25.8), h2/h3 between R2 and R3; fp32 20/20 |
| A1_alpha0 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainamrampuria/mtp-run-A1-alpha0`; R2 with α = 0 (JN-10) |
| A1_alpha1 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainamrampuria/mtp-run-A1-alpha1`; R2 with α = 1 (JN-10) |
| R8 (mr) | 12500 | ✅ | flores_mr ✅ | – | – | – | `jainam2142/mtp-run-R8`; Misal-1B NTP, `--grouper mr_rules_v1` |
| R9 (mr) | 12500 | ✅ | flores_mr ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R9`; Misal-1B k=4 resblock, `--grouper mr_rules_v1` |
| Rsd_soft | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-soft`; Rsd + soft-label KL; = Rsd (×1.37 / ×1.31); `bench_verify` ✅ |
| Rsd_k6 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-k6`; 6 heads; 1.71 tokens/step but ×1.28 (heads cost ~2 ms each); `bench_verify` ✅ |
| Rsd_seq | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-seq`; sequential heads; **2.00 / 1.89 tokens/step, ×1.61 / ×1.52**; `bench_verify` ✅ (draft 7.0 ms) |
| Rsd_long | 25000 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-long`; 2× data + steps; ×1.41 / ×1.36; `bench_verify` ✅ |
| Rsd_struct | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-struct`; + structural loss; = Rsd speed, GI +3.1 / +1.7; `bench_verify` ✅ |
| Rsd_struct_rand | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-struct-rand`; random control of Rsd_struct; GI Rsd_struct − rand +1.9 / +1.4 (CIs exclude 0) |
| Rsd_seq_s43 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-seq-s43`; ×1.60 / ×1.54 (replicates Rsd_seq); `bench_verify` ✅ |
| Rsd_seq_k6 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-seq-k6`; 2.09 tokens/step but ×1.53 / ×1.44 (draft 11.7 ms); `bench_verify` ✅ |
| Rsd_r8 (mr) | 12500 | ✅ | flores_mr ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-r8`; head 0 = R8; ×1.37 / ×1.36; `ignore_eos on` |
| Rsd_r8_seq (mr) | 12500 | ✅ | flores_mr ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-r8-seq`; **×1.44 / ×1.45** (best Marathi); `bench_verify` ✅ |
| Rsd_r8_seq_s43 (mr) | 12500 | ✅ | flores_mr ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-r8-seq-s43`; ×1.44 / ×1.46 (replicates) |
| Rsd_r8_seq_k6 (mr) | 12500 | ✅ | flores_mr ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-r8-seq-k6`; 1.84 tokens/step but ×1.29 / ×1.31 (draft 17.3 ms) |
| R3_s43 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R3-s43`; second seed of R3; in-group h1 23.6 (below R6a_s43); `bench_verify` ✅ |
| R6a_s43 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R6a-s43`; `--grouper hi_rules_v0`; R3 > R6a gap does not replicate; `bench_verify` ✅ |
| R10 (mr) | 12500 | ✅ | flores_mr ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R10`; R3 recipe on Misal, `mr_rules_v1`; in-group = R10a |
| R10a (mr) | 12500 | ✅ | flores_mr ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R10a`; random_mr_v1 control, `--grouper mr_rules_v1` |
| Rsd_mr (mr) | 12500 | ✅ | flores_mr ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-mr`; `ignore_eos`; FixedK ×1.25 / ×1.20 |
| Rsd_mr_soft (mr) | 12500 | ✅ | flores_mr ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-mr-soft`; `ignore_eos`; = Rsd_mr |
| R7s | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R7s`; stanza-parser groups; GI 79.2 / 82.4 (R6a 74.9 / 79.5) |
| R7s_s43 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R7s-s43`; GI 79.7 / 82.0 (R6a_s43 77.1 / 81.1) |
| R4 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R4`; R3 + contrastive; GI 79.4 / 83.8 |
| R4a | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R4a`; random control of R4; GI 76.9 / 81.3 |
| R10_s43 (mr) | 12500 | ✅ | flores_mr ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R10-s43`; in-group = R10a_s43; GI 81.0 / 78.6 |
| R10a_s43 (mr) | 12500 | ✅ | flores_mr ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R10a-s43`; GI 79.9 / 76.5 |
| Rsd_seq_struct | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-seq-struct`; ×1.60 / ×1.52; GI = random control |
| Rsd_seq_struct_random | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-Rsd-seq-struct-rand`; ×1.63 / ×1.53 |

Sanity flags: ConfidenceCut speed-up < 1 for R2 (few drafts proposed); R2 indiccorp fixed_k fp32 re-check 19/20 resolved (exact tie). Head 0 of R1/R2 within 0.031 / 0.006 of R0 on `indiccorp_eval`.
