### Table 4: draft policies and tree decoding (Kaggle T4)

| model | draft policy | nodes / step | IndicCorp tok/step | speed-up | FLORES tok/step | speed-up |
|---|---|---|---|---|---|---|
| R2 | fixed k | 3 | 1.39 | ×1.15 | 1.36 | ×1.12 |
| R2 | confidence cut | ≤3 | 1.10 | ×0.94 | 1.07 | ×0.91 |
| Rsd | fixed k | 3 | 1.67 | ×1.36 | 1.60 | ×1.32 |
| Rsd | confidence cut | ≤3 | 1.36 | ×1.14 | 1.30 | ×1.09 |
| Rsd | tree entropy n40 | 39 | 2.42 | ×1.74 | 2.33 | ×1.68 |
| Rsd | tree static n64 | 62 | 2.50 | ×1.73 | 2.39 | ×1.68 |
| Rsd | fixed k+fast | 3 | 1.67 | ×1.56 | 1.60 | ×1.49 |
| Rsd | tree static n64+fast | 62 | 2.50 | ×2.13 | 2.39 | ×2.02 |
| Rsd | tree static n64+pld4x6+fast | 64 | 3.10 | ×2.59 | 3.20 | ×2.64 |
| Rsd | tree entropy n40+pld4x6 | 40 | 3.01 | ×2.12 | 3.14 | ×2.23 |
| Rsd | tree static n64+pld4x6 | 64 | 3.10 | ×2.11 | 3.20 | ×2.20 |
| Rsd_k6 | fixed k | 5 | 1.71 | ×1.28 | 1.63 | ×1.21 |
| Rsd_k6 | tree entropy n40 | 39 | 2.45 | ×1.56 | 2.34 | ×1.50 |
| Rsd_seq | fixed k | 3 | 2.00 | ×1.61 | 1.89 | ×1.52 |
| Rsd_seq | confidence cut | ≤3 | 1.60 | ×1.32 | 1.51 | ×1.23 |
| Rsd_seq | tree entropy n40 | 39 | 2.82 | ×1.79 | 2.70 | ×1.71 |
| Rsd_seq | tree static n40 | 39 | 2.79 | ×2.02 | 2.67 | ×1.94 |
| Rsd_seq | tree static n64 | 62 | 2.89 | ×2.01 | 2.78 | ×1.92 |
| Rsd_seq | fixed k+fast | 3 | 2.00 | ×1.82 | 1.89 | ×1.71 |
| Rsd_seq | tree static n64+fast | 62 | 2.89 | ×2.39 | 2.78 | ×2.27 |
| Rsd_seq | tree static n64+pld4x6+fast | 64 | 3.49 | ×2.64 | 3.63 | ×2.70 |
| Rsd_seq | tree static n64+pld4x6 | 64 | 3.49 | ×2.11 | 3.63 | ×2.19 |
| Rsd_seq | tree entropy n40+pld4x6 | 40 | 3.40 | ×2.12 | 3.56 | ×2.22 |
| Rsd_seq | tree static n40+pld4x6 | 40 | 3.36 | ×2.11 | 3.53 | ×2.22 |
| Rsd_r8_seq | fixed k | 3 | 1.81 | ×1.44 | 1.77 | ×1.44 |
| Rsd_r8_seq | tree static n40 | 39 | 2.41 | ×1.77 | 2.36 | ×1.75 |
| Rsd_r8_seq | tree static n64 | 62 | 2.48 | ×1.74 | 2.44 | ×1.73 |
| Rsd_r8_seq | fixed k+fast | 3 | 1.80 | ×1.66 | 1.77 | ×1.61 |
| Rsd_r8_seq | tree static n64+fast | 62 | 2.47 | ×2.11 | 2.43 | ×2.07 |
| Rsd_r8_seq | tree static n64+pld4x6+fast | 65 | 3.95 | ×3.01 | 3.88 | ×2.94 |
| Rsd_r8_seq | tree static n40+pld4x6 | 42 | 3.87 | ×2.35 | 3.77 | ×2.33 |
| Rsd_r8_seq | tree entropy n40+pld4x6 | 41 | 3.88 | ×2.35 | 3.81 | ×2.34 |
| Rsd_r8_seq | tree static n64+pld4x6 | 65 | 3.96 | ×2.33 | 3.87 | ×2.31 |

- Greedy decoding = 1 token/step = ×1.00 (27.4-27.7 tok/s). All policies produce output identical to greedy (fp32 re-check 20/20).
- Trees: Medusa sparse trees (static) or one tree per head-entropy cell (entropy), fitted on IndicCorp train prompts; n = node budget. +pld4x6 = up to 4 prompt-lookup chains of 6 tokens copied from the context, grafted onto the tree.
- Rsd_k6's tree row was timed with the engine before its per-step overhead fix (a lower bound); all other tree rows use the final engine.
- +fast = linear layers compute from cached fp16 weights (drafter DR-2, bit-identical outputs); the greedy baseline of those rows uses it too (36.4-36.6 tok/s for Rsd_seq, 35.2-35.3 for Rsd, vs 27.4-27.7 without; 26.4-26.6 for Marathi Rsd_r8_seq, vs 22.9 without).
