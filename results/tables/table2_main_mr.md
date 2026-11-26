### Table 2: Marathi (Misal-1B)

| run | setup | h0 ppl | top-1 h1 | h2 | h3 | in-group h1 | h2 | h3 | FixedK tok/step (speed-up), IndicCorp | FLORES |
|---|---|---|---|---|---|---|---|---|---|---|
| R8 | NTP | 70.2 | – | – | – | – | – | – | – | – |
| R9 | 3 resblock heads | 70.7 | 11.5 | 7.6 | 6.0 | 28.3 | 17.4 | 11.7 | 1.31 (×1.07) | 1.30 (×1.05) |
| R10 | + group loss (rules) | 72.8 | 12.0 | 7.8 | 6.0 | 30.5 | 18.6 | 13.3 | 1.34 (×1.10) | 1.31 (×1.07) |
| R10a | + group loss (random) | 72.8 | 12.0 | 7.8 | 6.0 | 30.9 | 18.8 | 12.6 | 1.35 (×1.08) | 1.30 (×1.05) |
| Rsd_mr | frozen instruct + self-distill | 133.0 | 9.3 | 5.7 | 4.3 | 22.2 | 11.7 | 8.1 | 1.53 (×1.25) | 1.46 (×1.20) |
| Rsd_r8 | frozen R8 + self-distill | 70.2 | 9.8 | 6.2 | 4.4 | 21.9 | 11.6 | 5.9 | 1.71 (×1.37) | 1.68 (×1.36) |
| Rsd_r8_seq | Rsd_r8 + sequential heads | 70.2 | 11.0 | 8.6 | 8.3 | 23.9 | 14.2 | 9.9 | 1.81 (×1.44) | 1.77 (×1.44) |

- Per-head columns on IndicCorp eval; top-1 and in-group top-1 in %. In-group = the target token continues the source token's word group (grouper `hi_rules_v0` / `mr_rules_v1`).
- Kaggle T4, step 12500 (Rsd_long 25000), 200 prompts × 64 new tokens; outputs identical to greedy (fp32 re-check 20/20).
