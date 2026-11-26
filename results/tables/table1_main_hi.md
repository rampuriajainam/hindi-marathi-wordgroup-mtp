### Table 1: Hindi (ganga-1b)

| run | setup | h0 ppl | top-1 h1 | h2 | h3 | in-group h1 | h2 | h3 | FixedK tok/step (speed-up), IndicCorp | FLORES |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | NTP | 17.0 | – | – | – | – | – | – | – | – |
| R1 | 1 linear head | 17.6 | 15.4 | – | – | 28.7 | – | – | 1.31 (×1.25) | 1.28 (×1.23) |
| R2 | 3 resblock heads | 17.1 | 15.0 | 8.1 | 5.3 | 23.4 | 19.0 | 17.5 | 1.39 (×1.15) | 1.36 (×1.12) |
| R3 | + group loss (rules) | 17.3 | 15.5 | 8.4 | 5.5 | 25.8 | 24.1 | 22.2 | 1.39 (×1.15) | 1.35 (×1.11) |
| R4 | + group loss + contrastive | 17.3 | 15.5 | 8.4 | 5.6 | 26.1 | 24.1 | 24.0 | 1.40 (×1.14) | 1.36 (×1.13) |
| R7s | + group loss (parser) | 17.3 | 15.6 | 8.3 | 5.4 | 25.5 | 22.8 | 19.9 | 1.39 (×1.14) | 1.36 (×1.12) |
| Rsd | frozen + self-distill | 17.6 | 13.9 | 7.1 | 4.5 | 19.9 | 13.1 | 13.5 | 1.67 (×1.36) | 1.60 (×1.32) |
| Rsd_seq | Rsd + sequential heads | 17.6 | 21.2 | 17.3 | 15.8 | 23.6 | 20.3 | 24.6 | 2.00 (×1.61) | 1.89 (×1.52) |

- Per-head columns on IndicCorp eval; top-1 and in-group top-1 in %. In-group = the target token continues the source token's word group (grouper `hi_rules_v0` / `mr_rules_v1`).
- Kaggle T4, step 12500 (Rsd_long 25000), 200 prompts × 64 new tokens; outputs identical to greedy (fp32 re-check 20/20).
