### Table 3: linguistic vs random word groups

| linguistic − random | in-group h1 | h2 | h3 | GI rules, IndicCorp | FLORES | GI stanza, IndicCorp | FLORES | Δ tok/step |
|---|---|---|---|---|---|---|---|---|
| R3 − R6a | +0.6 | +2.5 | +2.3 | +2.9 [+1.0, +4.8] ✓ | +2.8 [+1.3, +4.3] ✓ | +2.0 [-0.1, +4.1] | +0.9 [-0.8, +2.5] | -0.00 |
| R3_s43 − R6a_s43 | -1.5 | -0.2 | +1.8 | +2.1 [-0.3, +4.4] | +0.7 [-1.6, +2.9] | +1.5 [-0.8, +3.9] | -0.2 [-2.7, +2.3] | -0.03 |
| R7s − R6a | +0.3 | +1.3 | +0.0 | +4.3 [+2.4, +6.4] ✓ | +2.9 [+1.4, +4.3] ✓ | +4.1 [+2.1, +6.3] ✓ | +1.7 [-0.1, +3.3] | -0.01 |
| R7s_s43 − R6a_s43 | -0.3 | +1.8 | +0.0 | +2.6 [+0.9, +4.3] ✓ | +0.9 [-1.0, +2.8] | +3.5 [+1.8, +5.3] ✓ | +1.1 [-1.1, +3.3] | -0.02 |
| R4 − R4a | +0.8 | +3.2 | +3.5 | +2.5 [+0.2, +4.7] ✓ | +2.5 [+0.3, +4.7] ✓ | +0.5 [-1.8, +2.8] | +1.4 [-0.9, +3.7] | -0.01 |
| R10 − R10a | -0.4 | -0.2 | +0.7 | +0.3 | +1.6 | – | – | -0.01 |
| R10_s43 − R10a_s43 | -0.2 | +0.0 | +0.0 | +1.1 [-1.6, +3.6] | +2.2 [-0.6, +4.7] | – | – | -0.00 |
| Rsd_struct − Rsd_struct_rand | +1.2 | +2.5 | +2.3 | +1.9 [+0.8, +2.9] ✓ | +1.4 [+0.4, +2.4] ✓ | – | – | -0.01 |
| Rsd_seq_struct − Rsd_seq_struct_rand | +1.1 | +1.1 | +2.3 | -0.0 [-1.1, +1.0] | -0.0 [-0.9, +0.8] | – | – | -0.02 |

- In-group top-1 difference in points (IndicCorp eval). Group Integrity (GI) = share of accepted multi-token spans that end on a word-group boundary, FixedK, 200 prompts; scored with the rules grouper (`hi_rules_v0` / `mr_rules_v1`) and re-scored with the stanza dependency parser.
- Brackets: paired bootstrap 95% CI over prompts (2,000 resamples); ✓ = excludes 0. Plain values: runs evaluated before per-prompt outputs were stored (no CI).
- R3 / R4 / R10 are trained on the rules grouping, R7s on the stanza parser grouping; R7s scored with the rules is scored on a grouping it never saw.
