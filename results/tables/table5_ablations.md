### Table 5: ablations

| run | change | h0 loss | IndicCorp tok/step (speed-up) | FLORES |
|---|---|---|---|---|
| A1_alpha0 | α = 0 | 2.834 | 1.38 (×1.14) | 1.32 (×1.09) |
| R2 | α = 0.1 (default) | 2.841 | 1.39 (×1.15) | 1.36 (×1.12) |
| A1_alpha1 | α = 1 | 2.981 | 1.43 (×1.18) | 1.40 (×1.16) |
| Rsd | frozen + self-distill | 2.865 | 1.67 (×1.36) | 1.60 (×1.32) |
| Rsd_s43 | Rsd, seed 43 | 2.865 | 1.66 (×1.35) | 1.62 (×1.33) |
| Rsd_soft | soft labels | 2.865 | 1.67 (×1.37) | 1.60 (×1.31) |
| Rsd_k6 | 6 heads | 2.865 | 1.71 (×1.28) | 1.63 (×1.21) |
| Rsd_long | 2× data and steps | 2.865 | 1.72 (×1.41) | 1.66 (×1.36) |
| Rsd_seq | sequential heads | 2.865 | 2.00 (×1.61) | 1.89 (×1.52) |
| Rsd_seq_s43 | seq, seed 43 | 2.865 | 1.99 (×1.60) | 1.91 (×1.54) |
| Rsd_seq_k6 | seq, 6 heads | 2.865 | 2.09 (×1.53) | 1.97 (×1.44) |

- α = share of the extra heads' gradient that reaches the shared LoRA (head_backbone_grad). Frozen runs keep head 0 = ganga-1b exactly (same h0 loss).
