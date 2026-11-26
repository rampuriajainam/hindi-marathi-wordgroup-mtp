# JI-6a: Hindi rules cache handoff

Local validation: **2026-10-19**, all rows checked. Evidence:
`results/grouping/boundary_cache_validation.json`. No model was run for validation.

| Field | Verified value |
|---|---|
| Grouper | `hi_rules_v1` |
| Language | `hi` |
| Tokenizer | `LingoIITGN/ganga-1b` |
| Maximum length | 128 |
| Train rows | 200,000 |
| Eval rows | 500 |
| FLORES rows | 1,012 |
| Local root | `boundary_cache/` |
| Kaggle URL / version | **Unverified; add the actual published URL and version** |
| Attached Kaggle root | **Determine from the attached dataset; do not assume a path** |

Each split is a folder `hi_hi_rules_v1_{train,eval,flores}` containing Arrow data,
dataset descriptors and its own `meta.json`. Validation checks all rows' label
lengths and boundary/group consistency in addition to metadata/counts, and saves
text hashes. Labels are rule-generated; Hindi human-gold evaluation remains pending.

No upload or dataset publication was performed here. The earlier placeholders
did not establish that an uploaded copy had been checked. Revalidate the actual
attached copy using `scripts/validate_boundary_cache.py` before training.
See [JAI_NEXT_STEPS.md](JAI_NEXT_STEPS.md) for the command and later model runs.
