# Jai completion handoff

## Latest review confirmation — 2026-10-22

Jai has now confirmed reviewing the annotations and finding them correct. The
active 200 Hindi and 150 Marathi rows are project gold: `human_reviewed`,
`reviewed_by: "jai"`, `dataset_role: "gold"`, `review_required: false`. The
annotation contents are unchanged and `annotator: "ai_draft"` retains origin.
This is reviewer self-attestation, not an independently observed review.
See [the review record](../results/grouping/annotation_gold_review_2026-10-22.json).
This supersedes the provisional status below. Om's independent 50 annotations,
agreement and final grouper scoring remain pending. No model runs were launched.

## Current annotation update — 2026-10-22

All 200 Hindi and 150 Marathi sentences have fresh-context AI annotations and a
separate AI review against the project annotation guide. Jai accepted eight random
examples and authorized publication, explicitly requiring another review. Active
`data/gold/{hi,mr}_gold.jsonl` rows have `annotation_status: "approved"`,
`review_required: true`, and the original `annotator: "ai_draft"` provenance.
They remain provisional AI annotations, rather than completed human gold.
See [the annotation provenance/readme](../data/gold/README.md) and
[the sampled examples and approval](../results/grouping/annotation_review_sample_2026-10-22.json).

Next: review the full sets and adjudicate the noted uncertainties; obtain Om's
independent 50 annotations; then produce human-gold scoring and agreement. Om's
packet and historical human files are unchanged. Previous active annotation
files are backed up under `data/gold/archive/2026-10-22_pre_ai_redo/`.
The sanity check passed 148 offline data/tool tests; source identity, partition,
label, and protected-file checks also passed. Local production-model runs remain
deferred to Jainam. Cache publication, C5, token-dump error analysis and final
paper results below remain pending.

The annotation-only follow-up is prepared on
`jai/hi-mr-annotations-review-20261022` for a draft PR against `main`.

## Historical handoff — 2026-10-19

Work branch: `jai/finish-data-linguistics`, based on `origin/main` at `59ff27b`.
The previous Marathi progress commit was copied onto this branch. Nothing was
merged into or pushed to `main`. No model inference, parser run or training is
to be launched on Jai's laptop. Model-dependent work below is for Jainam later.

## What is ready

| Task | Delivered / verified | Remaining dependency |
|---|---|---|
| JI-1 Hindi rules | v1 implementation, tests and existing 5,000-sentence measurement | Reviewed Hindi labels for accuracy |
| JI-2 Hindi annotation | Guide, 200 texts, blind 50 packet, safe resumable CLI | Jai's 200 and Om's independent 50 reviews |
| JI-3 parser | Lazy Trankit and separately named Stanza implementation; existing smoke reports | No new parser runs on laptop; use Stanza explicitly later |
| JI-4 random | Teammate implementation and historical defaults preserved | None for code |
| JI-5 scoring | Pending-status rejection, packet identity checks, partial-set safeguards | Full reviewed sets, then parser scoring and κ |
| JI-6 caches | Local 200,000 / 500 / 1,012 rows; reusable full-row validator | Verify/upload Kaggle dataset and record its URL/version |
| JI-7 Marathi | Rules and existing statistics; 55 reviewed rows preserved; 95 remaining texts prepared | Human review of remaining 95 |
| JI-8 reach | Existing complete Hindi 5-tokenizer and Marathi 2-tokenizer studies | No required rerun |
| JI-9 error analysis | All-run pairing checks, R3/R6a comparisons, JSON + Markdown and examples | Matched token dumps from Om/Jainam |
| JI-10 C5 | Local cache input, progress reporting, class prevalence and both-class F1 metrics | Jainam's 2,000-sentence frozen-model run |
| Write-up | `docs/paper/data_and_linguistics.md`, `related_work.md`, `docs/references.bib` | Insert actual C5/κ/error-analysis results after they exist |

The 90-row Marathi progress file contains **55 `human_reviewed` + 35 `needs_review`**
records. The 55 are nonconsecutive IDs, so resuming at row 56 would be incorrect.
The remaining 60 IDs have no saved annotation. The provisional 90-row report is
now labelled mixed-status, not gold. Existing score values are preserved for audit.

The shared ChatGPT URL could not be retrieved. These instructions reflect the
repository and files in this checkout. If the chat or external folders contain
newer annotations/results, reconcile them by exact ID/text before adding labels.

## 1. Human annotation — Jai / Marathi reviewer / Om

Read `docs/annotation_guide.md`. Do not turn rule drafts into human gold. The
original 55 reviewed rows record Ruchir Padhye as annotator; retain that provenance.
The source-skewed 55-row set is not the balanced full Marathi evaluation set.

For ordinary installations use `python`. On this laptop, the old `.venv` points
at a missing Python 3.11.9 installation. A workspace-local runtime was downloaded
from python.org and configured to use the existing `.venv` packages:

```powershell
$env:PYTHONUTF8 = '1'
$jaiPython = Join-Path $PWD '.tools\python311\python.exe'
& $jaiPython scripts/jai_status.py
```

These are text-only annotation commands; they load no model. Start Marathi once
with `--resume-from` and a NEW output:

```powershell
& $jaiPython scripts/annotate.py --input data/annotation/mr_sentences.jsonl --out data/gold/mr_gold.jsonl --resume-from data/gold/mr_progress_original_90.jsonl --annotator 'YOUR NAME'
```

This copies only the 55 reviewed records and asks for the remaining 95. It never
shows the pending/draft group suggestions. On later sessions omit `--resume-from`:

```powershell
& $jaiPython scripts/annotate.py --input data/annotation/mr_sentences.jsonl --out data/gold/mr_gold.jsonl --annotator 'YOUR NAME'
& $jaiPython scripts/annotate.py --input data/annotation/hi_sentences.jsonl --out data/gold/hi_gold.jsonl --annotator jai
```

Enter zero-based merged spans, a type per multiword span, then notes. Blank spans
mean all singletons; `q` stops safely. Every reviewed row is flushed immediately.
`data/annotation/mr_remaining_text_only.jsonl` is also available for sharing the
95 outstanding texts, but use the full canonical input for the resume commands.

Give Om **only** `data/annotation/hi_om50_text_only.jsonl` and the guide, and ask him
to return `data/gold/hi_gold_om50.jsonl`. No team message has been sent automatically.
Om runs independently in his own checkout:

```bash
python scripts/annotate.py --input data/annotation/hi_om50_text_only.jsonl --out data/gold/hi_gold_om50.jsonl --annotator om --blind
```

Expected completion: Hindi 200/200, Marathi 150/150, Om 50/50. Run `jai_status.py`
again to verify. Do not calculate κ after changing either annotator's decisions
through adjudication; retain the independent originals.

## 2. Cache handoff — no model required

The cache folders already exist locally under `boundary_cache/`. Their row counts
and labels are checked in `results/grouping/boundary_cache_validation.json`.
Reproduce validation without loading a model:

```powershell
& $jaiPython scripts/validate_boundary_cache.py boundary_cache/hi_hi_rules_v1_train boundary_cache/hi_hi_rules_v1_eval boundary_cache/hi_hi_rules_v1_flores
```

The validator checks metadata, schema, every row's lengths, integer token IDs,
binary masks/boundaries, monotone group IDs and consistency of group starts. It
records a text hash for each split. It does not establish semantic group accuracy.

If a Kaggle dataset has already been published, record its actual URL and version
in `docs/HINDI_CACHE_HANDOFF.md`; the repository currently contains no proof of
publication. Otherwise upload the three folders (or the existing archive after
verifying its contents) as `mtp-boundary-cache`, with each folder's own `meta.json`.
Keep Arrow caches and model weights out of Git. On Kaggle, validate the attached
copy again and use the printed root, rather than assuming a mount layout.

## 3. Missing token dumps — ask Om/Jainam

The committed aggregate JSONs cannot reconstruct token-level correctness or
examples. Obtain matched `tokens_indiccorp_eval.jsonl` and/or `tokens_flores_hi.jsonl`
for R2, R3 and R6a; request R5/R6 only if those variants are actually run. Keep
dumps outside Git (already ignored). All runs must use the same texts, tokenizer,
maximum length and checkpoint step. Use one common analysis grouper.

After the dumps arrive, the analysis uses a tokenizer and rules, but no LM:

```bash
python scripts/group_error_analysis.py --baseline R2 --dump R2=results/R2_hi_mtp_k4_resblock/tokens_indiccorp_eval.jsonl --dump R3=results/R3_hi_k4_struct/tokens_indiccorp_eval.jsonl --dump R6a=results/R6a_hi_k4_struct_random/tokens_indiccorp_eval.jsonl --grouper hi_rules_v1 --out results/grouping/error_analysis_indiccorp.json
```

The command produces JSON and a Markdown table, plus up to 10 improved and 5 worse
far-head examples for each run versus R2. It rejects unmatched or duplicate keys,
including a mismatched random control. If fewer examples exist, report the actual
number. Do not manufacture examples from aggregate percentages.

## 4. Write-up and review

Use the new data/linguistics and related-work sections alongside Om's existing
`docs/paper/setup_and_evaluation.md`. C3 figures and measurements are ready now.
Keep the 55-row Marathi comparison explicitly partial. Add the human annotation
counts, κ, full scoring, C5 layer curves and error-analysis examples only after
their prerequisites have completed. The completed main results do not support
a grouping-driven speed-up; the write-up reflects that finding.

All changes stay on `jai/finish-data-linguistics` and are submitted for PR review
against `main`, as requested. No merge or team message is performed here.
Because `INTERFACES.md` adds an optional progress callback and report fields, use
`[contract change]` in the PR title, per the repository's working rules.

## 5. Model-dependent work — Jainam, later on Kaggle/GPU

**Deferred by Jai's explicit request. Do not execute these on his laptop.** This
section is last so annotation, code review and file handoffs can finish first.
No new LM training is needed merely to run C5: it freezes the base LM and trains
small CPU logistic probes. Parser scoring and token-dump generation also run
models and are deferred here with any optional new training.

Use a checkout of this branch, or a refreshed private `mtp-code` Kaggle input
containing these changes. Attach the validated cache dataset and the R2/R3/R6a
checkpoint datasets where needed. Enable Internet and the desired GPU. From the
repo root, use the team's environment setup (torch comes from the platform):

```bash
pip uninstall -y torchao
pip install -r requirements.txt -r requirements-jai.txt
pip install -e .
python -m pytest -q
```

The full suite passed 397 tests before the no-model instruction. Later final
data/tool checks are recorded separately; the full model-related suite must be
rerun here after the final code changes.

After the human sets are complete, score the explicit Stanza alternative (models
may download on first use). The scorer only chooses a final grouper after all
canonical IDs/texts are present and every row is reviewed:

```bash
python scripts/score_groupers.py --lang hi --gold data/gold/hi_gold.jsonl --agreement data/gold/hi_gold_om50.jsonl --groupers hi_rules_v0 hi_rules_v1 stanza random random_hi_v1
python scripts/score_groupers.py --lang mr --gold data/gold/mr_gold.jsonl --groupers mr_rules_v1 stanza random_mr_v1
```

Preserve these JSON/Markdown outputs. A winner does not retroactively change
the groupers used in completed R3/R6a/R7s runs. Do not silently replace historical
configs or re-label their logged training conditions.

Run the full C5 study, using actual attached cache paths for `TRAIN_CACHE` and
`EVAL_CACHE` (the two folders ending `_train` and `_eval`):

```bash
python scripts/probe_layers.py --lang hi --n 2000 --device auto --groupers hi_rules_v1 hi_rules_v0 random random_hi_v1 words --train-cache "$TRAIN_CACHE" --eval-cache "$EVAL_CACHE" --out results/probing
```

Without attached caches, omit both cache options and the existing corpus loader
is used. Save all five `hi_*.json` reports and `hi_layer_probes.pdf`. Check the
actual retained training count, disjoint train/eval hashes, all model layers,
class prevalences and majority baselines. Compare macro/attachment F1 as well as
boundary F1. Never rename `smoke_tiny_hi.json` as a production result.

If matched dumps are not already available, generate them from completed
checkpoints. Set these variables to actual run folders containing `config.yaml`
and `step_*`; use a separate output folder to preserve the official eval JSONs:

```bash
python scripts/evaluate.py --run_dir "$R2_RUN" --datasets indiccorp_eval flores_hi --dump_tokens --out dump_results
python scripts/evaluate.py --run_dir "$R3_RUN" --datasets indiccorp_eval flores_hi --dump_tokens --out dump_results
python scripts/evaluate.py --run_dir "$R6A_RUN" --datasets indiccorp_eval flores_hi --dump_tokens --out dump_results
```

Use the project's local FLORES Kaggle input as documented in the evaluation
notebook, or its authenticated loader if required. Point the section-3 analysis
command at the generated dump folders and return only the compact analysis
JSON/Markdown for Git; share the raw dumps separately.

Jainam's main experiments already exist on current `main`. Additional training is
optional research work, not a code-completion requirement. If full reviewed gold
motivates an additional matched v1 comparison, use a separately named config/run
pair, with `hi_rules_v1` versus `random_hi_v1`, and document the new loss/seed/data
choices. Existing R5/R6 templates are not evidence of completed runs and should
not be launched unchanged without Jainam's recipe decision. Return the resulting
checkpoint/dataset URLs, exact commit and evaluation artifacts before updating
the report with new model claims.
