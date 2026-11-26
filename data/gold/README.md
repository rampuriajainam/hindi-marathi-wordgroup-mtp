# Annotation sets and provenance

## Current project gold — Jai review confirmation, 2026-10-22

Jai subsequently stated, "i have reviewed your annotations and they are right."
The 200 Hindi and 150 Marathi active rows are now designated project gold with
`dataset_role: "gold"`, `annotation_status: "human_reviewed"`, `reviewed_by: "jai"`,
and `review_required: false`. This records the reviewer's confirmation; the
assistant did not independently observe the review. All annotation contents,
including texts, groups, types and notes, are unchanged. `annotator: "ai_draft"`
and the original method retain AI provenance. Human review does not imply human
authorship. The scorer accepts these explicitly reviewed AI-origin records.

See [the review record](../../results/grouping/annotation_gold_review_2026-10-22.json)
for before/after file hashes and the prior commit. The provisional history below
remains an audit record, superseded by this confirmation. Original AI drafts,
historical human annotations and Om's blind packet remain unchanged. Independent
Om annotations and agreement are still pending; scores have not been recomputed.

## Fresh AI reannotation — 2026-10-22

At Jai's request, `hi_gold.jsonl` (200 sentences) and `mr_gold.jsonl`
(150 sentences) were reannotated from the canonical texts using
[`docs/annotation_guide.md`](../../docs/annotation_guide.md). Fresh-context AI
annotators saw only that guide and their assigned input sentences, without
previous annotations, rule predictions, grouping code or project results.
Separate fresh-context AI reviewers checked every sentence against the same guide.

Jai accepted eight randomly selected examples for provisional publication, while
explicitly requesting another review. The active files therefore record
`annotation_status: "approved"`, `approved_by: "jai"`, and
`review_required: true`. `annotator: "ai_draft"` preserves their AI origin.
This approval does not claim that every sentence was individually reviewed by a
human; the scorer still correctly rejects these files as human gold. Original AI
drafts retain `needs_review` in `data/annotation/hi_ai_redo.jsonl` and
`mr_ai_redo.jsonl`. Ambiguous decisions are recorded in row notes, and
`*_ai_review*.json` files record the separate AI review changes.
The validation/provenance report is
[`annotation_redo_2026-10-22.json`](../../results/grouping/annotation_redo_2026-10-22.json).
The exact eight examples and provisional approval are recorded in
[`annotation_review_sample_2026-10-22.json`](../../results/grouping/annotation_review_sample_2026-10-22.json).
The sanity check passed 148 offline data/tool tests and all annotation identity,
coverage, label, and protected-file checks. The canonical texts of `hi_0173` and
`mr_0058` already contain damaged characters; the source text is preserved.

The exact previous active files are preserved in
[`archive/2026-10-22_pre_ai_redo/`](archive/2026-10-22_pre_ai_redo/).
`mr_gold_reviewed_55.jsonl` and `mr_progress_original_90.jsonl` retain their original
content and provenance. Om's blind packet was not modified, and no Om annotations
were created or changed. Existing historical scores were not recomputed.

## Human annotation workflow and historical progress

Create `hi_gold.jsonl`, `mr_gold.jsonl` and `hi_gold_om50.jsonl` with `scripts/annotate.py` after reading `docs/annotation_guide.md`. Source-selected text packets and **unreviewed** rule suggestions are in `data/annotation/`. Do not rename a draft to gold; annotation status is checked by the scorer.

For a new human annotation session, use a new output filename: the CLI deliberately
refuses to resume a file containing pending AI annotations. Preserve the AI copy
and historical human originals when preparing the eventual reviewed set.

As of 2026-10-19, `mr_gold_reviewed_55.jsonl` preserves 55 reviewed records.
`mr_progress_original_90.jsonl` includes those same 55 plus 35 `needs_review`
records. Use `annotate.py --resume-from data/gold/mr_progress_original_90.jsonl`
with a NEW `--out data/gold/mr_gold.jsonl` to retain only reviewed rows and annotate
all 95 remaining IDs. See [the full handoff](../../docs/JAI_NEXT_STEPS.md).
At that date, Hindi and Om's independent annotations were not present in this checkout.
