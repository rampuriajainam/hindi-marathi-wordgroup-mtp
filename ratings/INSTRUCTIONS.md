# Rating instructions (about 45 minutes each)

## Who rates what

| rater | sheet | items |
|---|---|---|
| Jai | `mr_sheet.csv` | Marathi, all 30 |
| Jainam | `hi_sheet_items01-30.csv` | Hindi 1-30 |
| Om | `hi_sheet_items21-50.csv` | Hindi 21-50 |

Hindi items 21-30 are rated by both Jainam and Om, so we can measure how much raters agree; rate them as you would any
other item. (`hi_sheet.csv` is the full Hindi sheet, items 1-50; the two parts above are its rows, item numbers
unchanged.)

## How to rate

Each item is a short Hindi or Marathi prompt with four machine continuations, labelled A-D. The labels are shuffled
per item; do not try to guess which system wrote what, and do not discuss items with the other raters until everyone
is done.

Save your own copy of your sheet with your name added (e.g. `mr_sheet_jai.csv`, `hi_sheet_items21-50_om.csv`) and fill
three columns for every row:

| column | scale |
|---|---|
| fluency (1-5) | 1 = broken or unreadable, 3 = understandable with errors or odd repetition, 5 = natural, as a person would write |
| coherence with prompt (1-5) | 1 = unrelated to the prompt or contradicts it, 5 = a sensible continuation of the prompt |
| rank in item (1=best) | rank the four continuations of the item from 1 (best overall) to 4 (worst); no ties |

Notes:
- Judge the text as written. Repetition (the same phrase or sentence again) counts against fluency.
- Continuations stop mid-sentence after 64 tokens; do not penalise the abrupt end.
- Excel: double-clicking the file should show Devanagari correctly; if it looks garbled, use Data > From Text/CSV and
  choose UTF-8.

Send the filled sheet back without changing the item, label or text columns.
