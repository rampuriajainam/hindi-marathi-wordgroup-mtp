# Second backbone: SD-seq heads on Qwen2.5-1.5B and sarvam-1

Question: with more tokens per word, do draft heads accept more tokens per step (word-internal tokens are easy) while
each token carries less text? The honest unit is **words** per verification step and per second. Same recipe as
`Rsd_seq` (frozen backbone, self-distilled text, 4 sequential heads, trees + prompt lookup + `--fast_linear`) on

| model | arch | params | layers / H | vocab | tokenizer | Hindi tokens/word (IndicCorp eval / FLORES) |
|---|---|---|---|---|---|---|
| `LingoIITGN/ganga-1b` (reference, `Rsd_seq`) | Mistral | 0.995B | 16 / 2048 | 30,000 | SentencePiece, no BOS | 1.118 / 1.176 |
| `sarvamai/sarvam-1` (`configs/Rsd_sarvam_seq.yaml`) | Llama, untied | 2.525B | 28 / 2048 | 68,096 | SentencePiece + byte fallback, adds `<s>` | 1.418 / 1.401 |
| `Qwen/Qwen2.5-1.5B` (`configs/Rsd_qwen_seq.yaml`) | Qwen2, tied, q/k/v bias | 1.544B | 28 / 1536 | 151,936 | byte-level BPE, no BOS, pad = EOS | 4.671 / 4.756 |

Tokens/word: 500 IndicCorp `hin_Deva` eval sentences (30,927 words) and 1,012 FLORES devtest sentences, tokenizers only,
no special tokens. Marathi IndicCorp eval: ganga 2.466, Misal 1.585, Qwen 6.589, sarvam 1.924. Both models are ungated
on the Hub (`/api/models/<id>`, 2026-11-13).

## 1. What had to change in the code

Nothing in the training / decoding path broke on Qwen2 or Llama (tested: `tests/test_second_model.py`). One fix:

- `mtp/eval/precast.py`: `fast_linear` skipped linear layers whose weight is tied to an embedding, i.e. Qwen's
  `lm_head` (233M params, the largest per-call fp32 -> fp16 cast). Now a tied layer's `forward` is redirected to the
  cached fp16 copy (`F.linear`), so the shared embedding stays fp32. Bit-identical (tests + real Qwen2.5-1.5B on CPU);
  no change for ganga / Misal (untied).

Things that work as is but need the right config:

- `data.cache_dir: null`. The boundary cache holds ganga token ids; `train.py` refuses it (tokenizer check) if attached.
  A frozen self-distilled run needs no groups for training; `data.grouper` only labels the logged eval split
  (works with Qwen offsets) and can be `null`.
- `evaluate.py` runs with no grouper (`--grouper` unset and `data.grouper: null`): `group_integrity` = None.
- Group integrity with byte-level BPE: `align.py` gives every token the group of the character it ends in, so
  tokens inside a split character are labelled correctly when the generated ids re-tokenize to themselves (22-23 of 32
  Qwen continuations). Otherwise `spec_decode.token_group_ids` decodes prefixes; a token that only completes a
  character gets an empty span (group -1) and spans touching it are not counted. GI for Qwen is therefore slightly
  undercounted and, at ~4.7 tokens/word, naturally low (most accepted spans end inside a word). It is not an outcome
  of this experiment.
- BOS: sarvam adds `<s>` like Misal (already handled by `make_prompts`, `token_group_ids`, the collator, `gen_selfdistill`).
- EOS: Qwen2.5-1.5B (base) never stopped early: 32 IndicCorp eval prompts, 128 new tokens, 0/32 hit EOS (bf16 and fp16).
  `--ignore_eos auto` = off is right. sarvam-1 is a base model too but was not run locally (see §4, EOS check).
- fp16: Qwen2.5 in fp16 gave no inf/NaN on those 32 prompts; fp16 and bf16 greedy outputs were identical on 9/32
  (near-ties, as with ganga). The evaluation's fp32 re-check is the correctness check.

## 2. Budgets (text-matched)

Every budget in the ganga runs counts tokens. Keeping them would give Qwen ~14 words of continuation instead of ~57.
Proposed: scale token budgets by tokens/word so the amount of TEXT stays the same.

| | ganga (Rsd_seq) | sarvam-1 | Qwen2.5-1.5B |
|---|---|---|---|
| prompt (8-16 words, unchanged) | ~13 tokens | ~18 (+ `<s>`) | ~56 |
| `max_new_tokens` (SD data, eval, fit_tree) | 64 | **80** (64 x 1.42 / 1.12) | **256** (64 x 4.67 / 1.12 = 267) |
| training `max_length` | 128 | 128 (texts ~98 tokens) | **320** (measured SD texts: 269-350 tokens, median 308) |
| micro-batch x grad_accum (8 texts / step) | 8 x 1 | 4 x 2 | 2 x 4 |
| prompt lookup `--prompt_lookup` | 4 6 | 4 6 and 4 8 | 4 6 and 4 24 (text-matched) |

Trade-off, measured on the laptop (RTX 4060, bf16, 50 frozen SD-seq steps, 8 texts/step):

| | s/step | vs ganga |
|---|---|---|
| ganga, Rsd_seq, 128 | 0.11 | 1x |
| Qwen, token-matched (`max_length 128`) | 0.49 | 4.5x |
| Qwen, text-matched (`max_length 320`) | 0.97 | 8.8x |

Text-matched Qwen costs 2x the token-matched one per step: the 152k-vocab head logits dominate. Same structural caveat
for the hypothesis: with k = 4 the heads can add at most 3 tokens/step = 0.64 Qwen words; ganga's Rsd_seq already
gives 1.89 words/step (chain) and 3.30 (tree n64 + lookup, `scripts/words_per_step.py` on its stored outputs). So Qwen
cannot match ganga in words/step at k = 4 whatever its acceptance; `--set num_heads=8` is the natural follow-up.

## 3. Memory on a 15 GB T4 (14.7 GiB usable)

fp32 weights + fp16 autocast (`dtype: auto`). Weights: Qwen 5.75 GiB, sarvam 9.41 GiB (ganga 3.71). `fast_linear`
adds an fp16 copy of every linear layer: Qwen 2.88 GiB, sarvam 4.44 GiB.

| stage | Qwen2.5-1.5B | sarvam-1 |
|---|---|---|
| gen_selfdistill (fp16 weights, batch 64) | ~2.9 + KV ~0.6 GiB: fine | ~4.7 + KV ~0.7 GiB: fine |
| training, frozen, 8 texts/step | **measured** 1.33 GiB activations per 320-token text (bf16 laptop, 5.54 GiB peak at micro-batch 2); T4: 5.75 + 2 x 1.33 + ~1.3 (fp16 casts of the frozen tied head saved for 3 heads' backward) ~= **9.7 GiB at micro-batch 2** (~12.4 at 4; batch 8 x 1 does not fit: OOM already at 4 on the 8 GB laptop) | estimate 9.41 + ~0.8 activations (4 x 98 tokens x 68k vocab) + ~0.8 fp16 head casts ~= **11-11.5 GiB at micro-batch 4** |
| evaluation + `--fast_linear` | 5.75 + 2.88 = **8.6 GiB**: fine | 9.41 + 4.44 = **13.85 GiB** + context/KV: **borderline** |

sarvam evaluation: run the per-head section and the spec-decode section as two calls (per-head without
`--fast_linear`), set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; if spec decode still OOMs, drop
`--fast_linear` for sarvam (speed-ups are then vs the slower greedy, as in ganga's pre-DR-2 numbers, which exist for
the same policies).

## 4. Kaggle recipe (T4 x2)

Setup as in `notebooks/kaggle_train.ipynb` cell 1 (code, `pip uninstall -y torchao`, requirements, tests). Internet on
(model downloads; IndicCorp streams). All commands from `/tmp/mtp` with `export PYTHONUTF8=1`.

### sarvam-1 (one session)

```bash
# 1. self-distillation data, one shard per GPU (100k texts = 12500 steps x 8, as Rsd_seq)
for g in 0 1; do CUDA_VISIBLE_DEVICES=$g python -u scripts/gen_selfdistill.py --lang hi --model_name sarvamai/sarvam-1 \
  --n 100000 --shard $g --num_shards 2 --max_new_tokens 80 --batch_size 64 \
  --out /kaggle/working/sd/sarvam_hi_sd.$g.jsonl > /kaggle/working/gen_sarvam_$g.log 2>&1 & done; wait
cat /kaggle/working/sd/sarvam_hi_sd.{0,1}.jsonl > /kaggle/working/sd/sarvam_hi_sd.jsonl
# EOS check: share of texts that stopped early (if more than a few %, regenerate with --min_new_tokens 80 and
# evaluate with --ignore_eos on, the Misal recipe)
python -c "import json; from transformers import AutoTokenizer as A; t=A.from_pretrained('sarvamai/sarvam-1'); \
L=[len(t(json.loads(l)['text'])['input_ids']) for l in open('/kaggle/working/sd/sarvam_hi_sd.jsonl')][:2000]; \
print(sorted(L)[::200])"
# 2. train (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --config configs/Rsd_sarvam_seq.yaml --run_root /kaggle/working/runs \
  --resume auto --stop_after_min 600 > /kaggle/working/train_Rsd_sarvam_seq.log 2>&1
R=/kaggle/working/runs/Rsd_sarvam_hi_k4_frozen_sd_seq
# 3. trees
CUDA_VISIBLE_DEVICES=0 python -u scripts/fit_tree.py --run_dir $R --nodes 25 40 64 --max_new_tokens 80 --out_dir /kaggle/working/results
T=/kaggle/working/results/Rsd_sarvam_hi_k4_frozen_sd_seq
# 4. evaluation, one dataset per GPU; per-head section first (no fast_linear), then spec decode
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for d in "0 indiccorp_eval" "1 flores"; do set -- $d; (CUDA_VISIBLE_DEVICES=$1 python -u scripts/evaluate.py --run_dir $R \
  --datasets $2 --out /kaggle/working/results && \
 CUDA_VISIBLE_DEVICES=$1 python -u scripts/evaluate.py --run_dir $R --datasets $2 --no_heads --spec_decode --spec_n 200 \
  --max_new_tokens 80 --policies fixed_k --tree_policies $T/tree_policy_static_n40.json $T/tree_policy_static_n64.json \
  --prompt_lookup 4 6 --fast_linear --out /kaggle/working/results) > /kaggle/working/eval_sarvam_$2.log 2>&1 & done; wait
```

### Qwen2.5-1.5B (two sessions; one session only with 50k texts)

Same commands with `--model_name Qwen/Qwen2.5-1.5B --max_new_tokens 256 --batch_size 64` (lower `--batch_size` to 32 if
generation is slow or OOMs), `--out /kaggle/working/sd/qwen_hi_sd.$g.jsonl` -> `qwen_hi_sd.jsonl`,
`--config configs/Rsd_qwen_seq.yaml`, `R=/kaggle/working/runs/Rsd_qwen_hi_k4_frozen_sd_seq`, and:

- fit_tree: `--max_new_tokens 64` (rank statistics per position; 300 + 200 greedy prompts at 256 tokens would take ~1.5 h);
- evaluate: `--max_new_tokens 256 --fp32_check_n 10 --prompt_lookup 4 6`, then a second spec-decode call with
  `--prompt_lookup 4 24 --policies fixed_k --out /kaggle/working/results_pld24` (a spec-decode call replaces the whole
  `spec_decode` section of its output file, so the second lookup length needs its own `--out`).
- Session 1: generation (both GPUs) + training with `--stop_after_min` up to the budget; publish `runs/` and `sd/`.
  Optional on the idle GPU 1 while GPU 0 trains: the token-matched control from the same file,
  `--set run_name=Rsd_qwen_hi_k4_frozen_sd_seq_t128 --set data.max_length=128 --set optim.batch_size=4 --set optim.grad_accum=2`
  (the first 128 tokens of each text = prompt + ~70 greedy tokens).
- Session 2: attach the published runs (cell 2 restores them), `--resume auto` to finish, fit_tree, evaluation.
- Single-session fallback: `--n 50000` and `--set optim.max_steps=6250` (Rsd_long showed 2x data = +0.05 tokens/step
  for ganga, so half the data should cost little; it is still a difference from Rsd_seq to state).

### Time estimates

How derived: training from measured laptop s/step x 1.73 (ganga Rsd_seq: 0.19 s/step on T4, `checkpoints/kaggle/Rsd_seq/train.log`,
vs 0.11 on the laptop); sarvam training not measured, ganga's T4 0.19 s/step scaled by non-embedding params (2.25B / 0.87B)
x tokens per text (98 / 77). Generation from laptop timings (Qwen batch 32 x 256: 16.5 s = 0.52 s/text; at batch 64 the
laptop ran out of memory and slowed 6x; ganga batch 64 x 64: 2.6 s) with T4 assumed 1-1.5x slower; check the first
`gen_*.log` lines. Evaluation from ganga's T4 greedy (27.8 tok/s, 36.5 with fast_linear) scaled by layers (28 / 16):
~21 tok/s Qwen, ~18 tok/s sarvam with fast_linear. These are estimates, not measurements.

| stage | sarvam-1 | Qwen2.5-1.5B (100k texts) | Qwen (50k) |
|---|---|---|---|
| gen_selfdistill, 2 GPUs | 2-3 h | 4-6 h | 2-3 h |
| training 12500 steps (6250) | ~2.2 h (0.6 s/step) | ~5.9 h (1.7 s/step) | ~2.9 h |
| fit_tree | ~0.3 h | ~0.5 h (`--max_new_tokens 64`) | ~0.5 h |
| evaluation, 2 datasets on 2 GPUs, 200 prompts | ~1.5 h | ~3.5-4 h (+ ~1.5 h second lookup length) | same |
| **total** | **~6-7 h: one session** | **~14-17 h: two sessions** | **~9-10 h: one session, little margin** |

## 5. Words per step and per second

`scripts/words_per_step.py results/<run>/eval_<dataset>.json --run_dir <run>` reads the stored per-prompt outputs (no
decoding). Per prompt: words = whitespace words of `decode(prompt_ids + ids)` minus those of `decode(prompt_ids)`;
steps = verification steps. Per policy: words/step = sum(words) / sum(steps); tokens/word = sum(tokens) / sum(words);
words/sec = tokens_per_sec / tokens_per_word (tokens_per_sec is total tokens / total seconds); the speed-up is unchanged.
Prompts are the same sentences and the same 8-16 words for every model (`make_prompts` counts words, seeded per sentence),
so per-prompt words/step can be paired across models (bootstrap over prompts). The reference, ganga Rsd_seq on
IndicCorp eval (fast_linear): chain 1.89 words/step, 62.5 words/s vs greedy 34.3; tree n64 + lookup 3.30, 90.7 words/s.

## 6. Laptop smoke (Qwen2.5-1.5B, not speed numbers)

64 IndicCorp train prompts -> `gen_selfdistill --max_new_tokens 256` -> 50 SD-seq steps (frozen, 2 x 4) -> `fit_tree`
(20 / 10 prompts) -> `evaluate.py --spec_decode --spec_n 4 --max_new_tokens 48` with the run's `dtype` set to `fp16`
(fp32 weights + fp16 autocast, the T4 path): fixed_k 1.66 tokens/step, static n25 2.21, + lookup 2.37; **fp32 re-check
4/4 for every policy**; fp16 matched greedy on 3/4 prompts (divergence at margin 0.000, an exact tie). `fast_linear`
bit-identical on the real model (CPU, bf16 autocast). Qwen's Hindi: fluent sentences but very repetitive: 57% of word
trigrams in its continuations are repeats, vs 31% for ganga on the same prompts, so prompt lookup and acceptance will be
inflated relative to ganga. Head-0 eval loss 1.46 nats/token = ~6.8 nats/word (ganga 3.05 x 1.12 = ~3.4).
