"""
KT-2 (b): is a draft easier to accept when the true token continues a word than when it starts one?

    python scripts/accept_by_token_type.py [--runs results/Rsd_hi_k4_frozen_sd_seq ...] [--n_boot 2000] [--seed 0] \\
        [--out results/analysis/accept_by_token_type.json]

Input: stored speculative-decoding outputs (results/<run>/eval_*.json, spec_decode entries with per_prompt
prompt_ids / ids / spans). Only the exact CHAIN policy is used: "fixed_k" when its per_prompt holds ids/spans, else
"fixed_k+fast" (the same fixed_k chain decoded with fp16-precast linear layers, mtp/eval/precast.py; older runs stored
token ids only for that entry). The entry used is recorded per row. No model is loaded; only the run's tokenizer
(model_name from the run's config in configs/, matched by run_name).

Span -> slots (mtp/eval/spec_decode.generate): a step that starts at generated index s proposes n drafts,
n = min(k-1, max_new_tokens - s - 1) (0 if the verifier token is EOS or it is the last token), with k = the run's
num_heads (config; checked against len(accept_rate_per_head) + 1). Draft slot d (1..n) targets index s+d. The step
emits span L = 1 + (accepted drafts), cut after EOS. So slots 1..L-1 were accepted, slot L (if L <= n) was the first
miss and its true token is ids[s+L] (the verifier's own next token, emitted by the next step), slots > L were never
evaluated (censored). A slot whose target index does not exist (generation ended at max_new_tokens or EOS) is not
evaluated. Check: per-head acceptance recomputed from the spans this way is compared with the stored
accept_rate_per_head (field "check_accept_rate_per_head").

Token classes (from the run's tokenizer pieces; ganga-1b and Misal-1B are SentencePiece-like, "▁" marks a word
start): punct (piece without "▁" is all punctuation/symbols, incl. the danda), digit (all digits, incl. Devanagari),
other (special tokens, byte-fallback pieces, a bare "▁"), else initial (piece starts with "▁") or internal (no "▁":
continues the previous word). The "▁" rule is checked against whitespace in the decoded text on a sample of prompts
(field "prefix_vs_text_agreement"). Position within the word: 1 for initial, else 1 + the number of preceding pieces
back to the nearest "▁" piece (reported as 2 and 3+).

Output per run, dataset, and per slot d plus pooled over slots: n evaluated per class, acceptance per class, the
difference internal - initial with a bootstrap 95% CI over prompts (n_boot resamples, seeded), the share of evaluated
initial+internal targets that are internal; pooled also slot-standardized (per-slot differences weighted by the slot's
evaluated initial+internal count), acceptance by position in the word, and the other classes.

Limitations (also in the JSON): targets are the verifier's own greedy output, not reference text; only evaluated
slots enter (everything after the first miss of a step is censored, so later slots are conditioned on the earlier
drafts being right); with ~1.1-1.2 tokens/word the internal class is small; a token class can correlate with other
things (frequency, the slot mix, repetition loops in greedy output) and this is not a causal estimate.
"""

import argparse
import glob
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import mtp` without `pip install -e .`

MAIN = ROOT.parent / "mtp-indic-research"      # main checkout: may hold result files the worktree lacks
DEFAULT_RUNS = [
    "results/Rsd_hi_k4_frozen_sd_seq",
    "results/Rsd_hi_k4_frozen_sd_seq_s43",
    "results/Rsd_hi_k4_frozen_sd",
    "results/Rsd_r8_mr_k4_frozen_sd_seq",
    "results/Rsd_r8_mr_k4_frozen_sd_seq_s43",
    "results/lossy/R2_hi_mtp_k4_resblock",
]
CHAIN_POLICIES = ("fixed_k", "fixed_k+fast")
CLASSES = ("initial", "internal", "punct", "digit", "other")
LIMITATIONS = (
    "Targets are the verifier's own greedy tokens. Only evaluated slots count: slots after the first miss of a step "
    "are censored, so slot d>1 is conditioned on the earlier drafts of that step being accepted. With ~1.1-1.2 "
    "tokens/word (ganga-1b, Misal-1B) the word-internal class is small. Class membership correlates with token "
    "frequency, slot mix and repetitive greedy output; this is descriptive, not causal."
)


def step_slots(spans, n_ids, k, max_new_tokens, eos_at=()):
    """Evaluated draft slots of one fixed_k generation.

    spans: tokens emitted per step; n_ids: generated length; k: heads (k-1 drafts per step);
    eos_at: generated indices that hold EOS (a step whose verifier token is EOS proposes no drafts).
    Returns (slots, censored): slots = [(target index, slot d, accepted)], censored = number of proposed slots never
    evaluated (after the first miss of a step, or beyond the end of the generation)."""
    eos_at = set(eos_at)
    slots, censored, s = [], 0, 0
    for L in spans:
        n = 0 if s in eos_at else max(0, min(k - 1, max_new_tokens - s - 1))
        for d in range(1, n + 1):
            if d <= L and s + d < n_ids:
                slots.append((s + d, d, d < L))
            else:
                censored += 1
        s += L
    return slots, censored


def classify_piece(piece, special=False):
    """initial / internal / punct / digit / other for a SentencePiece-like piece."""
    if special or piece is None:
        return "other"
    if piece.startswith("<0x") and piece.endswith(">"):
        return "other"
    body = piece.lstrip("▁")
    if not body or body.isspace():
        return "other"
    if all(unicodedata.category(c)[0] in "PS" for c in body):
        return "punct"
    if all(c.isdigit() for c in body):
        return "digit"
    return "initial" if piece.startswith("▁") else "internal"


def word_position(pieces, i):
    """1 if pieces[i] starts a word ("▁"), else 1 + pieces back to the nearest "▁" piece (None if none)."""
    j = i
    while j >= 0:
        if pieces[j] is not None and pieces[j].startswith("▁"):
            return i - j + 1
        j -= 1
    return None


def _boot_diff(counts, idx, slot):
    """Bootstrap internal - initial acceptance at `slot` (index into counts axis 1). counts: [P, S, 2cls, 2(n,a)]."""
    import numpy as np

    c = counts[idx, slot].sum(axis=1)                 # [B, 2, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = c[..., 1] / c[..., 0]
    return acc[:, 1] - acc[:, 0]


def analyze_entry(entry, tok, k, max_new_tokens, n_boot, seed):
    import numpy as np

    pp = entry["per_prompt"]
    special = set(tok.all_special_ids)
    eos = tok.eos_token_id
    P = len(pp["ids"])
    counts = np.zeros((P, k, 2, 2), dtype=np.int64)    # slot 0 = pooled, 1..k-1; class 0 initial, 1 internal
    other = {c: [0, 0] for c in CLASSES}                # n, accepted (pooled)
    pos = {"2": [0, 0], "3+": [0, 0]}
    censored_total = 0
    for p, (prompt, ids, spans) in enumerate(zip(pp["prompt_ids"], pp["ids"], pp["spans"])):
        assert sum(spans) == len(ids), "spans do not add up to the generated length"
        full = list(prompt) + list(ids)
        pieces = tok.convert_ids_to_tokens(full)
        eos_at = [i for i, t in enumerate(ids) if t == eos]
        slots, cens = step_slots(spans, len(ids), k, max_new_tokens, eos_at)
        censored_total += cens
        for i, d, acc in slots:
            g = len(prompt) + i
            cls = classify_piece(pieces[g], full[g] in special)
            other[cls][0] += 1
            other[cls][1] += acc
            if cls in ("initial", "internal"):
                c = 0 if cls == "initial" else 1
                for slot in (0, d):
                    counts[p, slot, c, 0] += 1
                    counts[p, slot, c, 1] += acc
                if cls == "internal":
                    wp = word_position(pieces, g)
                    if wp is not None:
                        key = "2" if wp == 2 else "3+"
                        pos[key][0] += 1
                        pos[key][1] += acc

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, P, size=(n_boot, P))
    tot = counts.sum(axis=0)                           # [S, 2, 2]

    def row(slot):
        n_i, a_i = tot[slot, 0]
        n_w, a_w = tot[slot, 1]
        r = {"n_initial": int(n_i), "n_internal": int(n_w),
             "acc_initial": a_i / n_i if n_i else None, "acc_internal": a_w / n_w if n_w else None,
             "internal_share": n_w / (n_i + n_w) if n_i + n_w else None}
        r["diff_internal_minus_initial"] = (r["acc_internal"] - r["acc_initial"]
                                            if n_i and n_w else None)
        if n_i and n_w:
            b = _boot_diff(counts, idx, slot)
            ok = b[~np.isnan(b)]
            r["diff_ci95"] = [float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))]
            r["boot_valid"] = int(ok.size)
        else:
            r["diff_ci95"], r["boot_valid"] = None, 0
        return r

    per_slot = {str(d): row(d) for d in range(1, k)}
    pooled = row(0)
    # slot-standardized pooled difference: per-slot differences weighted by the slot's evaluated count
    w = np.array([tot[d, :, 0].sum() for d in range(1, k)], dtype=float)
    valid = np.array([tot[d, 0, 0] > 0 and tot[d, 1, 0] > 0 for d in range(1, k)])
    if valid.any():
        w = np.where(valid, w, 0.0)
        w = w / w.sum()
        diffs = np.array([per_slot[str(d)]["diff_internal_minus_initial"] or 0.0 for d in range(1, k)])
        boots = np.stack([_boot_diff(counts, idx, d) if valid[d - 1] else np.zeros(n_boot) for d in range(1, k)], 1)
        bstd = (np.nan_to_num(boots) * w).sum(1)
        bad = np.isnan(boots[:, valid]).any(1)
        bstd = bstd[~bad]
        pooled["slot_standardized_diff"] = float((diffs * w).sum())
        pooled["slot_standardized_ci95"] = [float(np.percentile(bstd, 2.5)), float(np.percentile(bstd, 97.5))]
        pooled["slot_standardized_weights"] = {str(d): float(w[d - 1]) for d in range(1, k)}
    pooled["by_word_position"] = {
        "1": {"n": int(tot[0, 0, 0]), "acc": pooled["acc_initial"]},
        **{key: {"n": v[0], "acc": v[1] / v[0] if v[0] else None} for key, v in pos.items()}}
    classes = {c: {"n": v[0], "acc": v[1] / v[0] if v[0] else None} for c, v in other.items()}
    n_eval = sum(v[0] for v in other.values())
    for c in classes:
        classes[c]["share"] = classes[c]["n"] / n_eval if n_eval else None
    return {"n_prompts": P, "n_evaluated_slots": n_eval, "n_censored_slots": censored_total,
            "classes_pooled": classes, "per_slot": per_slot, "pooled": pooled}


def recomputed_accept_rate(entry, k, max_new_tokens, eos):
    """Per-head accepted/proposed from spans with the step_slots rule (proposed counts censored slots too)."""
    prop, acc = [0] * (k - 1), [0] * (k - 1)
    for ids, spans in zip(entry["per_prompt"]["ids"], entry["per_prompt"]["spans"]):
        s = 0
        for L in spans:
            n = 0 if ids[s] == eos else max(0, min(k - 1, max_new_tokens - s - 1))
            for d in range(1, n + 1):
                prop[d - 1] += 1
                acc[d - 1] += d < L
            s += L
    return [a / p if p else None for a, p in zip(acc, prop)]


def prefix_vs_text(entry, tok, n_prompts=20):
    """Share of generated tokens where ("▁" prefix) == (decoded text grows by a leading whitespace)."""
    agree = total = 0
    for prompt, ids in list(zip(entry["per_prompt"]["prompt_ids"], entry["per_prompt"]["ids"]))[:n_prompts]:
        full = list(prompt) + list(ids)
        pieces = tok.convert_ids_to_tokens(full)
        prev = tok.decode(full[: len(prompt)], skip_special_tokens=False)
        for g in range(len(prompt), len(full)):
            cur = tok.decode(full[: g + 1], skip_special_tokens=False)
            if not cur.startswith(prev):       # decode not prefix-stable here (byte fallback): skip
                prev = cur
                continue
            added = cur[len(prev):]
            prev = cur
            if not added or full[g] in tok.all_special_ids:
                continue
            total += 1
            agree += pieces[g].startswith("▁") == added[0].isspace()
    return {"agreement": agree / total if total else None, "tokens_checked": total, "prompts": n_prompts}


def run_config(run_name):
    import yaml

    for path in sorted(glob.glob(str(ROOT / "configs" / "*.yaml"))):
        try:
            cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(cfg, dict) and cfg.get("run_name") == run_name:
            return cfg, path
    return None, None


def find_run_dir(rel):
    for base in (ROOT, MAIN):
        p = base / rel
        if p.is_dir() and glob.glob(str(p / "eval_*.json")):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--max_new_tokens", type=int, default=64, help="as in the eval (scripts/evaluate.py default)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/analysis/accept_by_token_type.json"))
    args = ap.parse_args()
    from transformers import AutoTokenizer

    rows, skipped, toks = [], {}, {}
    for rel in args.runs:
        run_dir = find_run_dir(rel)
        if run_dir is None:
            skipped[rel] = "no eval_*.json found in worktree or main checkout"
            continue
        for path in sorted(run_dir.glob("eval_*.json")):
            d = json.loads(path.read_text(encoding="utf-8"))
            entry = None
            for name in CHAIN_POLICIES:
                for e in d.get("spec_decode", []):
                    if e.get("policy") == name and all(f in e.get("per_prompt", {})
                                                       for f in ("prompt_ids", "ids", "spans")):
                        entry = entry or e
            if entry is None:
                skipped[str(path)] = "no fixed_k entry with prompt_ids/ids/spans"
                continue
            cfg, cfg_path = run_config(d["run_name"])
            if cfg is None:
                skipped[str(path)] = f"no config with run_name {d['run_name']}"
                continue
            k = int(cfg["num_heads"])
            assert k == len(entry["accept_rate_per_head"]) + 1, "num_heads disagrees with accept_rate_per_head"
            mt = int(args.max_new_tokens)
            assert max(len(x) for x in entry["per_prompt"]["ids"]) <= mt, "generation longer than max_new_tokens"
            name = cfg["model_name"]
            if name not in toks:
                toks[name] = AutoTokenizer.from_pretrained(name)
            tok = toks[name]
            res = analyze_entry(entry, tok, k, mt, args.n_boot, args.seed)
            rec = recomputed_accept_rate(entry, k, mt, tok.eos_token_id)
            row = {"run": d["run_name"], "dataset": d["dataset"], "file": str(path), "config": cfg_path,
                   "tokenizer": name, "lang": cfg.get("lang"), "head_type": cfg.get("head_type", "parallel"),
                   "num_heads": k, "policy_entry": entry["policy"], "ignore_eos": entry.get("ignore_eos"),
                   "mean_accepted_len": entry.get("mean_accepted_len"),
                   "check_accept_rate_per_head": {"stored": entry["accept_rate_per_head"], "recomputed": rec},
                   "prefix_vs_text_agreement": prefix_vs_text(entry, tok)}
            row.update(res)
            rows.append(row)
            print(f"{d['run_name']} {d['dataset']} ({entry['policy']}): {res['n_evaluated_slots']} slots", flush=True)

    out = {"description": "KT-2 (b): draft acceptance by the true token's type (word-initial vs word-internal), "
                          "exact fixed_k chain, see scripts/accept_by_token_type.py docstring.",
           "limitations": LIMITATIONS, "n_boot": args.n_boot, "seed": args.seed,
           "max_new_tokens": args.max_new_tokens, "skipped": skipped, "results": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    def f(x, p=1):
        return "-" if x is None else f"{100 * x:.{p}f}"

    print("\n| run | dataset | slot | n initial | n internal | internal share | acc initial | acc internal | "
          "diff (pp) | 95% CI (pp) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        for slot, s in list(r["per_slot"].items()) + [("pooled", r["pooled"])]:
            ci = s["diff_ci95"]
            print(f"| {r['run']} | {r['dataset']} | {slot} | {s['n_initial']} | {s['n_internal']} | "
                  f"{f(s['internal_share'])}% | {f(s['acc_initial'])}% | {f(s['acc_internal'])}% | "
                  f"{f(s['diff_internal_minus_initial'])} | "
                  f"{'-' if ci is None else f'[{f(ci[0])}, {f(ci[1])}]'} |")
        p = r["pooled"]
        if "slot_standardized_diff" in p:
            ci = p["slot_standardized_ci95"]
            print(f"| {r['run']} | {r['dataset']} | pooled, slot-std | | | | | | "
                  f"{f(p['slot_standardized_diff'])} | [{f(ci[0])}, {f(ci[1])}] |")
    print("\n| run | dataset | acc pos 2 (n) | acc pos 3+ (n) | punct share / acc | digit share / acc | "
          "other share | prefix vs text |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        wp, c = r["pooled"]["by_word_position"], r["classes_pooled"]
        print(f"| {r['run']} | {r['dataset']} | {f(wp['2']['acc'])}% ({wp['2']['n']}) | "
              f"{f(wp['3+']['acc'])}% ({wp['3+']['n']}) | {f(c['punct']['share'])}% / {f(c['punct']['acc'])}% | "
              f"{f(c['digit']['share'])}% / {f(c['digit']['acc'])}% | {f(c['other']['share'])}% | "
              f"{f(r['prefix_vs_text_agreement']['agreement'], 2)}% |")
    if skipped:
        print("\nskipped:", json.dumps(skipped, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
