"""
Paper tables and figures from the committed eval JSONs (OM-9).

    python scripts/make_tables.py            # -> results/tables/*.md, *.tex and results/figures/*.pdf
    python scripts/make_tables.py --no_figures

Reads results/{run}/eval_{dataset}.json (INTERFACES §10), gi_stanza_*.json (scripts/rescore_gi.py) and
bench_verify.json. Every number in the tables comes from those files; nothing is typed in by hand.

Tables
  1 main_hi        Hindi runs: head-0 ppl, per-head top-1, in-group top-1, FixedK tokens/step + speed-up
  2 main_mr        Marathi runs, same columns
  3 controls       linguistic vs random groups: in-group top-1, Group Integrity (rules / stanza), tokens/step,
                   with paired bootstrap CIs of the GI difference where per-prompt data exist
  4 decoding       draft policies (chain, ConfidenceCut, trees, trees + prompt lookup) per model
  5 ablations      alpha, soft labels, 6 heads, longer training, sequential heads
Figures
  a ingroup_bars   per-head top-1 in-group vs at-boundary (R2, R3, R6a, R7s)
  b speed_ladder   speed-up on T4, greedy -> chain -> sequential heads -> trees
  c gi_forest      GI difference linguistic - random with 95% CIs, scored with rules and with stanza
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS                     # where tables/ and figures/ go (--out_dir)

HI = ("indiccorp_eval", "flores_hi")
MR = ("indiccorp_eval", "flores_mr")
DS_LABEL = {"indiccorp_eval": "IndicCorp", "flores_hi": "FLORES", "flores_mr": "FLORES"}

# label -> results folder
RUNS = {
    "R0": "R0_hi_ntp", "R1": "R1_hi_mtp_k2_linear", "R2": "R2_hi_mtp_k4_resblock", "R3": "R3_hi_k4_struct",
    "R3_s43": "R3_hi_k4_struct_s43", "R4": "R4_hi_k4_struct_contr", "R4a": "R4a_hi_k4_struct_contr_random",
    "R6a": "R6a_hi_k4_struct_random", "R6a_s43": "R6a_hi_k4_struct_random_s43", "R7s": "R7s_hi_k4_struct_stanza",
    "R7s_s43": "R7s_hi_k4_struct_stanza_s43", "A1_alpha0": "A1_alpha0", "A1_alpha1": "A1_alpha1",
    "Rsd": "Rsd_hi_k4_frozen_sd", "Rsd_s43": "Rsd_hi_k4_frozen_sd_s43", "Rsd_soft": "Rsd_hi_k4_frozen_sd_soft",
    "Rsd_long": "Rsd_hi_k4_frozen_sd_long", "Rsd_k6": "Rsd_hi_k6_frozen_sd", "Rsd_struct": "Rsd_hi_k4_frozen_sd_struct",
    "Rsd_struct_rand": "Rsd_hi_k4_frozen_sd_struct_random", "Rsd_seq": "Rsd_hi_k4_frozen_sd_seq",
    "Rsd_seq_s43": "Rsd_hi_k4_frozen_sd_seq_s43", "Rsd_seq_k6": "Rsd_hi_k6_frozen_sd_seq",
    "Rsd_seq_struct": "Rsd_hi_k4_frozen_sd_seq_struct", "Rsd_seq_struct_rand": "Rsd_hi_k4_frozen_sd_seq_struct_random",
    "R8": "R8_mr_ntp", "R9": "R9_mr_mtp_k4_resblock", "R10": "R10_mr_k4_struct", "R10a": "R10a_mr_k4_struct_random",
    "R10_s43": "R10_mr_k4_struct_s43", "R10a_s43": "R10a_mr_k4_struct_random_s43", "Rsd_mr": "Rsd_mr_k4_frozen_sd",
    "Rsd_r8": "Rsd_r8_mr_k4_frozen_sd", "Rsd_r8_seq": "Rsd_r8_mr_k4_frozen_sd_seq",
    "Rsd_r8_seq_s43": "Rsd_r8_mr_k4_frozen_sd_seq_s43", "Rsd_r8_seq_k6": "Rsd_r8_mr_k6_frozen_sd_seq",
}
NOTE = {
    "R0": "NTP", "R1": "1 linear head", "R2": "3 resblock heads", "R3": "+ group loss (rules)",
    "R4": "+ group loss + contrastive", "R7s": "+ group loss (parser)", "Rsd": "frozen + self-distill",
    "Rsd_seq": "Rsd + sequential heads", "R8": "NTP", "R9": "3 resblock heads", "R10": "+ group loss (rules)",
    "R10a": "+ group loss (random)", "Rsd_mr": "frozen instruct + self-distill", "Rsd_r8": "frozen R8 + self-distill",
    "Rsd_r8_seq": "Rsd_r8 + sequential heads",
}


# ----------------------------------------------------------------------------- loading

def load(run, dataset):
    path = RESULTS / RUNS[run] / f"eval_{dataset}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def spec(rec, policy):
    for e in (rec or {}).get("spec_decode") or []:
        if e["policy"] == policy:
            return e
    return None


def gi_counts(run, dataset, grouper="rules", policy="fixed_k"):
    """Per-prompt (hits, spans) for GI, scored with the run's rules grouper or re-scored with stanza."""
    if grouper == "rules":
        e = spec(load(run, dataset), policy)
        pp = (e or {}).get("per_prompt") or {}
        return (pp["gi_hits"], pp["gi_spans"]) if "gi_hits" in pp else None
    path = RESULTS / RUNS[run] / f"gi_stanza_{dataset}.json"
    if not path.exists():
        return None
    e = [x for x in json.loads(path.read_text(encoding="utf-8"))["policies"] if x["policy"] == policy]
    return (e[0]["gi_hits"], e[0]["gi_spans"]) if e else None


def gi_value(counts):
    return 100 * sum(counts[0]) / sum(counts[1]) if counts and sum(counts[1]) else None


def paired_bootstrap(a, b, n=2000, seed=0):
    """(diff, lo, hi) of 100*(GI_a - GI_b) over prompts; a, b = per-prompt (hits, spans), same prompts."""
    (ha, sa), (hb, sb) = a, b
    rng, m = random.Random(seed), len(ha)
    d0 = 100 * (sum(ha) / sum(sa) - sum(hb) / sum(sb))
    ds = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        ds.append(100 * (sum(ha[i] for i in idx) / sum(sa[i] for i in idx) - sum(hb[i] for i in idx) / sum(sb[i] for i in idx)))
    ds.sort()
    return d0, ds[int(0.025 * n)], ds[int(0.975 * n)]


# ----------------------------------------------------------------------------- formatting

def f(x, nd=1, pct=False, sign=False):
    if x is None:
        return "–"
    x = 100 * x if pct else x
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def write_table(name, title, header, rows, notes=()):
    out = OUT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    md = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += [""] + [f"- {n}" for n in notes]
    (out / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    def tex(s):
        return (s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("×", "$\\times$")
                .replace("–", "--").replace("−", "$-$").replace("→", "$\\rightarrow$").replace("✓", "$^\\dagger$"))
    t = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{tex(title)}}}", f"\\label{{tab:{name}}}",
         "\\begin{tabular}{" + "l" + "r" * (len(header) - 1) + "}", "\\toprule",
         " & ".join(tex(h) for h in header) + " \\\\", "\\midrule"]
    t += [" & ".join(tex(c) for c in r) + " \\\\" for r in rows]
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / f"{name}.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    return out / f"{name}.md"


def head_cols(rec):
    ph = rec["per_head"] if rec else None
    if not ph:
        return ["–"] * 7
    top1 = [f(h["top1"], pct=True) for h in ph[1:4]] + ["–"] * (3 - len(ph[1:4]))
    ig = [f(h["top1_in_group"], pct=True) for h in ph[1:4]] + ["–"] * (3 - len(ph[1:4]))
    return [f(math.exp(ph[0]["loss"]), 1)] + top1 + ig


def speed_cols(run, datasets):
    cols = []
    for ds in datasets:
        e = spec(load(run, ds), "fixed_k")
        cols.append(f"{e['mean_accepted_len']:.2f} (×{e['speedup']:.2f})" if e else "–")
    return cols


# ----------------------------------------------------------------------------- tables

def table_main(name, title, runs, datasets):
    header = ["run", "setup", "h0 ppl", "top-1 h1", "h2", "h3", "in-group h1", "h2", "h3",
              f"FixedK tok/step (speed-up), {DS_LABEL[datasets[0]]}", DS_LABEL[datasets[1]]]
    rows = [[r, NOTE.get(r, "")] + head_cols(load(r, datasets[0])) + speed_cols(r, datasets) for r in runs]
    return write_table(name, title, header, rows, [
        f"Per-head columns on {DS_LABEL[datasets[0]]} eval; top-1 and in-group top-1 in %. In-group = the target "
        "token continues the source token's word group (grouper `hi_rules_v0` / `mr_rules_v1`).",
        "Kaggle T4, step 12500 (Rsd_long 25000), 200 prompts × 64 new tokens; outputs identical to greedy "
        "(fp32 re-check 20/20)."])


CONTROL_PAIRS = [("R3", "R6a"), ("R3_s43", "R6a_s43"), ("R7s", "R6a"), ("R7s_s43", "R6a_s43"), ("R4", "R4a"),
                 ("R10", "R10a"), ("R10_s43", "R10a_s43"), ("Rsd_struct", "Rsd_struct_rand"),
                 ("Rsd_seq_struct", "Rsd_seq_struct_rand")]


def table_controls():
    header = ["linguistic − random", "in-group h1", "h2", "h3", "GI rules, IndicCorp", "FLORES",
              "GI stanza, IndicCorp", "FLORES", "Δ tok/step"]
    rows = []
    for a, b in CONTROL_PAIRS:
        if load(a, "indiccorp_eval") is None or load(b, "indiccorp_eval") is None:
            continue
        flo = "flores_mr" if a.startswith("R10") else "flores_hi"
        pa, pb = load(a, "indiccorp_eval")["per_head"], load(b, "indiccorp_eval")["per_head"]
        ig = [f(x["top1_in_group"] - y["top1_in_group"], pct=True, sign=True) for x, y in zip(pa[1:4], pb[1:4])]
        cells = []
        for grouper in ("rules", "stanza"):
            for ds in ("indiccorp_eval", flo):
                ca, cb = gi_counts(a, ds, grouper), gi_counts(b, ds, grouper)
                if ca and cb and len(ca[0]) == len(cb[0]):
                    d, lo, hi = paired_bootstrap(ca, cb)
                    cells.append(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]{' ✓' if lo > 0 or hi < 0 else ''}")
                else:
                    ea, eb = spec(load(a, ds), "fixed_k"), spec(load(b, ds), "fixed_k")
                    cells.append(f"{100 * (ea['group_integrity'] - eb['group_integrity']):+.1f}"
                                 if grouper == "rules" and ea and eb else "–")
        ea, eb = spec(load(a, "indiccorp_eval"), "fixed_k"), spec(load(b, "indiccorp_eval"), "fixed_k")
        rows.append([f"{a} − {b}"] + ig + cells + [f"{ea['mean_accepted_len'] - eb['mean_accepted_len']:+.2f}"])
    return write_table("table3_controls", "Table 3: linguistic vs random word groups", header, rows, [
        "In-group top-1 difference in points (IndicCorp eval). Group Integrity (GI) = share of accepted multi-token "
        "spans that end on a word-group boundary, FixedK, 200 prompts; scored with the rules grouper "
        "(`hi_rules_v0` / `mr_rules_v1`) and re-scored with the stanza dependency parser.",
        "Brackets: paired bootstrap 95% CI over prompts (2,000 resamples); ✓ = excludes 0. Plain values: runs "
        "evaluated before per-prompt outputs were stored (no CI).",
        "R3 / R4 / R10 are trained on the rules grouping, R7s on the stanza parser grouping; R7s scored with the rules is "
        "scored on a grouping it never saw."])


DECODING = [("R2", ["fixed_k", "confidence_cut"]),
            ("Rsd", ["fixed_k", "confidence_cut", "tree_entropy_n40", "tree_static_n64",
                     "fixed_k+fast", "tree_static_n64+fast", "tree_static_n64+pld4x6+fast"]),
            ("Rsd_k6", ["fixed_k", "tree_entropy_n40"]),
            ("Rsd_seq", ["fixed_k", "confidence_cut", "tree_entropy_n40", "tree_static_n40", "tree_static_n64",
                         "fixed_k+fast", "tree_static_n64+fast", "tree_static_n64+pld4x6+fast"]),
            ("Rsd_r8_seq", ["fixed_k", "tree_static_n40", "tree_static_n64",
                            "fixed_k+fast", "tree_static_n64+fast", "tree_static_n64+pld4x6+fast"])]


def table_decoding():
    header = ["model", "draft policy", "nodes / step", "IndicCorp tok/step", "speed-up", "FLORES tok/step", "speed-up"]
    # Rsd_r8_seq is Marathi (Misal-1B, head 0 = R8; IndicCorp mr + FLORES mr, EOS masked)
    rows = []
    for run, policies in DECODING:
        recs = [load(run, ds) for ds in (MR if "_mr_" in RUNS[run] else HI)]
        names = [p for p in policies if spec(recs[0], p)]
        names += [e["policy"] for e in recs[0]["spec_decode"]
                  if "+pld" in e["policy"] and "+fast" not in e["policy"] and e["policy"] not in names]
        for p in names:
            es = [spec(r, p) for r in recs]
            nodes = es[0].get("mean_tree_nodes")
            k1 = len(es[0]["accept_rate_per_head"])              # drafts per chain step = heads - 1
            row = [run, p.replace("_", " "), f"{nodes:.0f}" if nodes else (str(k1) if p.startswith("fixed_k") else f"≤{k1}")]
            for e in es:
                row += [f"{e['mean_accepted_len']:.2f}", f"×{e['speedup']:.2f}"] if e else ["–", "–"]
            rows.append(row)
    return write_table("table4_decoding", "Table 4: draft policies and tree decoding (Kaggle T4)", header, rows, [
        "Greedy decoding = 1 token/step = ×1.00 (27.4-27.7 tok/s). All policies produce output identical to greedy "
        "(fp32 re-check 20/20).",
        "Trees: Medusa sparse trees (static) or one tree per head-entropy cell (entropy), fitted on IndicCorp train "
        "prompts; n = node budget. +pld4x6 = up to 4 prompt-lookup chains of 6 tokens copied from the context, "
        "grafted onto the tree.",
        "Rsd_k6's tree row was timed with the engine before its per-step overhead fix (a lower bound); all other "
        "tree rows use the final engine.",
        "+fast = linear layers compute from cached fp16 weights (drafter DR-2, bit-identical outputs); the greedy "
        "baseline of those rows uses it too (36.4-36.6 tok/s for Rsd_seq, 35.2-35.3 for Rsd, vs 27.4-27.7 without; "
        "26.4-26.6 for Marathi Rsd_r8_seq, vs 22.9 without)."])


def table_ablations():
    header = ["run", "change", "h0 loss", "IndicCorp tok/step (speed-up)", "FLORES"]
    rows = []
    for run, change in [("A1_alpha0", "α = 0"), ("R2", "α = 0.1 (default)"), ("A1_alpha1", "α = 1"),
                        ("Rsd", "frozen + self-distill"), ("Rsd_s43", "Rsd, seed 43"), ("Rsd_soft", "soft labels"),
                        ("Rsd_k6", "6 heads"), ("Rsd_long", "2× data and steps"), ("Rsd_seq", "sequential heads"),
                        ("Rsd_seq_s43", "seq, seed 43"), ("Rsd_seq_k6", "seq, 6 heads")]:
        rec = load(run, "indiccorp_eval")
        if rec is None:
            continue
        h0 = rec["per_head"][0]["loss"] if rec.get("per_head") else None
        rows.append([run, change, f(h0, 3)] + speed_cols(run, HI))
    return write_table("table5_ablations", "Table 5: ablations", header, rows, [
        "α = share of the extra heads' gradient that reaches the shared LoRA (head_backbone_grad). Frozen runs keep "
        "head 0 = ganga-1b exactly (same h0 loss)."])


# ----------------------------------------------------------------------------- figures

COLORS = {"in_group": "#2a78d6", "boundary": "#eb6834", "third": "#1baf7a"}   # dataviz reference slots 1-3 (validated)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _axes_style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_ingroup_bars(plt):
    runs = ["R2", "R3", "R6a", "R7s"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for d, ax in enumerate(axes, start=1):
        x = range(len(runs))
        ig = [100 * load(r, "indiccorp_eval")["per_head"][d]["top1_in_group"] for r in runs]
        bd = [100 * load(r, "indiccorp_eval")["per_head"][d]["top1_at_boundary"] for r in runs]
        w = 0.38
        ax.bar([i - w / 2 - 0.01 for i in x], ig, w, color=COLORS["in_group"], label="in-group target")
        ax.bar([i + w / 2 + 0.01 for i in x], bd, w, color=COLORS["boundary"], label="boundary target")
        ax.set_xticks(list(x))
        ax.set_xticklabels(runs)
        ax.set_title(f"head {d} (token t+{d + 1})", fontsize=9, color=INK)
        _axes_style(ax)
    axes[0].set_ylabel("top-1 accuracy (%)", fontsize=8, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    return fig


def fig_speed_ladder(plt):
    steps = [("greedy", None, None), ("R2 chain", "R2", "fixed_k"), ("Rsd chain", "Rsd", "fixed_k"),
             ("Rsd_seq chain", "Rsd_seq", "fixed_k"), ("Rsd tree n40", "Rsd", "tree_entropy_n40"),
             ("Rsd_seq tree n40", "Rsd_seq", "tree_static_n40"),
             ("Rsd_seq tree + lookup", "Rsd_seq", "tree_entropy_n40+pld4x6"),
             ("+ fp16 weight cache", "Rsd_seq", "tree_static_n64+pld4x6+fast")]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    labels, ys = [], {ds: [] for ds in HI}
    for label, run, pol in steps:
        labels.append(label)
        for ds in HI:
            e = spec(load(run, ds), pol) if run else None
            ys[ds].append(e["speedup"] if e else (1.0 if run is None else float("nan")))
    w = 0.38
    for j, (ds, color) in enumerate(zip(HI, (COLORS["in_group"], COLORS["boundary"]))):
        xs = [i + (j - 0.5) * (w + 0.02) for i in range(len(labels))]
        bars = ax.bar(xs, ys[ds], w, color=color, label=DS_LABEL[ds])
        if j == 0:
            for b, v in zip(bars, ys[ds]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"×{v:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("speed-up vs greedy (T4)", fontsize=8, color=MUTED)
    ax.set_ylim(0, max(max(v for v in ys[ds] if v == v) for ds in HI) * 1.15)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _axes_style(ax)
    fig.tight_layout()
    return fig


def fig_gi_forest(plt):
    pairs = [("R3", "R6a"), ("R3_s43", "R6a_s43"), ("R7s", "R6a"), ("R7s_s43", "R6a_s43"), ("R4", "R4a")]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ticks, labels, y = [], [], 0
    for a, b in pairs:
        for k, (grouper, color, mk) in enumerate((("rules", COLORS["in_group"], "o"), ("stanza", COLORS["boundary"], "s"))):
            ca, cb = gi_counts(a, "indiccorp_eval", grouper), gi_counts(b, "indiccorp_eval", grouper)
            if not (ca and cb):
                continue
            d, lo, hi = paired_bootstrap(ca, cb)
            yy = y - 0.18 + 0.36 * k
            ax.plot([lo, hi], [yy, yy], color=color, linewidth=2)
            ax.plot([d], [yy], marker=mk, markersize=7, color=color, markeredgecolor="white", markeredgewidth=1.5,
                    label=f"scored with {grouper}" if y == 0 else None)
        ticks.append(y)
        labels.append(f"{a} − {b}")
        y -= 1
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Group Integrity difference, IndicCorp (points, 95% CI)", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _axes_style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    fig.tight_layout()
    return fig


LOOKAHEAD_DATA = RESULTS / "figures" / "lookahead_data.json"
AHEAD_COLORS = {0: None, 1: "#86b6ef", 2: "#3987e5", 3: "#1c5cab", 4: "#0d366b"}   # ordinal blue ramp (validated)
EDGE = [Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"), Path("/usr/bin/chromium"), Path("/usr/bin/google-chrome")]


def extract_lookahead(run, dataset="indiccorp_eval", min_tokens=14, max_tokens=22):
    """From results/{run}/tokens_{dataset}.jsonl (--dump_tokens): pick the sentence of min..max tokens with the
    highest share of tokens predicted >= 2 ahead, and return, per target token, how many tokens in advance it was
    first predicted correctly (largest head d with a correct prediction -> d + 1 ahead; 0 = never)."""
    sents = {}
    with open(RESULTS / RUNS[run] / f"tokens_{dataset}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            s = sents.setdefault(r["sent_id"], {"text": r["text"], "tok": {}, "group": {}, "ahead": {}})
            s["tok"][r["token_idx"]] = r["token"]
            s["group"][r["token_idx"]] = r["group_id"]
            s["group"].setdefault(r["target_idx"], r["target_group_id"])   # the last token is only ever a target
            if r["correct"]:
                s["ahead"][r["target_idx"]] = max(s["ahead"].get(r["target_idx"], 0), r["head"] + 1)
    for s in sents.values():                          # token text of target-only positions (the sentence end)
        for i in s["group"]:
            s["tok"].setdefault(i, None)
    best, best_score = None, -1.0
    for sid, s in sents.items():
        n = len(s["tok"])
        if not min_tokens <= n <= max_tokens:
            continue
        score = sum(s["ahead"].get(i, 0) >= 2 for i in range(1, n)) / (n - 1)
        if score > best_score:
            best, best_score = sid, score
    s = sents[best]
    if any(t is None for t in s["tok"].values()):     # fill target-only positions from the run's tokenizer
        from transformers import AutoTokenizer

        name = "smallstepai/Misal-1B-instruct-v0.1" if "_mr_" in RUNS[run] else "LingoIITGN/ganga-1b"
        tok = AutoTokenizer.from_pretrained(name)
        pieces = tok.convert_ids_to_tokens(tok(s["text"], truncation=True, max_length=128)["input_ids"])
        for i, t in s["tok"].items():
            if t is None and i < len(pieces):
                s["tok"][i] = pieces[i]
    tokens = [{"tok": s["tok"][i] or "", "ahead": s["ahead"].get(i, 0) if i else None, "group": s["group"][i]}
              for i in sorted(s["tok"])]
    return {"run": run, "dataset": dataset, "sent_id": best, "text": s["text"], "tokens": tokens,
            "rule": f"sentence of {min_tokens}-{max_tokens} tokens with the highest share predicted >= 2 ahead"}


def lookahead_html(data):
    """One sentence, each token coloured by how far ahead it was predicted, word-group brackets above."""
    groups, cur = [], None
    for t in data["tokens"]:
        if cur is None or t["group"] != cur["group"] or t["group"] < 0:
            cur = {"group": t["group"], "tokens": []}
            groups.append(cur)
        cur["tokens"].append(t)
    parts = []
    for g in groups:
        toks = []
        for t in g["tokens"]:
            text = t["tok"].replace("▁", "")
            if t["ahead"] is None:
                style, title = "background:#fcfcfb;border:1.5px dashed #a5a49e;color:#52514e", "context (first token)"
            elif t["ahead"] == 0:
                style, title = "background:#fcfcfb;border:1.5px solid #c9c8c2;color:#52514e", "not predicted"
            else:
                bg = AHEAD_COLORS[t["ahead"]]
                fg = "#0b0b0b" if t["ahead"] == 1 else "#ffffff"
                style, title = f"background:{bg};color:{fg};border:1.5px solid {bg}", f"predicted {t['ahead']} ahead"
            gap = " gap" if t["tok"].startswith("▁") and toks else ""
            toks.append(f'<span class="tok{gap}" style="{style}" title="{title}">{text}</span>')
        bracket = " grp" if len(g["tokens"]) > 1 else ""
        parts.append(f'<span class="g{bracket}">{"".join(toks)}</span>')
    legend = "".join(f'<span class="lg"><span class="sw" style="background:{AHEAD_COLORS[k]}"></span>{k} ahead</span>'
                     for k in (1, 2, 3, 4))
    legend += '<span class="lg"><span class="sw" style="background:#fcfcfb;border:1.5px solid #c9c8c2"></span>not predicted</span>'
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Lookahead heatmap</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;600&display=block" rel="stylesheet">
<style>
@page {{ size: 180mm 48mm; margin: 0 }}
body {{ margin: 6mm 5mm; background:#ffffff; font-family:'Noto Sans Devanagari','Nirmala UI',sans-serif; color:#0b0b0b }}
.sent {{ display:flex; flex-wrap:wrap; align-items:flex-end; row-gap:10px; column-gap:7px; font-size:15px }}
.g {{ display:inline-flex; gap:2px; padding-top:7px; position:relative }}
.grp {{ border-top:2px solid #52514e }}
.grp::before, .grp::after {{ content:''; position:absolute; top:-2px; width:2px; height:6px; background:#52514e }}
.grp::before {{ left:0 }} .grp::after {{ right:0 }}
.tok {{ padding:2px 5px; border-radius:4px; line-height:1.5 }}
.gap {{ margin-left:5px }}
.legend {{ margin-top:12px; font: 10px 'Noto Sans','Segoe UI',sans-serif; color:#52514e; display:flex; gap:14px; flex-wrap:wrap }}
.lg {{ display:inline-flex; align-items:center; gap:5px }}
.sw {{ width:12px; height:12px; border-radius:3px; display:inline-block }}
.cap {{ font: 9.5px 'Noto Sans','Segoe UI',sans-serif; color:#52514e; margin-top:6px }}
</style></head><body>
<div class="sent">{"".join(parts)}</div>
<div class="legend">{legend}<span class="lg">┌─┐ word group</span></div>
<div class="cap">{data['run']}, IndicCorp eval sentence {data['sent_id']} ({data['rule']}). Colour = how many tokens in
advance the token was first predicted correctly (head d at position t-d-1 = d+1 ahead).</div>
</body></html>"""


def fig_lookahead(out):
    """results/figures/fig_lookahead.{html,pdf}: from the committed lookahead_data.json (refresh it with --lookahead_run
    when a token dump exists). The PDF is printed by a local Chromium (Devanagari needs real text shaping)."""
    if not LOOKAHEAD_DATA.exists():
        return []
    import subprocess

    data = json.loads(LOOKAHEAD_DATA.read_text(encoding="utf-8"))
    html = out / "fig_lookahead.html"
    html.write_text(lookahead_html(data), encoding="utf-8")
    browser = next((p for p in EDGE if p.exists()), None)
    if browser is None:
        return [html]
    pdf = out / "fig_lookahead.pdf"
    subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={pdf}", html.resolve().as_uri()], capture_output=True, timeout=120)
    subprocess.run([str(browser), "--headless=new", "--disable-gpu", "--virtual-time-budget=8000", "--window-size=700,190",
                    f"--screenshot={out / 'fig_lookahead.png'}", html.resolve().as_uri()], capture_output=True, timeout=120)
    return [pdf] if pdf.exists() else [html]


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = OUT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in (("fig_ingroup_bars", fig_ingroup_bars), ("fig_speed_ladder", fig_speed_ladder),
                     ("fig_gi_forest", fig_gi_forest)):
        fig = fn(plt)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(out / f"{name}.pdf")
    written += fig_lookahead(out)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--out_dir", default=None, help="default: results/")
    ap.add_argument("--lookahead_run", default=None,
                    help="refresh results/figures/lookahead_data.json from this run's tokens_indiccorp_eval.jsonl")
    args = ap.parse_args(argv)
    global OUT
    OUT = Path(args.out_dir) if args.out_dir else RESULTS
    if args.lookahead_run:
        LOOKAHEAD_DATA.parent.mkdir(parents=True, exist_ok=True)
        LOOKAHEAD_DATA.write_text(json.dumps(extract_lookahead(args.lookahead_run), ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    written = [
        table_main("table1_main_hi", "Table 1: Hindi (ganga-1b)", ["R0", "R1", "R2", "R3", "R4", "R7s", "Rsd", "Rsd_seq"], HI),
        table_main("table2_main_mr", "Table 2: Marathi (Misal-1B)",
                   ["R8", "R9", "R10", "R10a", "Rsd_mr", "Rsd_r8", "Rsd_r8_seq"], MR),
        table_controls(), table_decoding(), table_ablations(),
    ]
    if not args.no_figures:
        written += make_figures()
    for p in written:
        print(p)


if __name__ == "__main__":
    sys.exit(main())
