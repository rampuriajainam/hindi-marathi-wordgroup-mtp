"""
Figures for paper/main_v2.tex, plain matplotlib, from committed result JSONs.

    python scripts/make_figures_v2.py        # -> results/figures/*.pdf, *.png and paper/figures/*.pdf

  fig_cost            speed-up and GPU energy per token against verified positions per token (results/cost_curve)
  fig_speed_rep       speed-up against repetition (rep-4) for exact, typical, repetition-penalised and sampled trees
                      (results/lossy_tree, results/rp_trees; IndicCorp, tree + lookup)
  fig_tokenizers      tokens per step vs words per step for ganga-1b, sarvam-1, Qwen2.5-1.5B (Table tab:models)
  fig_lexicon         Qwen2.5 Hindi: tokens per step with and without word-completion branches (results/lexicon)
  fig_tree            schematic of one draft tree: head branches, prompt-lookup chain, lexicon chains
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "figures"
PAPER = ROOT / "paper" / "figures"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]
HI, MR = "Rsd_hi_k4_frozen_sd_seq", "Rsd_r8_mr_k4_frozen_sd_seq"


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fig_cost():
    t = jload(R / "cost_curve" / "cost_table.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for run, lang, col in ((HI, "Hindi", C[0]), (MR, "Marathi", C[1])):
        rows = {r["policy"]: r for r in t[run]}
        for fam, ls, mk, lab in ((lambda p: p.startswith("tree") and "pld" not in p, "--", "o", "tree"),
                                 (lambda p: p.startswith("tree") and "pld" in p, "-", "s", "tree + lookup")):
            pts = sorted((r["pos_per_tok"], r["speedup"], r["j_per_tok"]) for p, r in rows.items() if fam(p))
            pts = [(1.0, 1.0, rows["greedy"]["j_per_tok"]), (rows["fixed_k"]["pos_per_tok"], rows["fixed_k"]["speedup"],
                                                             rows["fixed_k"]["j_per_tok"])] + pts
            a1.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker=mk, ms=3.5, color=col, label=f"{lang}, {lab}")
            a2.plot([p[0] for p in pts], [p[2] for p in pts], ls=ls, marker=mk, ms=3.5, color=col, label=f"{lang}, {lab}")
    a1.set(xlabel="verified positions per emitted token", ylabel="speed-up over greedy (T4)")
    a2.set(xlabel="verified positions per emitted token", ylabel="GPU energy per token (J)")
    a1.annotate("greedy", (1, 1), (2.5, 1.05), fontsize=6.5)
    a1.annotate("chain", (2.0, 1.87), (3.5, 1.62), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5))
    a2.set_ylim(0, None)
    a1.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def fig_speed_rep():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=False)
    for ax, run, lang in ((axes[0], HI, "Hindi"), (axes[1], MR, "Marathi")):
        pts = {}
        for sub in ("lossy_tree", "rp_trees"):
            for q in jload(R / sub / run / "quality_indiccorp_eval.sarvam-1.json")["policies"]:
                pts.setdefault(q["policy"], q)
        fams = [("exact", lambda p: p == "tree_static_n64+pld4x6", C[0], "o"),
                ("typical acceptance", lambda p: p.startswith("tree_static_n64+pld4x6+typ"), C[1], "^"),
                ("repetition penalty", lambda p: p.startswith("tree_static_n64+pld4x6+rp"), C[2], "s"),
                ("sampling", lambda p: p.startswith("tree_static_n64+pld4x6+samp"), C[3], "D")]
        for lab, f, col, mk in fams:
            xs = [(q["rep4"], q["speedup"], q["policy"]) for p, q in pts.items() if f(p)]
            ax.scatter([x[0] for x in xs], [x[1] for x in xs], color=col, marker=mk, s=18, label=lab, zorder=3)
            for r4, s, p in xs:
                tag = p.split("+")[-1]
                tag = {"samp0.7p0.9": "T=0.7", "samp1p1": "T=1", "rp1.1": "r=1.1", "rp1.2": "r=1.2"}.get(tag)
                if tag:
                    ax.annotate(tag, (r4, s), (4, 3), textcoords="offset points", fontsize=6)
        ax.axvline(pts["human"]["rep4"], color="gray", ls=":", lw=1)
        ax.text(pts["human"]["rep4"] + 0.008, 0.97, "human text", fontsize=6, color="gray", transform=ax.get_xaxis_transform(), va="top")
        ax.set(xlabel="repeated 4-grams in the output (rep-4)", title=f"{lang}, IndicCorp, tree + lookup", ylim=(0.9, None))
    axes[0].set_ylabel("speed-up over greedy / plain sampling")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


# Table tab:models (paper/main_v2.tex), IndicCorp, exact tree + lookup; Qwen and sarvam-1 from results/*/words_*.json,
# ganga-1b from its eval JSON divided by the tokens per word in the table.
MODELS = [("ganga-1b\n(1.12 tok/word)", 3.49, 3.30), ("sarvam-1\n(1.42 tok/word)", 3.42, 2.22),
          ("Qwen2.5-1.5B\n(4.67 tok/word)", 5.67, 1.17)]


def fig_tokenizers():
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    x = range(len(MODELS))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], [m[1] for m in MODELS], w, color=C[0], label="tokens per step")
    b2 = ax.bar([i + w / 2 for i in x], [m[2] for m in MODELS], w, color=C[1], label="words per step")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{b.get_height():.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([m[0] for m in MODELS])
    ax.set_ylabel("per verification step")
    ax.set_ylim(0, 6.6)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_lexicon():
    run = R / "lexicon" / "Rsd_qwen_hi_k4_frozen_sd_seq"
    pols = [("chain", "fixed_k"), ("tree", "tree_static_n64"), ("tree\n+ lexicon", "tree_static_n64+lex4e2x12"),
            ("tree\n+ lookup", "tree_static_n64+pld4x6"), ("tree + lookup\n+ lexicon", "tree_static_n64+pld4x6+lex4e2x12")]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    w = 0.38
    for j, (ds, lab) in enumerate((("indiccorp_eval", "IndicCorp"), ("flores_hi", "FLORES"))):
        e = {x["policy"]: x for x in jload(run / f"eval_{ds}.json")["spec_decode"]}
        ys = [e[p]["mean_accepted_len"] for _, p in pols]
        bars = ax.bar([i + (j - 0.5) * w for i in range(len(pols))], ys, w, color=C[j], label=lab)
        if j == 0:
            for b, p in zip(bars, pols):
                ax.text(b.get_x() + w, b.get_height() + 0.12, f"×{e[p[1]]['speedup']:.2f}", ha="center", fontsize=6)
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels([p[0] for p in pols], fontsize=6.5)
    ax.set_ylabel("tokens per step (Qwen2.5, Hindi)")
    ax.set_ylim(0, 7.2)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_tree():
    """Schematic, not a real tree: node labels are roles, not tokens."""
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.axis("off")
    nodes, edges = {}, []

    def add(name, x, y, kind, label, parent=None):
        nodes[name] = (x, y, kind, label)
        if parent:
            edges.append((parent, name, kind))

    add("root", 0, 0, "root", "$t_0$")
    add("a", 1, 1.6, "head", "h1:1", "root"); add("b", 1, 0.5, "head", "h1:2", "root"); add("c", 1, -0.5, "head", "h1:3", "root")
    add("a1", 2, 2.1, "head", "h2:1", "a"); add("a2", 2, 1.3, "head", "h2:2", "a"); add("b1", 2, 0.5, "head", "h2:1", "b")
    add("a11", 3, 2.1, "head", "h3:1", "a1")
    add("l1", 4, 2.1, "lex", "lex", "a11"); add("l2", 5, 2.1, "lex", "lex", "l1")
    add("l3", 3, 1.3, "lex", "lex", "a2")
    for i in range(4):
        add(f"p{i}", i + 1, -1.5, "pld", "copy", "root" if i == 0 else f"p{i - 1}")
    add("r1", 1, -2.4, "lex", "lex", "root"); add("r2", 2, -2.4, "lex", "lex", "r1")
    col = {"root": "black", "head": C[0], "pld": C[1], "lex": C[2]}
    for a, b, kind in edges:
        (x0, y0), (x1, y1) = nodes[a][:2], nodes[b][:2]
        ax.plot([x0, x1], [y0, y1], color=col[kind], lw=1, zorder=1)
    for x, y, kind, label in nodes.values():
        ax.scatter([x], [y], s=380, color="white", edgecolor=col[kind], linewidth=1.4, zorder=2)
        ax.text(x, y, label, ha="center", va="center", fontsize=5.5, zorder=3)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=col[k], marker="o", mfc="white", lw=1, label=l) for k, l in
         (("head", "MTP heads (top-k per head)"), ("pld", "prompt lookup (copied n-gram)"),
          ("lex", "lexicon (completions of the current word)"))]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=7)
    ax.text(0, 0.45, "verifier's\ntoken", ha="center", fontsize=6)
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(-2.8, 2.5)
    fig.tight_layout()
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    for name, fn in (("fig_cost", fig_cost), ("fig_speed_rep", fig_speed_rep), ("fig_tokenizers", fig_tokenizers),
                     ("fig_lexicon", fig_lexicon), ("fig_tree", fig_tree)):
        fig = fn()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=200)
        plt.close(fig)
        shutil.copy(OUT / f"{name}.pdf", PAPER / f"{name}.pdf")
        print("wrote", name)


if __name__ == "__main__":
    main()
