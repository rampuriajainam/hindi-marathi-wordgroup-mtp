"""JI-10 frozen-model forward passes plus CPU logistic probes; no LM training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.grouping.base import get_grouper
from mtp.eval.layer_probes import run_multi_layer_probes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", choices=["hi", "mr"], default="hi")
    p.add_argument("--model")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument(
        "--groupers",
        nargs="+",
        default=None,
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-positions", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/probing"))
    p.add_argument("--train-cache", type=Path, help="Read training texts from a validated local cache")
    p.add_argument("--eval-cache", type=Path, help="Read evaluation texts from a validated local cache")
    a = p.parse_args()
    if a.n <= 0:
        p.error("--n must be positive")
    if bool(a.train_cache) != bool(a.eval_cache):
        p.error("provide both --train-cache and --eval-cache")
    a.model = a.model or (
        "LingoIITGN/ganga-1b"
        if a.lang == "hi"
        else "smallstepai/Misal-1B-instruct-v0.1"
    )
    a.groupers = a.groupers or (
        ["hi_rules_v1", "hi_rules_v0", "random_hi_v1", "words"]
        if a.lang == "hi"
        else ["mr_rules_v1", "random_mr_v1", "words"]
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mtp.data.corpus import load_split

    # Use platform device policy without constructing LoRA or prediction heads.
    from mtp.device import pick_device, pick_dtype

    device = pick_device() if a.device == "auto" else str(torch.device(a.device))
    dtype = torch.float32 if device == "cpu" else pick_dtype("auto")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    if a.train_cache:
        from mtp.data.boundary_cache import load_cache

        training_cache, train_meta = load_cache(a.train_cache, a.model, a.max_length)
        evaluation_cache, eval_meta = load_cache(a.eval_cache, a.model, a.max_length)
        if train_meta["lang"] != a.lang or eval_meta["lang"] != a.lang:
            p.error("cache language does not match --lang")
        if len(training_cache) < a.n:
            p.error("training cache contains fewer than --n sentences")
        train = training_cache.select(range(a.n))["text"]
        evaluation = list(evaluation_cache["text"])
    else:
        train = load_split(a.lang, "train", a.n)
        evaluation = load_split(a.lang, "eval")
    # Exact duplicates across raw splits would leak evaluation. Exclude from train.
    eval_set = set(evaluation)
    train = [t for t in train if t not in eval_set]
    print(f"Probing {a.model}: {len(train)} train / {len(evaluation)} eval sentences on {device}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    reports = run_multi_layer_probes(
        model,
        tok,
        [get_grouper(name, a.lang) for name in a.groupers],
        train,
        evaluation,
        str(device),
        a.max_length,
        a.max_positions,
        a.seed,
        progress=lambda stage, done, total: print(f"{stage}: {done}/{total}", flush=True),
    )
    for name, report in reports.items():
        report["model"] = a.model
        report["data_source"] = "local_boundary_cache_texts" if a.train_cache else "load_split"
        report["requested_train_sentences"] = a.n
        (a.out / f"{a.lang}_{name}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        ax.plot(report["layer"], report["f1"], label=name)
    ax.set(
        xlabel="Frozen-model layer (0 = embeddings)",
        ylabel="Next-token boundary F1",
        ylim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / f"{a.lang}_layer_probes.pdf")
    plt.close(fig)
    print(f"Saved probe reports and figure to {a.out}", flush=True)


if __name__ == "__main__":
    main()
