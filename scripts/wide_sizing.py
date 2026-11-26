"""
Offline sizing check for a wide drafter state (drafter idea "seq_wide"): how much room do the next K states need, and
how much of it is predictable from h_t? Frozen backbone only, no training of the LM.

    python scripts/wide_sizing.py --train_file sd/hi_sd.jsonl --lang hi [--horizon 3] [--max_positions 60000]

Collects h_t and the stacked future Y_t = [h_{t+1}; ...; h_{t+K}] (K*H numbers) from the backbone's last hidden states:
the fit set from --train_file texts (self-distilled), the measure set from the eval sentences. Reports, in two
geometries (raw hidden states, and "logit" = each H-block multiplied by (W^T W)^{1/2}, W = the LM head, so distances
are the ones the LM head sees):

  pcs_single      principal components for 95 / 99% of the variance of h_{t+1}: the scale of one state
  pcs_stacked     the same for Y_t: room needed to hold K states
  linear          ridge h_t -> Y_t: R^2 per offset and overall, and the PCs of its predictions. A linear map of h_t
                  has rank <= H, so its prediction count can never exceed H: it cannot argue for D > H by itself.
  mlp             a 2-layer MLP h_t -> Y_t (hidden 2H, early stopping on 10% of the fit set): R^2 and prediction PCs.
                  mlp R^2 - linear R^2 is how much non-linear features of h_t add, which is what extra state
                  dimensions could hold.

Writes results/drafter/wide_sizing_{lang}.json.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import mtp` without `pip install -e .`

FRACS = (0.95, 0.99)


def collect(model, tokenizer, texts, horizon, device, max_length=128, batch_size=16):
    """(X [N, H], Y [N, K*H]) float32 on CPU, every position t with t + K inside the text."""
    import torch

    decoder = model.get_decoder()
    xs, ys = [], []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i: i + batch_size], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, add_special_tokens=False).to(device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            hid = decoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state.float()
        for b, n in enumerate(enc.attention_mask.sum(1).tolist()):
            h = hid[b, :n] if tokenizer.padding_side == "right" else hid[b, -n:]
            if n <= horizon:
                continue
            T = n - horizon
            xs.append(h[:T].cpu())
            ys.append(torch.cat([h[d: d + T] for d in range(1, horizon + 1)], dim=1).cpu())
    return torch.cat(xs), torch.cat(ys)


def pcs(Y, fracs=FRACS):
    """Number of principal components reaching each fraction of the total variance."""
    import torch

    Yc = Y - Y.mean(0)
    ev = torch.linalg.eigvalsh(Yc.T @ Yc / len(Yc)).flip(0).clamp(min=0)
    cum = ev.cumsum(0) / ev.sum()
    return {f"{int(100 * f)}%": int((cum < f).sum()) + 1 for f in fracs}


def r2(pred, Y, H):
    """Fraction of variance explained, overall and per H-block (offset)."""
    sst = ((Y - Y.mean(0)) ** 2)
    sse = (pred - Y) ** 2
    per = [float(1 - sse[:, d * H:(d + 1) * H].sum() / sst[:, d * H:(d + 1) * H].sum()) for d in range(Y.shape[1] // H)]
    return {"overall": float(1 - sse.sum() / sst.sum()), "per_offset": per}


def ridge(X, Y, lam=1e-3):
    import torch

    mx, my = X.mean(0), Y.mean(0)
    Xc = X - mx
    A = Xc.T @ Xc
    A += lam * A.diagonal().mean() * torch.eye(A.shape[0], device=A.device)
    W = torch.linalg.solve(A, Xc.T @ (Y - my))
    return lambda Z: (Z - mx) @ W + my


def mlp(X, Y, hidden, epochs=30, lr=1e-3, batch=512, seed=0):
    """2-layer SiLU MLP with early stopping on the last 10% of (X, Y). Returns the predict function."""
    import torch

    torch.manual_seed(seed)
    n_val = max(1, len(X) // 10)
    Xt, Yt, Xv, Yv = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]
    mx, sx, my, sy = Xt.mean(0), Xt.std(0) + 1e-6, Yt.mean(0), Yt.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, Y.shape[1])).to(X.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=X.device)
        for i in range(0, len(Xt), batch):
            j = perm[i: i + batch]
            loss = ((net((Xt[j] - mx) / sx) - (Yt[j] - my) / sy) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net((Xv - mx) / sx) - (Yv - my) / sy) ** 2).mean())
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(best_state)

    def predict(Z):
        with torch.no_grad():
            return net((Z - mx) / sx) * sy + my
    return predict


def sizing(X_fit, Y_fit, X_eval, Y_eval, H, G=None, mlp_hidden=None, mlp_epochs=30):
    """All measurements in one geometry. G: [H, H] applied to every H-block (None = raw)."""
    if G is not None:
        K = Y_fit.shape[1] // H
        tr = lambda M, k: (M.view(len(M), k, H) @ G).view(len(M), k * H)
        X_fit, X_eval, Y_fit, Y_eval = tr(X_fit, 1), tr(X_eval, 1), tr(Y_fit, K), tr(Y_eval, K)
    out = {"pcs_single": pcs(Y_eval[:, :H]), "pcs_stacked": pcs(Y_eval)}
    lin = ridge(X_fit, Y_fit)
    p = lin(X_eval)
    out["linear"] = {"r2": r2(p, Y_eval, H), "pcs_pred": pcs(p)}
    net = mlp(X_fit, Y_fit, mlp_hidden or 2 * H, epochs=mlp_epochs)
    p = net(X_eval)
    out["mlp"] = {"r2": r2(p, Y_eval, H), "pcs_pred": pcs(p)}
    return out


def logit_metric(lm_weight):
    """(W^T W)^{1/2}, scaled to unit mean eigenvalue: H-dim distances as the LM head sees them."""
    import torch

    w = lm_weight.float()
    ev, V = torch.linalg.eigh(w.T @ w)
    s = ev.clamp(min=0).sqrt()
    return (V * (s / s.mean())) @ V.T


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_file", required=True, help="jsonl with {'text'} per line (self-distilled)")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--model_name", default="LingoIITGN/ganga-1b")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--n_fit_texts", type=int, default=4000)
    ap.add_argument("--max_positions", type=int, default=60000)
    ap.add_argument("--mlp_epochs", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM

    from mtp.data.corpus import load_split
    from mtp.model.build import load_tokenizer

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    tok = load_tokenizer(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32).to(device).eval()
    with open(args.train_file, encoding="utf-8") as f:
        fit_texts = [json.loads(line)["text"] for line in f if line.strip()][: args.n_fit_texts]
    eval_texts = load_split(args.lang, "eval")
    X_fit, Y_fit = collect(model, tok, fit_texts, args.horizon, device)
    if len(X_fit) > args.max_positions:
        idx = torch.tensor(sorted(random.Random(0).sample(range(len(X_fit)), args.max_positions)))
        X_fit, Y_fit = X_fit[idx], Y_fit[idx]
    perm = torch.randperm(len(X_fit), generator=torch.Generator().manual_seed(0))   # early-stopping split is random
    X_fit, Y_fit = X_fit[perm].to(device), Y_fit[perm].to(device)
    X_eval, Y_eval = (t.to(device) for t in collect(model, tok, eval_texts, args.horizon, device))
    H = X_fit.shape[1]
    print(f"H {H}, horizon {args.horizon}, fit positions {len(X_fit)}, eval positions {len(X_eval)}", flush=True)
    G = logit_metric(model.get_output_embeddings().weight.detach().to(device))
    del model
    rec = {"model_name": args.model_name, "lang": args.lang, "H": H, "horizon": args.horizon,
           "n_fit": len(X_fit), "n_eval": len(X_eval), "geometries": {}}
    for name, g in (("raw", None), ("logit", G)):
        rec["geometries"][name] = res = sizing(X_fit, Y_fit, X_eval, Y_eval, H, g, mlp_epochs=args.mlp_epochs)
        print(name, json.dumps(res), flush=True)
    dest = Path(args.out) if args.out else ROOT / "results" / "drafter" / f"wide_sizing_{args.lang}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print("->", dest)


if __name__ == "__main__":
    main()
