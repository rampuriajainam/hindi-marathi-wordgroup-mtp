"""
Tree speculative decoding (OM, issue #33): verify a whole draft tree from a TreePolicy
(mtp/eval/tree_policy.py, INTERFACES §11) in ONE forward pass. Output == greedy decoding with head 0.

One step, at the last emitted position:
  1. t0 = head 0's argmax (always kept; the tree's root).
  2. nodes = policy.tree(...): (parent, d, rank) = head d's rank-th token as the child of `parent`
     (-1 = t0). Every head drafts from the same last position (parallel / Medusa heads).
  3. ONE pass over [t0, node_1 .. node_N] on top of the KV cache, with a tree attention mask (each
     token sees the prefix and its own ancestors) and positions prefix_len + depth.
  4. Walk from t0: while head 0's argmax at the current node equals one of its children, move there.
     The walked path is emitted (t0 + accepted drafts); head 0's argmax at its last node is the next
     step's t0, and every head's logits there seed the next tree.
  5. The KV cache keeps the prefix + the walked path only (tree entries are cropped, the path's
     keys/values re-appended in order, so positions stay consecutive).

Sequential heads (head_type "seq", #49): a node's token depends on its path. Node (p, d, r) is the
r-th token of head d applied to its parent's state and token (the root's state is the hidden state at
the last position, its token t0), exactly as draft_chain does along one path. All nodes of one depth
are drafted in one batched head call, so a tree costs k-1 head calls like the chain. The verify pass
runs head 0 only (extra_heads=False). Ranks therefore mean "rank under head d given the path", which
is what collect_ranks / fit_tree measure on a seq model (teacher-forced forward).

Two other walks (step 4), both off by default:
  - typical acceptance (policy.accept_eps, TypicalTree in typical.py; LOSSY): when the node's children
    miss head 0's argmax, follow the child with the highest p0 among those passing typical_threshold.
  - sampling (policy.sample = (T, top_p), SampledTree in sampling.py; lossless for SAMPLING): t0 and the
    token after each path node are drawn from head 0 with the uniform of their generated position, so the
    output equals plain sampling (greedy_generate(..., sample=...)) with the same seed.
"""

import contextlib
import time

import torch

from mtp.eval.sampling import SampledTree, position_uniforms, sample_token
from mtp.eval.spec_decode import _no_eos, _step_state, _sync, context_cache, group_integrity_counts
from mtp.eval.tree_policy import StaticTree
from mtp.eval.typical import TypicalTree, typical_threshold


def _static(policy):
    """True for a StaticTree, also inside decode-mode and prompt-lookup wrappers: its tree needs no head logits
    (before PromptLookup was unwrapped here, seq-head lookup variants computed an unused draft chain every step)."""
    from mtp.eval.lexicon import LexiconLookup
    from mtp.eval.penalty import PenalizedTree
    from mtp.eval.prompt_lookup import PromptLookup

    while isinstance(policy, (TypicalTree, SampledTree, PromptLookup, PenalizedTree, LexiconLookup)):
        policy = policy.base
    return isinstance(policy, StaticTree)


def _seq_draft_tokens(model, h, t0, nodes, ctx=None):
    """Node tokens for sequential heads, batched per depth. h: hidden state [H] at the last position.
    ctx: ContextCache for seq_ctx heads (the slot's context read is shared by every node at its depth)."""
    embed, lm_head = model.base_model.get_input_embeddings(), model._lm_head()
    reads = model.ctx_reads(h, ctx) if ctx is not None else None
    # seq_wide heads draft in a wider state (seq_state / seq_step / seq_read); plain seq heads in h itself
    step = getattr(model, "seq_step", None) or (lambda d, s, e: model.extra_heads[d - 1](s, e))
    read = getattr(model, "seq_read", None) or (lambda s: s)
    state = {-1: (model.seq_state(h) if hasattr(model, "seq_state") else h, t0)}   # node -> (head state, token)
    tokens = [None] * len(nodes)
    for d in sorted({n[1] for n in nodes}):
        idx = [i for i, n in enumerate(nodes) if n[1] == d]
        parents = sorted({nodes[i][0] for i in idx})
        s_par = torch.stack([state[p][0] for p in parents])
        e_par = embed(torch.tensor([state[p][1] for p in parents], device=h.device)).to(s_par.dtype)
        s_new = step(d, s_par, e_par)
        if reads is not None:
            s_new = s_new + reads[d - 1].to(s_new.dtype)
        top = lm_head(read(s_new)).topk(max(nodes[i][2] for i in idx) + 1, dim=-1).indices.tolist()   # one sync per depth
        for i in idx:
            j = parents.index(nodes[i][0])
            tokens[i] = top[j][nodes[i][2]]
            state[i] = (s_new[j], tokens[i])
    return tokens


def _draft_tokens(nodes, head_logits):
    """tree_policy.draft_tokens with one topk and one device sync for all heads (parallel heads)."""
    ds = sorted({d for _, d, _ in nodes})
    top = torch.stack([head_logits[d] for d in ds]).topk(max(r for _, _, r in nodes) + 1, dim=-1).indices.tolist()
    row = {d: i for i, d in enumerate(ds)}
    return [top[row[d]][r] for _, d, r in nodes]


def _tree_layout(nodes):
    """Tree rows: 0 = t0, i+1 = node i. Returns (parent row per row, depth per row)."""
    parents, depths = [-1], [0]
    for parent, d, _ in nodes:
        parents.append(parent + 1 if parent >= 0 else 0)
        depths.append(d)
    return parents, depths


def _tree_mask(parents, prefix_len, device):
    """[1, 1, M, prefix_len + M] bool: row i attends to the prefix, its ancestors and itself."""
    m = len(parents)
    rows = []                                                       # built in Python, one copy to the device
    for i in range(m):
        row = [False] * m
        j = i
        while j >= 0:
            row[j] = True
            j = parents[j]
        rows.append(row)
    tree = torch.tensor(rows, dtype=torch.bool)
    allow = torch.cat([torch.ones(m, prefix_len, dtype=torch.bool), tree], dim=1)
    return allow[None, None].to(device)


def _graft(tokens, parents, depths, chains, max_depth, at=0):
    """Add token chains under row `at` (default: the root) in place, down to depth max_depth at most. A chain
    token equal to an existing child of the same node reuses that row, so only new tokens become rows."""
    kids = {}
    for row in range(1, len(tokens)):
        kids.setdefault(parents[row], {})[tokens[row]] = row
    for chain in chains:
        cur = at
        for tok in chain[:max(max_depth - depths[at], 0)]:
            nxt = kids.get(cur, {}).get(tok)
            if nxt is None:
                nxt = len(tokens)
                tokens.append(tok)
                parents.append(cur)
                depths.append(depths[cur] + 1)
                kids.setdefault(cur, {})[tok] = nxt
            cur = nxt


def _keep_path(cache, n_tree, rows):
    """Drop the last n_tree cache entries except `rows` (tree row indices, in path order), which are
    re-appended so the cache = prefix + path."""
    idx = None
    for layer in cache.layers:
        stored = layer.keys.shape[-2]
        if stored != layer.get_seq_length():
            raise NotImplementedError("tree decoding past the sliding window is not supported")
        if idx is None:
            idx = torch.tensor(rows, device=layer.keys.device) + (stored - n_tree)
        k, v = layer.keys.index_select(-2, idx), layer.values.index_select(-2, idx)
        layer.crop(-n_tree)
        layer.update(k, v)


@torch.no_grad()
def generate_tree(model, tokenizer, prompt_ids, max_new_tokens, policy, grouper=None, *, use_cache=True, amp=None,
                  ignore_eos=False, trace=None, sample=None, sample_seed=0):
    """Returns (generated ids, stats); output == greedy decoding with head 0 (== generate()).

    stats: as generate() (tokens, forward_passes, steps, mean_accepted_len, proposed / accepted per
    head, seconds, tokens_per_sec, gi_hits / gi_spans, spans) plus nodes (tree nodes verified per step).
    proposed[d-1] counts steps whose tree had a node at depth d; accepted[d-1] steps that accepted one.
    With policy.accept_eps (lossy) also lossy_accepts: accepted drafts that were not head 0's argmax.
    sample: (temperature, top_p), default policy.sample; the output then == greedy_generate(..., sample=
    sample, sample_seed=sample_seed) instead of greedy. Cannot be combined with accept_eps.
    trace: optional list, filled with one dict per step (nodes, accepted, start, head_logits, tokens, path,
    verify_logits) for tests.
    """
    if not use_cache:
        raise ValueError("generate_tree needs the KV cache (use_cache=True)")
    eps = getattr(policy, "accept_eps", None)
    delta = getattr(policy, "accept_delta", None)
    sample = sample if sample is not None else getattr(policy, "sample", None)
    if sample is not None and eps is not None:
        raise ValueError("generate_tree: sampling and typical acceptance cannot be combined")
    penalty = getattr(policy, "penalty", None)     # repetition-penalised greedy (mtp/eval/penalty.py)
    if penalty is not None and (sample is not None or eps is not None):
        raise ValueError("generate_tree: the penalty mode cannot be combined with sampling or typical acceptance")
    if penalty is not None:
        from mtp.eval.penalty import apply_penalty
    u = position_uniforms(sample_seed, max_new_tokens) if sample is not None else None
    seq_heads = getattr(model, "head_type", None) == "seq"
    fwd = {"extra_heads": False} if seq_heads else {}
    device = next(model.parameters()).device
    amp = amp or contextlib.nullcontext
    k = model.num_heads
    ban = tokenizer.eos_token_id if ignore_eos else None
    eos = None if ignore_eos else tokenizer.eos_token_id
    seq = [int(t) for t in prompt_ids]
    new, spans, nodes_per_step = [], [], []
    proposed, accepted = [0] * (k - 1), [0] * (k - 1)
    passes = steps = lossy = 0

    _sync(device)
    t_start = time.perf_counter()
    with amp():
        out = model(torch.tensor([seq], device=device), use_cache=True, **fwd)
        passes += 1
        cache = out.aux["past_key_values"]
        ctx = context_cache(model, out.hidden[0])
        pos = -1                                   # row (in out's positions) of the last emitted token
        while len(new) < max_new_tokens:
            last = [lg[0, pos] for lg in out.logits]
            if sample is None and penalty is not None:
                t0 = int(apply_penalty(_no_eos(last[0].float(), ban), seq, penalty).argmax())
            elif sample is None:
                t0 = int(_no_eos(last[0], ban).argmax())
            else:              # the draw for this generated position (== the one that ended the last walk)
                t0 = sample_token(_no_eos(last[0], ban), u[len(new)], *sample)
            if seq_heads:      # the policy sees the greedy chain's logits (EntropyTree cells); StaticTree needs none
                h = out.hidden[0, pos]
                chain = [] if _static(policy) else model.draft_chain(
                    h, t0, k - 1, return_logits=True, **({"ctx": ctx} if ctx is not None else {}))[1]
                last = last[:1] + chain
            remaining = max_new_tokens - len(new)
            nodes = [] if t0 == eos else [n for n in policy.tree(last, torch.tensor(seq), _step_state(
                out, pos, tokenizer, grouper, steps)) if n[1] < k and n[1] <= remaining - 1]
            if not nodes and (t0 == eos or remaining == 1):
                new.append(t0)                     # last token: no verification pass needed
                spans.append(1)
                nodes_per_step.append(0)
                steps += 1
                break

            tokens = [t0] + (_seq_draft_tokens(model, h, t0, nodes, ctx) if seq_heads else _draft_tokens(nodes, last))
            parents, depths = _tree_layout(nodes)
            chains = policy.extra_chains(seq + [t0]) if t0 != eos and hasattr(policy, "extra_chains") else []
            n_head_rows = len(tokens)
            if chains:                             # copied n-gram chains (prompt_lookup.py) under the root
                _graft(tokens, parents, depths, chains, remaining - 1)
            if t0 != eos and hasattr(policy, "extend"):    # lexicon completions under head nodes (lexicon.py)
                for row, ext in policy.extend(seq, tokens, parents, depths, n_head_rows):
                    _graft(tokens, parents, depths, ext, remaining - 1, at=row)
            prefix_len = cache.get_seq_length()
            m = len(tokens)
            out = model(torch.tensor([tokens], device=device), past_key_values=cache, use_cache=True,
                        attention_mask=_tree_mask(parents, prefix_len, device),
                        position_ids=torch.tensor([[prefix_len + d for d in depths]], device=device),
                        cache_position=torch.arange(prefix_len, prefix_len + m, device=device), **fwd)
            passes += 1
            verify = _no_eos(out.logits[0][0], ban)                      # row i: head 0 after tokens[i]
            children = {}
            for row in range(1, m):
                children.setdefault(parents[row], {})[tokens[row]] = row
            path, lossy_at = [0], []        # lossy_at: path indices accepted without being the argmax
            if sample is not None:          # draw at generated position len(new) + len(path); the
                while children.get(path[-1]):   # draw that misses becomes the next t0 (same row, same u)
                    nxt = children[path[-1]].get(sample_token(verify[path[-1]], u[len(new) + len(path)], *sample))
                    if nxt is None:
                        break
                    path.append(nxt)
            else:
                greedy = verify.argmax(-1).tolist() if penalty is None else None
                while True:
                    kids = children.get(path[-1], {})
                    if greedy is not None:
                        want = greedy[path[-1]]
                    elif kids:   # penalised argmax with this node's own context: prefix + the path so far
                        want = int(apply_penalty(verify[path[-1]].float(), seq + [tokens[r] for r in path],
                                                 penalty).argmax())
                    else:
                        break
                    nxt = kids.get(want)
                    if nxt is None and eps is not None and kids:      # typical acceptance (lossy)
                        p, thr = typical_threshold(verify[path[-1]], eps, delta)
                        toks = list(kids)
                        pk = p[toks].tolist()
                        best = max(range(len(toks)), key=pk.__getitem__)      # ties: first child
                        if pk[best] >= thr:
                            nxt = kids[toks[best]]
                            lossy_at.append(len(path))
                    if nxt is None:
                        break
                    path.append(nxt)
            emitted = [tokens[r] for r in path]
            if eos in emitted:
                path = path[: emitted.index(eos) + 1]
                emitted = emitted[: len(path)]
            lossy += sum(i < len(path) for i in lossy_at)
            for d in range(1, k):
                proposed[d - 1] += any(dd == d for dd in depths[1:])
                accepted[d - 1] += d < len(path)
            if trace is not None:
                trace.append({"nodes": [tuple(n) for n in nodes], "accepted": len(path) - 1, "start": len(new),
                              "head_logits": [x.float().cpu() for x in last], "tokens": tokens,
                              "hidden": h.float().cpu() if seq_heads else None, "path": list(path),
                              "verify_logits": verify.float().cpu()})
            _keep_path(cache, m, path)
            if ctx is not None:
                ctx.append(*model.ctx_project(out.hidden[0, path]))
            seq += emitted
            new += emitted
            spans.append(len(emitted))
            nodes_per_step.append(m - 1)
            steps += 1
            pos = path[-1]
            if new[-1] == eos:
                break
    _sync(device)
    seconds = time.perf_counter() - t_start

    gi_hits, gi_spans = group_integrity_counts(prompt_ids, new, spans, tokenizer, grouper)
    return new, {
        "tokens": len(new), "forward_passes": passes, "steps": steps,
        "mean_accepted_len": len(new) / max(steps, 1),
        "proposed": proposed, "accepted": accepted,
        "seconds": seconds, "tokens_per_sec": len(new) / seconds if seconds > 0 else 0.0,
        "gi_hits": gi_hits, "gi_spans": gi_spans, "spans": spans, "nodes": nodes_per_step,
        **({"lossy_accepts": lossy} if eps is not None else {}),
    }
