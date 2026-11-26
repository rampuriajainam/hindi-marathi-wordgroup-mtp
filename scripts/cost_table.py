"""Cost curve: per policy tok/step, speed-up, verified positions per emitted token, mean GPU power, energy per token."""
import csv
import datetime as dt
import glob
import json
import sys

root = sys.argv[1]
power = {0: [], 1: []}
for row in csv.reader(open(f"{root}/power.csv", encoding="utf-8")):
    try:
        t = dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
        power[int(row[1])].append((t, float(row[2]), float(row[3])))
    except (ValueError, IndexError):
        continue


def mean_power(gpu, wall):
    xs = [(p, u) for t, p, u in power[gpu] if wall[0] <= t <= wall[1]]
    return (sum(p for p, _ in xs) / len(xs), sum(u for _, u in xs) / len(xs), len(xs)) if xs else (None, None, 0)


out = {}
for p in sorted(glob.glob(f"{root}/results*/*/eval_*.json")):
    r = json.load(open(p, encoding="utf-8"))
    gpu = 1 if "r8_mr" in r["run_name"] else 0
    half = bool(r.get("half"))
    print("\n##", r["run_name"], "half" if half else "autocast+fast_linear")
    rows = []
    e0 = r["spec_decode"][0]
    gw, gu, n = mean_power(gpu, e0["greedy_wall"])
    gtps = e0["greedy_tokens_per_sec"]
    rows.append(dict(policy="greedy", tok_step=1.0, speedup=1.0, tps=gtps, pos_per_tok=1.0, watts=gw, util=gu,
                     j_per_tok=gw / gtps if gw else None))
    for e in r["spec_decode"]:
        w, u, n = mean_power(gpu, e["wall"])
        pos = (e.get("mean_tree_nodes", 3) + 1) / e["mean_accepted_len"]
        rows.append(dict(policy=e["policy"], tok_step=e["mean_accepted_len"], speedup=e["speedup"], tps=e["tokens_per_sec"],
                         pos_per_tok=pos, watts=w, util=u, j_per_tok=w / e["tokens_per_sec"] if w else None,
                         match=e["match_rate"], fp32=(e.get("fp32_check") or {}).get("match_rate")))
    for x in rows:
        print("  %-26s tok/step %.2f  x%.2f  %5.1f tok/s  pos/tok %5.1f  %5.1f W  util %3.0f%%  %.2f J/tok (x%.2f)  fp32 %s" % (
            x["policy"], x["tok_step"], x["speedup"], x["tps"], x["pos_per_tok"], x["watts"] or 0, x["util"] or 0,
            x["j_per_tok"] or 0, (x["j_per_tok"] or 0) / rows[0]["j_per_tok"] if rows[0]["j_per_tok"] else 0, x.get("fp32")))
    out[r["run_name"] + ("_half" if half else "")] = rows
json.dump(out, open(f"{root}/cost_table.json", "w", encoding="utf-8"), indent=1)
