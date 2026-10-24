"""Part 2 runner.

  python run.py ping    --model-key glm-flash              # 1 real call: checks URL, auth, reasoning field, usage, cost
  python run.py play    --model-key glm-flash --rounds 12 --seeds 1 --cap 0.30     # smoke test
  python run.py play    --model-key glm-flash --rounds 100 --seeds 3 --cap 3.0     # benchmark
  python run.py play    --mock ...                         # offline, free
  python run.py summary --out data
Secrets: put them in .env (see .env.example) or environment variables. Never in code or chat.
"""
import argparse, asyncio, json, os, sys, time
from llm import LLM, MockLLM, Budget, Aborted, Fatal, load_env
from engine import ARMS, play_game, COOP, NASH

MODELS = {   # price $/M tokens (prompt, completion), from the Modal shared-endpoint list. Cached-prompt discount ignored (upper bound).
    "glm-flash":      dict(id="nvidia/GLM-5.3-Flash-NVFP4",       pin=0.15, pout=0.50),
    "deepseek-flash": dict(id="deepseek-ai/DeepSeek-V4.1-Flash",  pin=0.30, pout=1.20),
    "glm53":          dict(id="zai-org/GLM-5.3",                  pin=1.40, pout=4.40),
}


def build_llm(args, budget):
    if args.mock: return MockLLM(budget, seed=args.seed0)
    m = MODELS[args.model_key]
    suffix = args.model_key.upper().replace("-", "_")
    url = os.environ.get(f"ENDPOINT_URL_{suffix}") or os.environ.get("ENDPOINT_URL")
    if not url: sys.exit(f"Set ENDPOINT_URL_{suffix} (or ENDPOINT_URL) in .env")
    headers = {}
    if os.environ.get("API_KEY"): headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
    if os.environ.get("MODAL_PROXY_KEY") and os.environ.get("MODAL_PROXY_SECRET"):
        headers["Modal-Key"], headers["Modal-Secret"] = os.environ["MODAL_PROXY_KEY"], os.environ["MODAL_PROXY_SECRET"]
    if os.environ.get("EXTRA_HEADERS"): headers.update(json.loads(os.environ["EXTRA_HEADERS"]))
    if not headers: sys.exit("No credentials found: set MODAL_PROXY_KEY + MODAL_PROXY_SECRET (or API_KEY) in .env")
    return LLM(url, os.environ.get("MODEL_ID") or m["id"], m["pin"], m["pout"], budget, headers,
               max_tokens=args.max_tokens, temperature=args.temperature, concurrency=args.concurrency,
               extra_body={} if args.effort == "default" else {"reasoning_effort": args.effort})


def load_games(path):
    """Reads finished games; ignores a truncated last line left by a crash."""
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try: out.append(json.loads(line))
            except json.JSONDecodeError: pass
    return out


def key(s, rounds): return (s["model_key"], s["arm"], s["start"], s["framing"], s["seed"], rounds)


async def cmd_ping(args):
    b = Budget(0.05); llm = build_llm(args, b); await llm.open()
    try:
        t = time.time()
        msgs = [{"role": "user", "content": "Reply with exactly: PRICE: 7"}]
        if args.mock: r = await llm.chat(msgs)
        else:
            raw = await llm.client.post(llm.url, headers=llm.headers, json={"model": llm.model, "messages": msgs, "max_tokens": 200})
            print("HTTP", raw.status_code, f"{time.time() - t:.1f}s")
            if raw.status_code != 200: print(raw.text[:500]); return
            j = raw.json(); msg = j["choices"][0]["message"]
            print("message keys:", sorted(msg.keys()), "| usage:", j.get("usage"))
            print("finish_reason:", j["choices"][0].get("finish_reason"))
            r = await llm.chat(msgs, max_tokens=200)
        print("content:", repr(r["content"][:200])); print("reasoning present:", bool(r["reasoning"]), "| len", len(r["reasoning"]))
        print(f"tokens in/out {r['tin']}/{r['tout']}  est. cost ${b.spent:.6f}  (temperature accepted: {getattr(llm, "send_temperature", True)})")
    except (Fatal, Aborted) as e:
        sys.exit(f"STOP: {e}")
    finally: await llm.close()


async def cmd_play(args):
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"games_{'mock' if args.mock else args.model_key}.jsonl")
    done = {key(g, g["rounds"]) for g in load_games(path)}
    budget = Budget(args.cap); llm = build_llm(args, budget); await llm.open()
    arms, starts = args.arms.split(","), args.starts.split(",")
    for a in arms:
        if a not in ARMS: sys.exit(f"unknown arm {a}; choose from {ARMS}")
    specs = [dict(model_key="mock" if args.mock else args.model_key, arm=a, start=st, framing=args.framing, seed=s)
             for s in range(args.seed_from, args.seed_from + args.seeds) for st in starts for a in arms]
    todo = [s for s in specs if key(s, args.rounds) not in done]
    print(f"{len(specs)} games planned, {len(specs) - len(todo)} already done, {len(todo)} to run | cap ${args.cap:.2f} | {args.rounds} rounds")
    gsem, cnt, t0, lock = asyncio.Semaphore(args.games_parallel), {"n": 0}, time.time(), asyncio.Lock()

    def prog(s, rnd, cur):
        print(f"      ... {s['arm']} {s['start']} seed {s['seed']} round {rnd}/{args.rounds}  calls {cur['calls']}  {time.time() - t0:.0f}s", flush=True)

    async def one(s):
        async with gsem:
            if llm.fatal: return
            try: rec = await play_game(llm, s, args.rounds, prog)
            except Aborted: return
            except Fatal: return
        async with lock:
            with open(path, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n"); f.flush()
            cnt["n"] += 1
            print(f"  [{cnt['n']}/{len(todo)}] {s['arm']:9s} {s['start']} seed {s['seed']}  last10 {sum(rec['pa'][-10:] + rec['pb'][-10:]) / 20:.1f}"
                  f"  fails {rec['fails']} cuts {rec.get('length_cuts', 0)} simfail {rec.get('sim_fail', 0)}  ${rec['cost_usd']:.3f}  total ${budget.spent:.3f}  {time.time() - t0:.0f}s")
    try: await asyncio.gather(*(one(s) for s in todo))
    finally: await llm.close()
    print(f"done {cnt['n']}/{len(todo)} | spent ${budget.spent:.4f} | calls {budget.calls} | errors {budget.errors} retries {budget.retries} gave_up {budget.gave_up}"
          f" | usage-estimated calls {budget.estimated}")
    if llm.fatal: sys.exit(f"STOPPED: {llm.fatal}")
    if cnt["n"] < len(todo): print("Stopped early (budget cap). Re-run the same command to resume; finished games are kept.")


def cmd_summary(args):
    import pandas as pd
    rows = []
    for fn in sorted(os.listdir(args.out)):
        if not fn.startswith("games_") or not fn.endswith(".jsonl"): continue
        for g in load_games(os.path.join(args.out, fn)):
            pa, pb = g["pa"], g["pb"]; both = [(x + y) / 2 for x, y in zip(pa, pb)]
            last = both[-50:]
            sus = any(all(x >= COOP and y >= COOP for x, y in zip(pa[i:i + 5], pb[i:i + 5])) for i in range(len(pa) - 4))
            wars = sum(1 for s in (pa, pb) for i in range(1, len(s)) if s[i - 1] - s[i] >= 3)
            nq = sum(len(d["q"]) for ds in g["dec"] for d in ds)
            rows.append(dict(model=g["model_key"], arm=g["arm"], start=g["start"], framing=g["framing"], price_last=sum(last) / len(last),
                             coll=(sum(last) / len(last) - NASH) / (10 - NASH), sustained8=sus, wars=wars, fails=g["fails"], fail_rate=g.get("fail_rate", 0),
                             cuts=g.get("length_cuts", 0) / max(1, g.get("turns", 1)), sim_fail=g.get("sim_fail", 0), bad=g.get("fail_rate", 0) > 0.05,
                             queries=nq, sim_mae=g.get("sim_mae"), cost=g["cost_usd"]))
    if not rows: sys.exit("no games found")
    df = pd.DataFrame(rows)
    t = df.groupby(["model", "start", "framing", "arm"]).agg(n=("cost", "size"), price_last=("price_last", "mean"), coll=("coll", "mean"),
                                                         sustained8=("sustained8", "mean"), wars=("wars", "mean"), fail_rate=("fail_rate", "mean"), cuts=("cuts", "mean"), sim_fail=("sim_fail", "mean"), bad_games=("bad", "sum"),
                                                         queries=("queries", "mean"), sim_mae=("sim_mae", "mean"), cost=("cost", "sum"))
    print(t.round(2).to_string()); print(f"\ntotal logged cost ${df.cost.sum():.3f} over {len(df)} games")


if __name__ == "__main__":
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ping", "play", "summary"])
    ap.add_argument("--model-key", default="glm-flash", choices=list(MODELS)); ap.add_argument("--mock", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--starts", default="warm"); ap.add_argument("--framing", default="neutral")
    ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--seed-from", type=int, default=0); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=12); ap.add_argument("--cap", type=float, default=0.30)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "default"], help="reasoning_effort sent to the endpoint"); ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=48, help="simultaneous HTTP requests"); ap.add_argument("--games-parallel", type=int, default=12)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    if a.cmd == "summary": cmd_summary(a)
    else: asyncio.run({"ping": cmd_ping, "play": cmd_play}[a.cmd](a))
