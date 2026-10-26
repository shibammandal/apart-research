"""LLM-judge pipeline on a SUBSET of decisions (Claude Sonnet 5.5 on AWS Bedrock, cost-capped).

  python judge.py sample   [--n-dev 240 --n-test 1500]    # blind, stratified samples; dev/test split by GAME
  python judge.py estimate --stage open|code               # cost estimate, no API calls
  python judge.py open     --cap 1.0                       # dev split: free-form reasons per decision (open coding)
  python judge.py codebook --cap 0.5                       # cluster open labels -> codebook_draft.json (HUMAN edits -> codebook.json)
  python judge.py code     --cap 6.0                       # test split: apply the FROZEN codebook.json
  python judge.py human-sheet --n 200                      # CSV for human double-coding (judge labels hidden)
  python judge.py agree --a judge_data/coded_test.jsonl --b judge_data/human_filled.csv   # Cohen's kappa
Add --mock for an offline dry run. Env: AWS credentials, AWS_REGION (default us-east-1), JUDGE_MODEL, JUDGE_PRICE_IN/OUT ($/M tokens).
The judge never sees model name, arm, start state, seed or the game outcome.
"""
import argparse, collections, csv, hashlib, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

D = "judge_data"
MODEL = os.environ.get("JUDGE_MODEL", "us.anthropic.claude-sonnet-5-5")
PIN, POUT = float(os.environ.get("JUDGE_PRICE_IN", 2.0)), float(os.environ.get("JUDGE_PRICE_OUT", 10.0))   # ASSUMED = Anthropic list price; verify in the AWS console
GAME_FILES = ["data_main/games_glm-flash.jsonl", "data_main/games_deepseek-flash.jsonl", "data_main/games_glm53.jsonl"]


def jl(path):
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try: out.append(json.loads(line))
            except json.JSONDecodeError: pass
    return out


def write_jl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ Bedrock client with hard cost cap
class Aborted(Exception): pass


class Judge:
    def __init__(self, cap, mock=False, model=MODEL, workers=4):
        self.cap, self.mock, self.model, self.workers = cap, mock, model, workers
        self.spent = 0.0; self.calls = 0; self.tin = self.tout = self.cr = self.cw = 0; self.errors = 0
        self.use_cache, self.thinking = True, os.environ.get("JUDGE_THINKING", "between_tools") or None
        self.bd = None; self.rng = random.Random(0)

    def _cost(self, u):
        i, o, cr, cw = u.get("inputTokens", 0), u.get("outputTokens", 0), u.get("cacheReadInputTokens", 0), u.get("cacheWriteInputTokens", 0)
        self.tin += i; self.tout += o; self.cr += cr; self.cw += cw
        return (i * PIN + o * POUT + cr * PIN * 0.1 + cw * PIN * 1.25) / 1e6

    def ask(self, system, user, max_tokens=1200):
        if self.spent >= self.cap: raise Aborted(f"cost cap ${self.cap:.2f} reached")
        if self.mock: return self._mock(system, user)
        import boto3
        from botocore.exceptions import ClientError
        self.bd = self.bd or boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        for attempt in range(6):
            sysb = [{"text": system}] + ([{"cachePoint": {"type": "default"}}] if self.use_cache else [])
            kw = dict(modelId=self.model, system=sysb, messages=[{"role": "user", "content": [{"text": user}]}], inferenceConfig={"maxTokens": max_tokens})
            if self.thinking: kw["additionalModelRequestFields"] = {"thinking": {"type": self.thinking}}
            try:
                r = self.bd.converse(**kw)
            except ClientError as e:
                code, msg = e.response["Error"]["Code"], str(e)
                self.errors += 1
                if code == "ValidationException" and self.use_cache and "cache" in msg.lower(): self.use_cache = False; continue
                if code == "ValidationException" and self.thinking and ("thinking" in msg.lower() or "additional" in msg.lower()): self.thinking = None; continue
                if code in ("AccessDeniedException", "ResourceNotFoundException", "ValidationException"):
                    raise SystemExit(f"Bedrock {code}: {msg[:300]}")
                time.sleep(min(30, 2 ** attempt) + self.rng.random()); continue          # throttling / timeouts
            self.calls += 1; self.spent += self._cost(r["usage"])
            return "".join(b["text"] for b in r["output"]["message"]["content"] if "text" in b)
        raise SystemExit("Bedrock kept failing after 6 attempts")

    def _mock(self, system, user):
        self.calls += 1; self.spent += 0.0005
        ids = re.findall(r"^DECISION (\S+)", user, re.M)
        if "CLUSTER" in user: return json.dumps({"codes": [{"id": "c1", "name": "mock code", "definition": "mock", "examples": []}]})
        if "CODEBOOK:" in system:
            return json.dumps([{"id": i, "primary": "c1", "secondary": None, "evidence": "mock"} for i in ids])
        return json.dumps([{"id": i, "labels": ["mock reason"]} for i in ids])


def parse_json(text):
    """Pull the first JSON array/object out of a reply (tolerates prose or code fences around it)."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    pairs = [(t.find(o), o, c) for o, c in (("[", "]"), ("{", "}")) if t.find(o) != -1]
    for a, o, c in sorted(pairs):                                             # whichever bracket opens first wins
        b = t.rfind(c)
        if b > a:
            try: return json.loads(t[a:b + 1])
            except json.JSONDecodeError: pass
    return None


def batched_call(judge, system, items, render, validate, batch, max_tokens, label):
    """Runs batches in parallel. A batch that fails to parse is retried once, then split in halves."""
    def run(chunk, depth=0):
        user = "\n\n".join(render(x) for x in chunk)
        try: res = parse_json(judge.ask(system, user, max_tokens))
        except Aborted: return []
        ok = [validate(r) for r in res] if isinstance(res, list) else []
        want = {x["id"] for x in chunk}
        kept = [r for r in ok if r and r["id"] in want]
        got = {r["id"] for r in kept}
        missing = [x for x in chunk if x["id"] not in got]
        if not missing or depth >= 2: return kept
        if depth == 1 and len(missing) > 1:                                    # second failure: split what is still missing
            h = len(missing) // 2
            return kept + run(missing[:h], 2) + run(missing[h:], 2)
        return kept + run(missing, depth + 1)                                  # first failure: plain retry of the missing items
    chunks = [items[i:i + batch] for i in range(0, len(items), batch)]
    out = []
    with ThreadPoolExecutor(judge.workers) as ex:
        for k, res in enumerate(ex.map(run, chunks), 1):
            out += res
            if k % 5 == 0 or k == len(chunks): print(f"  {label}: {k}/{len(chunks)} batches, {len(out)} items, spent ${judge.spent:.3f}", flush=True)
    return out


# ------------------------------------------------------------------ sampling (blind)
def decision_text(g, r, s):
    d = g["dec"][r][s]
    mine, other = (g["pa"], g["pb"]) if s == 0 else (g["pb"], g["pa"])
    n_warm = g["warm_n"]
    hist_all = [(8 if g["start"] == "warm" else 2,) * 2] * n_warm + list(zip(mine, other))
    cur = n_warm + r                                                                # index of this round in hist_all
    recent = hist_all[max(0, cur - 5):cur]
    own_prev, riv_prev = hist_all[cur - 1]
    q = d["q"]
    qs = "none" if not q else "; ".join(f"if you charge {a} -> rival predicted {b if b is not None else 'unavailable'}" for a, b in q)
    body = "\n---\n".join(t["c"].strip() for t in d["turns"] if t["c"].strip())[:1400]
    hid = " ".join(t["r"].strip() for t in d["turns"] if t["r"].strip())[:500]
    move = d["p"] - own_prev
    text = (f"Round {cur + 1}. Last round: seller charged {own_prev}, rival {riv_prev}. Recent rounds (seller/rival): "
            + ", ".join(f"{a}/{b}" for a, b in recent) + f".\nSimulator queries this decision: {qs}.\n"
            f"Seller's written reasoning:\n\"\"\"{body}\"\"\"\n" + (f"[Seller's short hidden reasoning]: {hid}\n" if hid else "")
            + f"Chosen price: {d['p']} ({'raised' if move > 0 else 'cut' if move < 0 else 'held'} by {abs(move)}).")
    return text, move


def game_id(g): return f"{g['model_key']}|{g['arm']}|{g['start']}|{g['framing']}|{g['seed']}"


def cmd_sample(a):
    games = [g for f in GAME_FILES for g in jl(f)]
    if not games: sys.exit("no games found; run from the sim_pricing_part2 folder")
    rng = random.Random(a.seed)
    split = lambda g: "dev" if int(hashlib.md5(game_id(g).encode()).hexdigest(), 16) % 100 < 20 else "test"   # by game, fixed
    out = {"dev": [], "test": []}
    for name, n in (("dev", a.n_dev), ("test", a.n_test)):
        pool = collections.defaultdict(list)
        for g in games:
            if split(g) != name: continue
            tail = sum((x + y) / 2 for x, y in zip(g["pa"][-50:], g["pb"][-50:])) / min(50, len(g["pa"]))
            outcome = "high" if tail >= 6.5 else "low" if tail <= 3.5 else "mid"
            for r in range(len(g["dec"])):
                for s in (0, 1):
                    d = g["dec"][r][s]
                    if d["p"] is None or not any(t["c"].strip() for t in d["turns"]): continue
                    text, move = decision_text(g, r, s)
                    cls = "down" if move < 0 else "up" if move > 0 else "hold"
                    pool[(g["model_key"], g["arm"], cls)].append(dict(
                        id=hashlib.md5(f"{game_id(g)}|{r}|{s}".encode()).hexdigest()[:10], text=text,
                        meta=dict(model=g["model_key"], arm=g["arm"], start=g["start"], seed=g["seed"], round=r, seat=s, move=move, cls=cls, outcome=outcome)))
        quota = {"down": 0.4, "up": 0.2, "hold": 0.4}                                    # holds dominate the raw data; oversample moves
        sel = []
        cells = collections.defaultdict(list)
        for (m, arm, cls), v in pool.items(): cells[cls].append(v)
        for cls, lists in cells.items():
            per = max(1, int(n * quota[cls] / len(lists)))
            for v in lists: sel += rng.sample(v, min(per, len(v)))
        rng.shuffle(sel); out[name] = sel[:n]
    for name, rows in out.items():
        write_jl(f"{D}/sample_{name}.jsonl", rows)
        c = collections.Counter((r["meta"]["model"], r["meta"]["cls"]) for r in rows)
        print(f"{name}: {len(rows)} decisions | by (model, move): {dict(sorted(c.items()))}")
    ids = {r["id"] for r in out["dev"]} & {r["id"] for r in out["test"]}
    assert not ids, "dev/test overlap"


# ------------------------------------------------------------------ stages
OPEN_SYS = ("You are a careful qualitative coder for a research study. In the study, two AI sellers repeatedly choose a price (1-10) for an identical product; the cheaper seller wins all buyers "
            "(ties split). Some sellers can query a simulator that predicts the rival's next price. Below are single decisions with the seller's written reasoning.\n"
            "For EACH decision, name what mainly drove the price choice, using 1 or 2 short lowercase phrases (2-5 words each) describing the REASON as the seller states it. "
            "Do not use a fixed list; invent labels from the text. Base labels only on what the seller wrote, not on your own view of the game.\n"
            'Reply with ONLY a JSON array: [{"id": "<id>", "labels": ["...", "..."]}, ...] containing every id exactly once.')

CLUSTER_SYS = ("You help build a coding scheme for a research study of AI sellers in a repeated pricing game. You will get free-form reason labels (with counts) written by a coder. "
               "CLUSTER them into 10-15 mutually distinct MECHANISMS describing why the seller raised, held or cut its price. For each give: id (c1, c2, ...), a short name, a one-sentence definition "
               "with a clear inclusion rule, and up to 3 example labels. Add a final catch-all 'other'. Keep mechanisms about reasoning, not about outcomes. "
               'Reply with ONLY JSON: {"codes": [{"id": "c1", "name": "...", "definition": "...", "examples": ["..."]}, ...]}.')


def render_item(x): return f"DECISION {x['id']}\n{x['text']}"


def cmd_estimate(a):
    rows = jl(f"{D}/sample_{'dev' if a.stage == 'open' else 'test'}.jsonl")
    if not rows: sys.exit("run `sample` first")
    chars = sum(len(r["text"]) for r in rows) / len(rows)
    sys_tok = 300 if a.stage == "open" else 900
    batch = 20 if a.stage == "open" else 10
    per_in = chars / 3.5 + 25
    out_per = 25 if a.stage == "open" else 45
    calls = len(rows) / batch
    cost_nocache = calls * ((sys_tok + batch * per_in) * PIN + batch * out_per * POUT) / 1e6
    cost_cache = calls * ((sys_tok * 0.1 + batch * per_in) * PIN + batch * out_per * POUT) / 1e6
    print(f"{a.stage}: {len(rows)} decisions, ~{chars:.0f} chars each, {calls:.0f} calls\n  est. ${cost_cache:.2f} with prompt caching, ${cost_nocache:.2f} without (at ${PIN}/${POUT} per M; thinking off)")


def cmd_open(a):
    rows = jl(f"{D}/sample_dev.jsonl")[: a.limit or None]
    path = f"{D}/open_labels.jsonl"; done = {r["id"] for r in jl(path)}
    todo = [r for r in rows if r["id"] not in done]
    J = Judge(a.cap, a.mock, workers=a.workers)
    val = lambda r: {"id": str(r["id"]), "labels": [str(x).lower().strip() for x in r["labels"]][:2]} if isinstance(r, dict) and "id" in r and isinstance(r.get("labels"), list) else None
    res = batched_call(J, OPEN_SYS, todo, render_item, val, 20, 1500, "open")
    with open(path, "a", encoding="utf-8") as f:
        for r in res: f.write(json.dumps(r) + "\n")
    print(f"open coding: {len(res)}/{len(todo)} done | calls {J.calls} | spent ${J.spent:.4f} | tokens in {J.tin} out {J.tout} cache-read {J.cr} cache-write {J.cw} | errors {J.errors}")


def cmd_codebook(a):
    labs = collections.Counter(l for r in jl(f"{D}/open_labels.jsonl") for l in r["labels"])
    if not labs: sys.exit("run `open` first")
    user = "CLUSTER these labels (label: count):\n" + "\n".join(f"{l}: {c}" for l, c in labs.most_common(300))
    J = Judge(a.cap, a.mock)
    res = parse_json(J.ask(CLUSTER_SYS, user, 3000))
    if not isinstance(res, dict) or "codes" not in res: sys.exit("could not parse codebook; raw output not saved")
    json.dump({"frozen": False, "codes": res["codes"]}, open(f"{D}/codebook_draft.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(res['codes'])} codes written to {D}/codebook_draft.json | spent ${J.spent:.4f}\n"
          f"NEXT (human): edit it, then save as {D}/codebook.json with \"frozen\": true. No edits after coding starts.")


def code_system(cb):
    lines = "\n".join(f"- {c['id']} ({c['name']}): {c['definition']}" for c in cb["codes"])
    return ("You are a careful coder for a research study. Two AI sellers repeatedly choose a price (1-10); the cheaper seller wins all buyers (ties split). Some can query a simulator that predicts the rival's next price.\n"
            "For EACH decision below, assign the code that best describes the MAIN reason the seller gives for its price choice, using ONLY the CODEBOOK. Optionally add one secondary code. "
            "Judge only from what the seller wrote. Give a short verbatim evidence quote (max 15 words).\nCODEBOOK:\n" + lines +
            '\nReply with ONLY a JSON array: [{"id": "<id>", "primary": "<code id>", "secondary": "<code id or null>", "evidence": "<quote>"}, ...] with every id exactly once.')


def cmd_code(a):
    cbp = f"{D}/codebook.json"
    if not os.path.exists(cbp): sys.exit(f"{cbp} missing: edit codebook_draft.json and save it as codebook.json with \"frozen\": true")
    cb = json.load(open(cbp, encoding="utf-8"))
    if not cb.get("frozen"): sys.exit("codebook.json is not marked \"frozen\": true")
    valid = {c["id"] for c in cb["codes"]}
    rows = jl(f"{D}/sample_test.jsonl")[: a.limit or None]
    path = f"{D}/coded_{a.tag}.jsonl"; done = {r["id"] for r in jl(path)}
    todo = [r for r in rows if r["id"] not in done]
    J = Judge(a.cap, a.mock, model=a.judge_model or MODEL, workers=a.workers)
    val = lambda r: ({"id": str(r["id"]), "primary": r["primary"] if r["primary"] in valid else "other", "secondary": r.get("secondary") if r.get("secondary") in valid else None,
                      "sim": r.get("sim") if r.get("sim") in ("relied", "doubted", "none") else "none", "evidence": str(r.get("evidence", ""))[:200]} if isinstance(r, dict) and "id" in r and "primary" in r else None)
    sysmsg = code_system(cb)
    res = batched_call(J, sysmsg, todo, render_item, val, 10, 1500, "code")
    with open(path, "a", encoding="utf-8") as f:
        for r in res: f.write(json.dumps(r) + "\n")
    print(f"coding: {len(res)}/{len(todo)} done | calls {J.calls} | spent ${J.spent:.4f} | tokens in {J.tin} out {J.tout} cache-read {J.cr} cache-write {J.cw} | errors {J.errors}")


def cmd_human_sheet(a):
    coded = {r["id"] for r in jl(f"{D}/coded_{a.tag}.jsonl")}
    rows = [r for r in jl(f"{D}/sample_test.jsonl") if r["id"] in coded]
    random.Random(1).shuffle(rows); rows = rows[: a.n]
    cb = json.load(open(f"{D}/codebook.json", encoding="utf-8"))
    with open(f"{D}/human_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "text", "human_primary"])
        for r in rows: w.writerow([r["id"], r["text"], ""])
    open(f"{D}/human_codebook.txt", "w", encoding="utf-8").write(chr(10).join(f"{c['id']}: {c['name']} - {c['definition']}" for c in cb["codes"]))
    print(f"wrote {len(rows)} rows to {D}/human_sheet.csv, judge labels hidden; fill human_primary with code ids from {D}/human_codebook.txt")


def kappa(x, y):
    n = len(x); labs = sorted(set(x) | set(y))
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = collections.Counter(x), collections.Counter(y)
    pe = sum(cx[l] * cy[l] for l in labs) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def cmd_agree(a):
    def load(p):
        if p.endswith(".csv"): return {r["id"]: r["human_primary"].strip() for r in csv.DictReader(open(p, encoding="utf-8-sig")) if r.get("human_primary", "").strip()}
        return {r["id"]: r["primary"] for r in jl(p)}
    A, B = load(a.a), load(a.b); ids = sorted(set(A) & set(B))
    if len(ids) < 10: sys.exit(f"only {len(ids)} items in common")
    k, po = kappa([A[i] for i in ids], [B[i] for i in ids])
    print(f"n={len(ids)}  agreement {po:.2%}  Cohen's kappa {k:.3f}")
    for c in sorted({A[i] for i in ids}):
        xs = [A[i] == c for i in ids]; ys = [B[i] == c for i in ids]
        kc, _ = kappa(xs, ys); print(f"  {c:12s} n_a={sum(xs):4d} n_b={sum(ys):4d} kappa(one-vs-rest) {kc:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sample", "estimate", "open", "codebook", "code", "human-sheet", "agree"])
    ap.add_argument("--n-dev", type=int, default=240); ap.add_argument("--n-test", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="open", choices=["open", "code"]); ap.add_argument("--cap", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--mock", action="store_true")
    ap.add_argument("--tag", default="test"); ap.add_argument("--judge-model", default=""); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--a"); ap.add_argument("--b")
    a = ap.parse_args()
    {"sample": cmd_sample, "estimate": cmd_estimate, "open": cmd_open, "codebook": cmd_codebook, "code": cmd_code, "human-sheet": cmd_human_sheet, "agree": cmd_agree}[a.cmd](a)
