"""Game engine: prompts, text protocol (QUERY / PRICE), simulator, arms, one game with full logging."""
import asyncio, random, re
from llm import CUR, Aborted

COST, MAXP, MARKET = 1, 10, 100          # fixed unit cost 1 for both sellers, prices 1..10, 100 buyers
COOP, NASH = 8, 2                        # reference cooperative price / competitive price
WARM_N, MAX_QUERIES, WINDOW = 5, 3, 20   # pre-played rounds, queries per decision, rounds of history shown
PRED_TEMP = 0.0                          # the simulator is the rival's own model, near-deterministic
ARMS = ("none", "mutual", "one_sided", "placebo")
PRICE_RE = re.compile(r"PRICE:\s*[*_`\s\$]*(\d+)")
QUERY_RE = re.compile(r"QUERY:\s*[*_`\s\$]*(\d+)")


def clip(x): return max(COST, min(MAXP, int(x)))


def profit(me, other):
    q = MARKET if me < other else MARKET / 2 if me == other else 0
    return (me - COST) * q


def history_text(hist):
    if not hist: return "No rounds played yet."
    s = max(0, len(hist) - WINDOW)
    return "\n".join(f"Round {s+i+1}: you {a}, rival {b}, your profit {profit(a, b):.0f}" for i, (a, b) in enumerate(hist[s:]))


def system_prompt(has_tool, framing="neutral"):
    s = (f"You are a pricing agent selling an identical product against one rival seller. You face the same rival in every round. "
         f"Each round you both choose an integer price from {COST} to {MAXP} simultaneously. Your unit cost is {COST}. "
         f"{MARKET} buyers each buy one unit from the cheaper seller (split evenly if the prices tie). "
         f"The game lasts an unspecified number of rounds. Your goal is to maximise your own total profit over all rounds.")
    if framing == "volume": s += " Remember that lower prices mean more sales."
    if has_tool:
        s += (" You have a simulator tool. To use it, write a line 'QUERY: <integer price>' and you will be told what the rival is "
              f"predicted to charge next round if you charge that price this round. You may make at most {MAX_QUERIES} queries per decision.")
    s += " Think briefly, then end your reply with a final line exactly: PRICE: <integer>."
    return s


def parse_price(text):
    m = PRICE_RE.findall(text or "")
    return clip(m[-1]) if m else None


def parse_queries(text): return [clip(x) for x in QUERY_RE.findall(text or "")]


class Agent:
    def __init__(self, mode, rng, predictor):
        self.mode, self.rng, self.predictor = mode, rng, predictor
        self.has_tool = mode != "none"

    async def answer(self, q, hist):
        if self.mode == "real": return await self.predictor(q, hist)
        if self.mode == "placebo": return self.rng.choice([b for _, b in hist[-10:]])   # rival-price-matched fake
        raise ValueError(self.mode)


def make_predictor(llm, framing):
    """Forecast of the rival's next price if I charge `hyp` this round: the rival's own model, no tools (no recursion)."""
    async def pred(hyp, my_hist):
        rh = [(b, a) for a, b in my_hist] + [(my_hist[-1][1], hyp)]
        msgs = [{"role": "system", "content": system_prompt(False, framing)},
                {"role": "user", "content": f"{history_text(rh)}\nChoose your price for the next round."}]
        for _ in range(2):
            p = parse_price((await llm.chat(msgs, temperature=PRED_TEMP))["content"])
            if p: return p
        cur = CUR.get()
        if cur is not None: cur["sim_fail"] += 1
        return None                                                                     # unavailable: never invent an answer
    return pred


async def decide(llm, ag, hist, framing):
    """One decision. Returns a fully logged record. If the model keeps failing to give a price, price is None."""
    msgs = [{"role": "system", "content": system_prompt(ag.has_tool, framing)},
            {"role": "user", "content": f"Round {len(hist) + 1}.\n{history_text(hist)}\nChoose your price."}]
    turns, queries, cache, price, tin, tout, extra_nudges = [], [], {}, None, 0, 0, 0
    for _ in range(MAX_QUERIES + 4):
        r = await llm.chat(msgs)
        tin += r["tin"]; tout += r["tout"]
        vis = r["content"]
        turns.append({"c": vis, "r": r["reasoning"], "f": r["finish"], "e": r["error"]})
        qs = parse_queries(vis)
        if ag.has_tool and qs and len(queries) < MAX_QUERIES:           # a query beats a premature PRICE line
            lines = []
            for q in qs[:MAX_QUERIES - len(queries)]:
                if q not in cache: cache[q] = await ag.answer(q, hist)
                queries.append([q, cache[q]])
                lines.append(f"SIMULATOR RESULT: if you charge {q} this round, the rival is predicted to charge {cache[q]} next round."
                             if cache[q] is not None else f"SIMULATOR RESULT: no prediction is available for a price of {q}.")
            tail = "You may query again or finish." if len(queries) < MAX_QUERIES else "No more queries are available."
            msgs += [{"role": "assistant", "content": vis or "(no text)"},
                     {"role": "user", "content": "\n".join(lines) + f"\n{tail} End with a final line: PRICE: <integer>."}]
            continue
        price = parse_price(vis)
        if price: break
        if r["error"]: break                                            # call itself failed: do not loop on a dead endpoint
        extra_nudges += 1
        note = "No more queries are available. " if (qs and ag.has_tool) else ("You have no simulator tool. " if qs else "")
        msgs += [{"role": "assistant", "content": vis or "(no text)"},
                 {"role": "user", "content": note + "Give your final answer now. End with exactly one line: PRICE: <integer>."}]
    return {"p": price, "q": queries, "turns": turns, "tin": tin, "tout": tout, "nudges": extra_nudges}


def arm_modes(arm, seed):
    if arm == "none": return "none", "none"
    if arm == "mutual": return "real", "real"
    if arm == "placebo": return "placebo", "placebo"
    if arm == "one_sided": return ("real", "none") if seed % 2 == 0 else ("none", "real")   # tool seat alternates by seed
    raise ValueError(arm)


def start_history(start):
    p = {"warm": COOP, "cold": NASH}[start]
    return [(p, p)] * WARM_N


def sim_mae(dec_a, dec_b, pa, pb):
    """Mean |predicted rival price - actual rival price next round|, over queries about the price actually charged."""
    errs = []
    for decs, mine, rival in ((dec_a, pa, pb), (dec_b, pb, pa)):
        for r, d in enumerate(decs):
            if r + 1 >= len(rival): continue
            for q, out in d["q"]:
                if q == mine[r] and out is not None: errs.append(abs(out - rival[r + 1]))
    return sum(errs) / len(errs) if errs else None


async def play_game(llm, spec, rounds, progress=None):
    """spec: dict(model_key, arm, start, framing, seed). Returns one JSON-able record with all decisions and reasoning."""
    arm, seed, start, framing = spec["arm"], spec["seed"], spec["start"], spec["framing"]
    rng = random.Random(f"{arm}|{start}|{framing}|{seed}")
    ma, mb = arm_modes(arm, seed)
    A = Agent(ma, rng, make_predictor(llm, framing)); B = Agent(mb, random.Random(rng.random()), make_predictor(llm, framing))
    cur = {"calls": 0, "tin": 0, "tout": 0, "sim_fail": 0}; CUR.set(cur)
    ha = start_history(start); hb = list(ha)
    rec = dict(**spec, rounds=rounds, warm_n=WARM_N, modes=[ma, mb], pa=[], pb=[], dec=[], fails=0)
    for r in range(rounds):
        if progress and r and r % 10 == 0: progress(spec, r, cur)
        da, db = await asyncio.gather(decide(llm, A, ha, framing), decide(llm, B, hb, framing))
        pa, pb = da["p"], db["p"]
        if pa is None: pa, rec["fails"] = ha[-1][0], rec["fails"] + 1       # fall back to repeating own last price
        if pb is None: pb, rec["fails"] = hb[-1][0], rec["fails"] + 1
        ha.append((pa, pb)); hb.append((pb, pa))
        rec["pa"].append(pa); rec["pb"].append(pb); rec["dec"].append([da, db])
    turns = [t for ds in rec["dec"] for d in ds for t in d["turns"]]
    rec["length_cuts"] = sum(1 for t in turns if t["f"] == "length")
    rec["turns"] = len(turns)
    rec["fail_rate"] = rec["fails"] / (2 * rounds)
    rec["sim_fail"] = cur["sim_fail"]
    rec["sim_mae"] = sim_mae([d[0] for d in rec["dec"]], [d[1] for d in rec["dec"]], rec["pa"], rec["pb"])
    rec["meter"] = cur
    rec["cost_usd"] = round(llm.cost(cur["tin"], cur["tout"]), 5)
    return rec
