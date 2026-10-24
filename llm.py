"""LLM client for Modal shared endpoints (OpenAI-compatible chat/completions, billed per token).

Handles: retries with Retry-After, fatal auth errors, temperature rejected by the model, missing usage,
reasoning returned in a separate field or inside <think> tags, hard budget cap, circuit breaker.
"""
import asyncio, contextvars, json, os, random, re, time
import httpx

CUR = contextvars.ContextVar("cur", default=None)   # per-game meter dict, updated on every call
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class Aborted(Exception):
    """Budget cap hit or run stopped; the current game is discarded."""


class Fatal(Exception):
    """Unrecoverable (bad auth, wrong URL or model, endpoint down). Message is shown to the user."""


def load_env(path=".env"):
    """Tiny .env reader (no dependency). Existing environment variables win."""
    if not os.path.exists(path): return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def split_think(text):
    """-> (reasoning_inside_content, visible_answer). Handles a closed </think>; an unclosed <think> means truncation."""
    if "</think>" in text:
        a, b = text.rsplit("</think>", 1)
        return a.replace("<think>", "").strip(), b.strip()
    return "", text


class Budget:
    def __init__(self, cap):
        self.cap, self.spent = float(cap), 0.0
        self.calls = self.tin = self.tout = self.errors = self.retries = self.gave_up = self.estimated = 0

    def check(self):
        if self.spent >= self.cap:
            raise Aborted(f"budget cap ${self.cap:.2f} reached (spent ${self.spent:.4f})")


class LLM:
    def __init__(self, url, model, price_in, price_out, budget, headers=None, max_tokens=1500, temperature=0.7,
                 concurrency=16, timeout=300, max_attempts=6, breaker=20, extra_body=None):
        u = url.rstrip("/")
        if not u.endswith("/chat/completions"):
            u += "/chat/completions" if u.split("://", 1)[-1].count("/") else "/v1/chat/completions"   # bare host -> OpenAI-style /v1 path
        self.url = u
        self.model, self.pin, self.pout, self.budget = model, price_in, price_out, budget
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.max_tokens, self.temperature = max_tokens, temperature
        self.timeout, self.max_attempts, self.breaker = timeout, max_attempts, breaker
        self.extra_body = extra_body or {}
        self.send_temperature = True        # switched off if the model rejects the parameter
        self.sem = asyncio.Semaphore(concurrency)
        self.fail_streak, self.fatal, self.client = 0, None, None

    def cost(self, tin, tout): return tin * self.pin / 1e6 + tout * self.pout / 1e6

    async def open(self): self.client = httpx.AsyncClient(timeout=self.timeout)
    async def close(self):
        if self.client: await self.client.aclose()

    async def chat(self, messages, max_tokens=None, temperature=None):
        """-> dict(content, reasoning, finish, tin, tout, error). Never raises on a single bad reply; raises Aborted/Fatal."""
        if self.fatal: raise Aborted(self.fatal)
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens or self.max_tokens, **self.extra_body}
        t = self.temperature if temperature is None else temperature
        err = ""
        async with self.sem:
            for attempt in range(self.max_attempts):
                self.budget.check()
                if self.send_temperature and t is not None: body["temperature"] = t
                else: body.pop("temperature", None)
                try:
                    r = await self.client.post(self.url, headers=self.headers, json=body)
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    err = f"{type(e).__name__}"; self.budget.errors += 1
                    await self._sleep(attempt, None); continue
                if r.status_code in (401, 403):
                    self._die(f"auth failed ({r.status_code}). Check the proxy token / API key and header names. Body: {r.text[:200]}")
                if r.status_code == 404:
                    self._die(f"404: wrong endpoint URL or model name. URL={self.url} model={self.model}. Body: {r.text[:200]}")
                if r.status_code == 400:
                    low = r.text.lower()
                    if "temperature" in low and self.send_temperature:
                        self.send_temperature = False; continue          # drop the parameter for good, retry now
                    err = f"400: {r.text[:200]}"; self.budget.errors += 1
                    break                                                 # context too long / bad request: not retryable
                if r.status_code in RETRY_STATUS:
                    err = f"{r.status_code}"; self.budget.errors += 1; self.budget.retries += 1
                    await self._sleep(attempt, r.headers.get("retry-after")); continue
                if r.status_code != 200:
                    err = f"{r.status_code}: {r.text[:200]}"; self.budget.errors += 1; break
                try:
                    out = self._parse(r.json(), messages)
                except Exception as e:                                    # malformed JSON / missing fields
                    err = f"bad response: {type(e).__name__}"; self.budget.errors += 1
                    await self._sleep(attempt, None); continue
                self.fail_streak = 0
                return out
        self.budget.gave_up += 1; self.fail_streak += 1
        if self.fail_streak >= self.breaker:
            self._die(f"{self.fail_streak} consecutive failed calls; endpoint looks down. Last error: {err}")
        return {"content": "", "reasoning": "", "finish": "error", "tin": 0, "tout": 0, "error": err}

    def _die(self, msg):
        self.fatal = msg
        raise Fatal(msg)

    async def _sleep(self, attempt, retry_after):
        try: d = float(retry_after)
        except (TypeError, ValueError): d = min(60, 2 ** attempt)
        await asyncio.sleep(min(120, d) + random.random())

    def _parse(self, j, messages):
        ch = j["choices"][0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if isinstance(content, list):                                    # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        think, content = split_think(content)
        reasoning = (reasoning + "\n" + think).strip() if think else reasoning
        u = j.get("usage") or {}
        tin, tout = u.get("prompt_tokens"), u.get("completion_tokens")
        if tin is None or tout is None:                                  # usage missing: estimate, and say so
            tin = max(1, sum(len(str(m.get("content", ""))) for m in messages) // 3)
            tout = max(1, (len(content) + len(reasoning)) // 3)
            self.budget.estimated += 1
        b = self.budget
        b.calls += 1; b.tin += tin; b.tout += tout; b.spent += self.cost(tin, tout)
        cur = CUR.get()
        if cur is not None:
            cur["calls"] += 1; cur["tin"] += tin; cur["tout"] += tout
        return {"content": content, "reasoning": reasoning, "finish": ch.get("finish_reason") or "", "tin": tin, "tout": tout, "error": ""}


class MockLLM:
    """Offline stand-in with the same interface. Injects messy replies to exercise the parsers."""
    def __init__(self, budget, seed=0, price_in=0.15, price_out=0.5):
        self.budget, self.rng, self.pin, self.pout = budget, random.Random(seed), price_in, price_out
        self.model, self.fatal = "mock", None

    def cost(self, tin, tout): return tin * self.pin / 1e6 + tout * self.pout / 1e6
    async def open(self): pass
    async def close(self): pass

    async def chat(self, messages, max_tokens=None, temperature=None):
        self.budget.check(); await asyncio.sleep(0)
        sys_, last = messages[0]["content"], messages[-1]["content"]
        has_tool = "simulator tool" in sys_
        res = [int(m[1]) for m in re.findall(r"charge (\d+) this round, the rival is predicted to charge (\d+)", last)]
        used = sum(1 for m in messages[1:] if m["role"] == "assistant" and "QUERY" in m["content"])
        u = self.rng.random()
        if "No more queries" in last and u < 0.3: txt = "I still want to query QUERY: 6"            # asks again after the limit
        elif has_tool and used < 3 and u < 0.45: txt = f"Check retaliation.\nQUERY: {self.rng.randint(5, 9)}"
        elif u > 0.97: txt = "I am unsure what to do."                                              # no PRICE line
        elif u > 0.93: txt = ""                                                                      # empty reply
        else:
            p = 8 if (res and min(res) < 4) else self.rng.choice([6, 7, 7, 8, 8])
            txt = self.rng.choice([f"Hold.\nPRICE: {p}", f"Hold.\nPRICE: **{p}**", f"PRICE: {p}\n", f"<think>hmm</think>PRICE: {p}"])
        _, vis = split_think(txt)
        tin, tout = sum(len(m["content"]) for m in messages) // 4, len(txt) // 4 + 5
        b = self.budget; b.calls += 1; b.tin += tin; b.tout += tout; b.spent += self.cost(tin, tout)
        cur = CUR.get()
        if cur is not None: cur["calls"] += 1; cur["tin"] += tin; cur["tout"] += tout
        return {"content": vis, "reasoning": "mock reasoning" if self.rng.random() < .3 else "", "finish": "stop", "tin": tin, "tout": tout, "error": ""}
