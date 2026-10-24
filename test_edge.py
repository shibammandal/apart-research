"""Edge-case tests for the HTTP client against a scripted fake server (no network, no cost). Run: python test_edge.py"""
import asyncio, json
from aiohttp import web
from llm import LLM, Budget, Aborted, Fatal

OK = lambda content="PRICE: 7", **k: (200, {"choices": [{"message": {"content": content, **k.get("msg", {})}, "finish_reason": "stop"}],
                                           **({"usage": {"prompt_tokens": 100, "completion_tokens": 20}} if k.get("usage", True) else {})}, {})
SCRIPT, SEEN = [], []

async def handler(req):
    SEEN.append(await req.json())
    st, body, hdr = SCRIPT.pop(0) if SCRIPT else OK()
    if isinstance(body, str): return web.Response(status=st, text=body, headers=hdr)
    return web.json_response(body, status=st, headers=hdr)

async def main():
    app = web.Application(); app.router.add_post("/v1/chat/completions", handler)
    run = web.AppRunner(app); await run.setup(); site = web.TCPSite(run, "127.0.0.1", 8765); await site.start()
    url = "http://127.0.0.1:8765/v1"; M = [{"role": "user", "content": "hi"}]
    async def mk(**kw):
        l = LLM(url, "m", 1.0, 2.0, Budget(kw.pop("cap", 10)), {"X": "y"}, **kw); await l.open(); return l
    ok = fail = 0
    def check(name, cond):
        nonlocal ok, fail
        print(("PASS " if cond else "FAIL ") + name); ok += cond; fail += (not cond)

    SCRIPT[:] = [(429, "slow", {"Retry-After": "0"}), (503, "down", {}), OK()]; l = await mk()
    r = await l.chat(M); check("retries 429 then 503 then succeeds", r["content"] == "PRICE: 7" and l.budget.retries == 2)

    SCRIPT[:] = [(401, "no", {})]; l = await mk()
    try: await l.chat(M); check("401 is fatal", False)
    except Fatal as e: check("401 is fatal with clear message", "auth" in str(e))
    try: await l.chat(M); check("after fatal, later calls abort", False)
    except Aborted: check("after fatal, later calls abort", True)

    SCRIPT[:] = [(404, "nope", {})]; l = await mk()
    try: await l.chat(M); check("404 is fatal", False)
    except Fatal as e: check("404 is fatal", "model name" in str(e))

    SCRIPT[:] = [(400, "temperature is deprecated for this model", {}), OK()]; SEEN.clear(); l = await mk()
    r = await l.chat(M); check("temperature rejected: retried without it", r["content"] == "PRICE: 7" and "temperature" not in SEEN[-1] and not l.send_temperature)
    await l.chat(M); check("temperature stays off afterwards", "temperature" not in SEEN[-1])

    SCRIPT[:] = [OK(usage=False)]; l = await mk()
    r = await l.chat(M); check("missing usage is estimated and flagged", l.budget.estimated == 1 and r["tin"] > 0)

    SCRIPT[:] = [OK(None, msg={"content": None, "reasoning_content": "thinking only"})]; l = await mk()
    r = await l.chat(M); check("null content with reasoning_content", r["content"] == "" and r["reasoning"] == "thinking only")

    SCRIPT[:] = [OK("<think>let me see</think>\nPRICE: 5")]; l = await mk()
    r = await l.chat(M); check("<think> split out of content", r["content"] == "PRICE: 5" and r["reasoning"] == "let me see")

    SCRIPT[:] = [OK([{"type": "text", "text": "PRICE: 4"}])]; l = await mk()
    r = await l.chat(M); check("content as list of parts", r["content"] == "PRICE: 4")

    SCRIPT[:] = [(400, "maximum context length exceeded", {})]; l = await mk()
    r = await l.chat(M); check("context-length 400 returns error dict, no raise, no retry", r["error"].startswith("400") and len(SEEN) > 0 and l.budget.gave_up == 1)

    SCRIPT[:] = [(200, "not json{", {}), OK()]; l = await mk()
    r = await l.chat(M); check("malformed JSON is retried", r["content"] == "PRICE: 7")

    SCRIPT[:] = [(503, "x", {"Retry-After": "0"})] * 12; l = await mk(max_attempts=2, breaker=2)
    await l.chat(M)
    try: await l.chat(M); check("circuit breaker trips", False)
    except Fatal: check("circuit breaker trips after repeated failures", True)

    SCRIPT[:] = [OK(), OK()]; l = await mk(cap=0.0001)
    await l.chat(M)
    try: await l.chat(M); check("budget cap aborts", False)
    except Aborted: check("budget cap aborts before the next call", True)

    await run.cleanup(); print(f"\n{ok} passed, {fail} failed"); raise SystemExit(1 if fail else 0)

asyncio.run(main())
