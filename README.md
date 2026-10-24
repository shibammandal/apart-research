# Do rival-simulation tools push LLM sellers toward collusion or price wars?

Two LLM sellers play a repeated pricing game (prices 1-10, unit cost 1, 100 buyers, cheaper seller wins all, ties split).
Arms: `none`, `mutual` (both have a rival-simulator tool), `one_sided`, `placebo` (fake tool). Start: `warm` (8 vs 8) or `cold` (2 vs 2).
Every decision is logged with its full reasoning. See `final_plan.md` for the design.

## Run another model with your own API credits

### 1. Install
```
git clone <this repo> && cd <repo folder>
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python test_edge.py        # 14 offline tests, should print "14 passed, 0 failed"
python run.py play --mock --rounds 20 --seeds 1 --out /tmp/mock     # free end-to-end dry run
```

### 2. Credentials (any OpenAI-compatible chat/completions endpoint)
Copy `.env.example` to `.env` (never commit it) and set:
```
ENDPOINT_URL_<KEY>=https://your-provider/v1        # <KEY> = your model key in capitals, "-" becomes "_"
API_KEY=sk-...                                      # sent as "Authorization: Bearer ..."
```
Modal shared endpoints use `MODAL_PROXY_KEY` and `MODAL_PROXY_SECRET` instead of `API_KEY`.
A bare host (no path) gets `/v1/chat/completions` appended.

### 3. Register the model
Edit the `MODELS` dict at the top of `run.py`; prices are $ per million tokens:
```python
"my-model": dict(id="provider/model-name", pin=0.30, pout=1.20),
```
The endpoint URL is then read from `ENDPOINT_URL_MY_MODEL`.

### 4. Check the connection (fractions of a cent)
```
python run.py ping --model-key my-model
```
It must print `HTTP 200`. 401/403 = wrong token, 404 = wrong URL or model id.

### 5. Decide the reasoning setting BEFORE the big run (important)
Thinking models can spend thousands of tokens per call, which cost us hours and about 10x the money.
The `--effort` flag sends `reasoning_effort` to the endpoint:
- `--effort low` (default): worked for GLM-5.3-Flash (cut output from ~1500 to ~100 tokens per call).
- `--effort default`: sends nothing. Needed for DeepSeek-V4.1-Flash, where `low` turned thinking ON and every call hit the 4000-token limit.
Test both on the 12-round smoke test below and look at the cost and the `cuts` column. If a model ignores or misuses the setting, use `default`.
If your provider has a different way to control thinking, add it in `build_llm` in `run.py` (`extra_body`).

### 6. Smoke test (about $0.1-0.5)
```
python run.py play --model-key my-model --rounds 12 --seeds 1 --cap 0.5 --out data_smoke
python run.py summary --out data_smoke
```
Good signs: `fail_rate` ~0, `cuts` ~0, `sim_fail` ~0, time per game in minutes, cost per game matches your expectation.
If `fail_rate` is above 0.05 the game is flagged in `bad_games`; fix the token limit (`--max-tokens`) or the effort setting first.

### 7. Main run
Cost scales as (games x rounds). Our numbers: GLM-5.3-Flash about $0.05 per game, DeepSeek-V4.1-Flash $0.05 to $0.43 per game (the `mutual` arm is the expensive one).
```
python -u run.py play --model-key my-model --rounds 100 --seeds 10 --starts warm --cap 10 --games-parallel 20 --concurrency 32 --out data_main
python -u run.py play --model-key my-model --rounds 100 --seeds 5  --starts cold --cap 5  --games-parallel 20 --concurrency 32 --out data_main
```
- `--cap` is a hard stop in dollars (it can overshoot by one in-flight batch of calls). Set it below what you can afford.
- If you get HTTP 429 (rate limit), lower `--concurrency`; the client retries automatically.
- Safe to stop and re-run the same command: finished games are kept (`games_<model-key>.jsonl`).
- Use the same settings as the others (100 rounds, neutral framing, seeds counted from 0) so results can be pooled.

### 8. Look at the results and send them back
```
python run.py summary --out data_main
```
Send `data_main/games_<model-key>.jsonl` (full reasoning logs, tens of MB) and the exact command you ran, the model id, the date, and the `--effort` setting.
Do not send `.env`.

## Files
- `llm.py` client (retries, auth errors, budget cap, reasoning capture, mock)
- `engine.py` game, prompts, simulator, arms, logging
- `run.py` commands: `ping`, `play`, `summary`
- `test_edge.py` offline tests
- `final_plan.md` design and analysis plan
