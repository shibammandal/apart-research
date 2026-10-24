# Final plan: what leads LLM sellers to collude (H1) or undercut (H2)

This replaces `PLAN.md` and `DRAFT.md`. Numbers marked (est.) are estimates until measured on the endpoint.

## 1. Question and contribution
Part 1 asked whether a "simulate your rival" tool raises prices (H1, stability) or lowers them (H2, price war).
Part 2 asks what leads to each. Contribution: a taxonomy of the scenario factors and the reasoning mechanisms associated with H1 vs H2.
Claims: setup effects (arm, model, start state) are causal because we assign them. Reasoning mechanisms are reported as associated, and the limitation is stated.

## 2. The game
Two LLM sellers, 100 rounds, hidden horizon. Integer prices 1-10, fixed unit cost 1 for both, 100 buyers, cheaper seller takes all, ties split.
Profit = (price - 1) x buyers. Nash price about 2, cooperative reference 8, monopoly 10. Last 20 rounds of history are shown.
Pre-played 5 rounds: warm = both at 8, cold = both at 2. Pre-played rounds are not analysed.

## 3. Arms
| Arm | Seller A | Seller B |
|---|---|---|
| none | no tool | no tool |
| mutual | real simulator | real simulator |
| one_sided | real simulator | none (tool seat alternates by seed) |
| placebo | fake simulator | fake simulator |

Real simulator = the rival's own model, no tools, on the rival's view of the history with the queried price added. Placebo = a random draw from the rival's last 10 prices.
Protocol: `QUERY: n` (max 3 per decision), end with `PRICE: n`.

## 4. Models (Modal shared endpoints, billed per token)
| Key | Model | $/MTok in/out | Plan |
|---|---|---|---|
| glm-flash | nvidia/GLM-5.3-Flash-NVFP4 | 0.15 / 0.50 | 5 seeds, warm + cold (first, smallest) |
| deepseek-flash | deepseek-ai/DeepSeek-V4.1-Flash | 0.30 / 1.20 | 5 seeds, warm + cold |
| glm53 | zai-org/GLM-5.3 | 1.40 / 4.40 | 2 seeds, warm only |
Judge for coding the reasoning: a Claude model on AWS Bedrock, never a player.

## 5. What is logged
Per game, one JSON line: prices, and per round per seller the full reasoning and answer text of every turn (reasoning kept separate when the
endpoint returns it), every query and the simulator's answer, finish reasons, token counts, parse failures, simulator error vs the rival's actual next price, cost.
Games are resumable: finished games are kept, rerun the same command.

## 6. Code (this folder)
- `llm.py`: client (retries, Retry-After, fatal auth/URL errors, temperature auto-drop, missing usage, `<think>` split, budget cap, circuit breaker), plus a mock.
- `engine.py`: prompts, protocol, simulator, arms, one logged game.
- `run.py`: `ping`, `play`, `summary`.
- `test_edge.py`: 14 offline edge-case tests of the client (all pass).
- `.env.example`: where the endpoint URL and token go.

## 7. Run order for the smallest model (glm-flash)
1. Copy `.env.example` to `.env`, fill the URL and token.
2. `python run.py ping --model-key glm-flash` - checks URL, auth, whether reasoning comes back, usage, cost per call.
3. `python run.py play --model-key glm-flash --rounds 12 --seeds 1 --cap 0.30` - smoke test (all 4 arms).
4. `python run.py play --model-key glm-flash --rounds 100 --seeds 1 --cap 2.0` - one full-length game per arm; read `python run.py summary` and a few reasoning logs.
5. Check: parse failures under about 3%, real cost per game vs estimate, whether prices vary (not stuck at 2 or 8 in every game).
6. Only then the full grid: `--seeds 5 --starts warm,cold --cap <budget>`.

## 8. From logs to taxonomy
1. Split games 20% development / 80% test (by game).
2. A judge model reads development decisions with context and proposes labels; cluster into about 10-15 mechanisms; write `codebook.md` and freeze it.
3. Code the test split blind to arm and outcome.
4. Humans double-code about 200 decisions; report kappa per code; use codes with kappa >= 0.7.
5. Link: next move (up / hold / down) on codes + arm + model + round, game-clustered errors; time to first price war and first sustained 8; factors to mechanisms with Holm correction.
6. Output: one table and one figure (factors -> mechanisms -> H1 / H2 / neither).

## 9. Cost (est.) and budget
About 900 tokens in and 350 out per call; calls per 100-round game: none 200, one_sided 600, mutual 1000, placebo 1000.
Full grid about $9 (glm-flash), $20 (deepseek-flash), $15 (glm53 anchor). Total about $44 against $50.99 of credit. Thin.
Controls: hard cap per run, cheapest model first, benchmark before scaling, seeds rescaled from measured cost.
Open: whether the $50.99 credit applies to shared endpoints (the page says plan credits do not). The ping/smoke runs will show whether the balance drops.

## 10. Known risks
- All games may collapse to 2 or stay at 8: then there is little to explain. Step 7.5 checks this before spending.
- Few seeds: lean on decision-level analysis; report game-level intervals honestly.
- Reasoning text may not be the true cause (cf. arXiv 2609.18346): stated limitation.
- Novelty: no abstract found describing a rival-simulation tool (search 2026-10-24). Confirm in full texts before writing "to our knowledge".
- Unverified until the first ping: auth header names (`Modal-Key` / `Modal-Secret` assumed), whether `temperature` is accepted, the reasoning field name.
