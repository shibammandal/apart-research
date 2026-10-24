# Part 2 draft: what makes an LLM duopoly go toward H1 (prices up) or H2 (price war)?

Status: DRAFT. Numbers marked (est.) are estimates until the 20-call test and the benchmark run (Section 9).

## 1. Question and contribution

Part 1 asked whether giving competing LLM sellers a "simulate your rival" tool raises prices (H1, stability / cartel)
or lowers them (H2, price war).
Part 2 asks what leads to each outcome.
The contribution is a taxonomy: the scenario factors and the reasoning mechanisms that lead to H1, and those that lead to H2.

## 2. How the experiment runs (simple version)

- Two LLM sellers play a repeated pricing game (integer prices 1-10, fixed unit production cost of 1 for both sellers, so profit = (price - 1) x buyers; winner-take-all, ties split).
- Every call goes to a Modal **shared endpoint** (a hosted model, billed per token). No GPU is rented, nothing is deployed.
- The notebook only sends web requests, so it runs on a small CPU machine.
- Every decision is stored with its full reasoning text.

## 3. Arms (4)

| Arm | Seller A | Seller B |
|---|---|---|
| `none` | no tool | no tool |
| `mutual` | real simulator | real simulator |
| `one_sided` | real simulator | no tool |
| `placebo` | fake simulator (a random past rival price) | fake simulator |

The real simulator is the rival's own model, called without tools on the rival's view of the history. Same text protocol as Part 1 (`QUERY: n`, `PRICE: n`, at most 3 queries).

## 4. Scenario factors (the "what could cause it" axes)

| Factor | Levels | Why |
|---|---|---|
| Arm | 4 (above) | the tool itself |
| Model | 3 families (Section 5) | does it generalise |
| Start state | warm (8 vs 8 for 5 rounds), cold (2 vs 2 for 5 rounds) | tool *sustains* cooperation vs *creates* it vs *breaks* it |
| Framing | neutral, "volume" (one extra sentence) | robustness, one model only |
| Rounds | 100, horizon hidden | enough time for wars and recoveries |

Observed (not set) per query, logged from the data: what the simulator said (rival retaliates / holds), and how accurate it was.
These are mediators and come from the logs, not from extra runs.

## 5. Models (shared endpoint)

| Role | Model | Price in/out per MTok | Plan |
|---|---|---|---|
| Main A (cheap) | `nvidia/GLM-5.3-Flash-NVFP4` | $0.15 / $0.50 | 5 seeds, both starts |
| Main B | `deepseek-ai/DeepSeek-V4.1-Flash` | $0.30 / $1.20 | 5 seeds, both starts |
| Anchor | `zai-org/GLM-5.3` | $1.40 / $4.40 | 2 seeds, warm start only |
| Judge (not Modal) | a Claude model on Bedrock | per token on AWS | codes the reasoning |

Rule: the judge must not be one of the players. Kimi K3 and Qwen3.8-Max are skipped for cost; add only if credits remain.

## 6. Logging (per agent, per round)

Full reasoning text (the thinking part, if the endpoint returns it, kept separate from the final answer), every query and
its output, both prices, round, arm, seed, model, start state, framing, simulator prediction vs the rival's actual next
price, token counts. One JSON line per game; games are resumable.

## 7. From logs to the taxonomy

1. Pilot batch with full logging. Check that the text is complete.
2. Split games: 20% development, 80% test, split by game and never by round.
3. A judge model reads decisions in the development split with their context and proposes free-form labels.
   Cluster them into about 10-15 mechanisms. Write `codebook.md`: definition and two examples each. Freeze it.
   Starting guesses, not conclusions: deterrence confirmed, exploitation, tit-for-tat / signalling, myopic undercut,
   stability preference, end-game, tool distrust, copying the rival, anchoring on the queried price, volume focus.
4. Code the test split with the frozen codebook. The judge never sees arm or outcome.
5. Humans double-code about 200 random decisions. Report kappa per code. A code is used in a claim only if kappa >= 0.7.
6. Models:
   - decision level: next move (up / hold / down) on mechanism codes, arm, model, round; standard errors clustered by game;
   - game level: time to first price war and to first sustained 8 (both agents >= 8 for 5 rounds), by mechanism mix;
   - factors to mechanisms: mixed-effects with model and seed as random effects; Holm correction.
7. Output: a table and one figure: factors, mechanisms, outcome (H1 / H2 / neither).

## 8. Known weaknesses a reviewer will raise, and the answer

| Concern | Answer |
|---|---|
| Reasoning text may not be the real cause | State it as a limitation: mechanisms are reported as associated with H1 / H2, not proven causes. The arm, model and start-state effects are causal because they are randomised |
| Codes chosen after seeing outcomes | Frozen codebook on a separate split, blind judge |
| One judge model may be biased | Human kappa, plus a second judge on a 10% subset |
| Few seeds | Use the decision-level analysis (thousands of decisions per model); report game-level CIs honestly |
| Models may be thinking-only or non-thinking, so the text isn't comparable | Compare final rationale across models; use thinking text only within a model |
| Many games are "neither" | Report the share; do not force a label |
| Shared endpoint models can change under us | Record the model name and run dates; run all arms of a model within the same day |

## 9. Cost (est.) and budget

Assumption: about 900 input and 350 output tokens per call, and calls per 100-round game: none 200, one_sided 600, mutual 1000, placebo 1000.
Cost of one seed across the four arms: about $2.0 on DeepSeek Flash, about $0.9 on GLM Flash NVFP4, about $7.4 on GLM 5.3.

| Run | Games | Cost (est.) |
|---|---|---|
| GLM Flash NVFP4, 5 seeds x 2 starts | 40 | ~$9 |
| DeepSeek Flash, 5 seeds x 2 starts | 40 | ~$20 |
| GLM 5.3, 2 seeds x 1 start | 8 | ~$15 |
| Volume framing, one model, 2 seeds | 8 | ~$2-4 |
| **Total** | | **~$46-48** |

Your Modal credit is $50.99. That is thin. If a model returns long thinking text, output tokens could double the cost.
So:
- hard cap in code from the first call (stop when the estimated spend reaches the cap),
- order runs cheapest first,
- measure real cost on 3 games before launching anything, then rescale seeds,
- judge calls on AWS are paid separately.

## 10. Before the real run

1. Create a proxy token in Modal and a shared endpoint (token goes in an environment variable, never in the notebook).
2. Make 20 calls: does the balance drop, does the response contain reasoning, what are the rate limits.
3. Run 3 full games per arm on the cheap model, measure cost and parse failures.
4. Freeze the config. Then run.

## 11. Timeline

Day 1: test calls, pilot logging, benchmark. Day 2: main runs. Day 3: development split, codebook. Day 4: coding, human kappa. Day 5: analysis, figures. Day 6: write-up.
