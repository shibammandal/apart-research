# Part 2: a taxonomy of what pushes an LLM duopoly toward H1 (prices up) or H2 (price war)

H1 = a rival-simulation tool stabilises the market (prices rise / stay high).
H2 = a rival-simulation tool destabilises it (prices fall, price wars).
Part 1 asks *whether* the tool does either. Part 2 asks *what causes each*. That is the contribution.

## What the taxonomy is

A table with three linked layers. Each cell is measured, not asserted.

| Layer | Question | How it is obtained |
|---|---|---|
| A. Scenario factors | Which setup conditions predict H1 or H2? | Manipulated: arm, model, tool content, framing |
| B. Mechanisms | Which reasoning patterns sit between the setup and the price move? | Coded from the full reasoning logs |
| C. Outcomes | H1 episode (sustained >= 8) / H2 episode (drop >= 3, war) / neither | Computed from prices |

Result statement: "H1 occurs when {A-factors} through {B-mechanisms}; H2 occurs when ...".

The key rule: **layer A is causal (we randomise it). Layer B is correlational unless we validate it.**
Reasoning text is not guaranteed to be the real cause (CoT unfaithfulness). A reviewer will say this first.
So every mechanism in the final taxonomy must pass one of two checks: (i) an intervention flips the outcome
(inject the mechanism via the tool output, as in the gate), or (ii) it predicts the price move at the
decision level after controlling for arm and model.

## Design

Arms (4): `none`, `mutual`, `one_sided`, `placebo`. (Noisy-simulator arm dropped, as you said.)
Models: the 4 open-weight ones from Part 1, plus whatever Bedrock/OpenRouter models fit (a Haiku anchor, a Llama 70B).
Rounds: 100 (Part 1 used fewer). Seeds: as many as the budget allows, see cost below.

Scenario factors to vary (these are the "what could cause it" axes). Keep each to 2-3 levels:
1. Arm (above).
2. Model: family, size, reasoning vs non-reasoning.
3. What the simulator says. Observed rather than set: log whether each query returned "rival retaliates"
   (output < queried price when queried price < own) or "rival holds/ignores". This is the likely
   hinge between H1 and H2, and it is free to compute from logs.
4. Prompt framing: neutral vs "volume" (already built).
5. Start state: warm start at 8/8 vs cold start at Nash (2/2). Tests whether the tool *sustains* cooperation (H1)
   or *breaks* it (H2) versus *creates* it from nothing. Part 1 only had the warm start.
6. Horizon known vs hidden (end-game effects are a classic alternative explanation).

Do not run the full cross product. Use the 4 arms x 2 start states x 2 framings on every model
(16 cells), horizon only on one model.

## Logging (the part that must change from Part 1)

Part 1 notebooks truncate reasoning to 300 characters per decision (`ta[:300]` in `play_game`) and the
`<think>` block of R1 is mixed in. That is not enough for coding. Part 2 must store, per agent per round:
- full reasoning text, with `<think>` and the final answer stored separately
- every query and its simulator output
- both prices, the round, the arm, the seed, the model, the variant
- simulator prediction vs the rival's actual next price (accuracy per round)
- token counts

Storage is small (100 rounds x 2 agents x ~500 tokens x a few thousand games is under 1 GB of text).

## Building the taxonomy (steps)

1. **Pilot logs, then freeze the design.** Run a small batch with full logging. Check that the text is complete.
2. **Open coding on a development split (20% of games, never used again for testing).**
   An LLM judge (a different model from the players, e.g. Claude) reads decision-level reasoning with the context
   (history, queries, outputs, the price chosen) and proposes free-form labels. Cluster the labels into a codebook of
   roughly 10-15 mechanisms. Two human passes (you plus one other person) to merge and define them.
   Seed codes to look for, not to assume: *deterrence confirmed* ("simulator says they retaliate, so I hold"),
   *exploitation* ("simulator says they won't retaliate, so undercut"), *tit-for-tat / signalling*,
   *myopic undercut*, *stability preference*, *end-game / horizon*, *tool distrust*, *copying the rival's price*,
   *anchoring on the queried price* (the Nova failure), *volume focus*.
3. **Freeze the codebook** (write it to `codebook.md` with definitions and 2 examples each). No edits after this.
4. **Code the held-out 80% automatically** with the LLM judge, blind to arm and outcome (strip those from the prompt).
5. **Validate the coding.** Humans double-code ~200 random decisions. Report Cohen's kappa per code
   (target >= 0.7 for any code you use in a claim) and judge-vs-human agreement. Drop or merge codes below the bar.
6. **Link mechanisms to outcomes** at two levels:
   - decision level: multinomial logit, next-price move in {up, hold, down} ~ mechanism codes + arm + model + round,
     with game-clustered standard errors;
   - game level: which mechanism mix precedes an H1 episode vs an H2 episode (survival/hazard on time to first drop and to first sustained 8).
7. **Link scenario factors to mechanisms and outcomes** (this is layer A to B to C): mixed-effects model with model and seed as random effects. Report effect sizes with CIs and Holm correction.
8. **Causal validation of the top mechanisms.** For each of the 3-4 strongest mechanisms, run an intervention on the gate: inject the simulator output or a prompt line that induces it, check that the outcome moves the predicted way. A mechanism that only correlates is reported as "associated", not "causes".
9. **Write the taxonomy** as a decision tree / table plus one figure: factors on the left, mechanisms in the middle, H1/H2 on the right (Sankey).

## What would make the contribution weak (fix before running)

- Codes defined after seeing outcomes, then tested on the same games. Fix: dev/test split, frozen codebook.
- Judge sees the arm or outcome and codes accordingly. Fix: blind it.
- Taxonomy built from one model family. Fix: report per-model and show which parts generalise; label the rest model-specific.
- Reasoning models only expose `<think>`; non-reasoning ones give brief justification, so code rates are not comparable. Fix: code only the final visible rationale for cross-model comparisons; use `<think>` as a separate within-model analysis.
- A simulator that is wrong (low MAE accuracy) can be the real cause. Fix: include simulator accuracy as a factor.
- Many "neither" games. Report them; do not force every game into H1 or H2.

## Budget (Modal credits)

The four Part 1 notebooks already have caps summing to $98. Part 2 cannot also spend $100 unless Part 1 is cut down.
Two options:
- **Reuse:** run Part 1 as is (cheap seeds), and put the new logging into Part 2 runs only. Part 2 then needs a fresh budget.
- **Merge (recommended):** treat Part 2 as the main run. Change the Part 1 notebooks to save full reasoning and run only those, with more rounds and the added factors. Part 1 stays as the pilot (the gate and Haiku results).

Longer rounds and full logging raise output tokens, so cost per game goes up roughly with rounds. At 100 rounds,
expect to cut seeds per cell to about 5 on the 8B/14B models, 3 on 32B, 2 on R1. Judge cost (coding ~40k decisions with Claude Haiku via Bedrock) is about $30-60 at pilot prices and is paid on AWS, not Modal.
Verify the judge cost on 200 decisions before launching it on all.

## Folder contents (planned)

- `PLAN.md` this file
- `codebook.md` (written after step 3)
- `build_notebooks_p2.py` (copy of the Part 1 builder with full logging and the new factors)
- `code_reasoning.py` (LLM judge with blinding, runs on AWS)
- `analyze_taxonomy.py` (regressions, kappa, figures)
