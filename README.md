# Do Rival-Simulation Tools Stabilize Collusion or Spark Price Wars?
### A Causal and Mechanistic Taxonomy of LLM Duopoly Dynamics

This repository contains the complete experimental code, simulation engine, LLM-judge pipeline, and raw dataset for our investigation of repeated Bertrand duopolies between autonomous large language models.

---

## Headline Findings

1. **Rival-Simulation Tools Destabilize Cooperation (H2 confirmed, H1 refuted):** In warm-start markets ($p=8$), providing sellers with rival-simulation tools cuts sustained collusion from 100% to 55% in GLM-Flash and from 60% to 20% in DeepSeek-Flash, while increasing price wars up to 8.3-fold.
2. **Cold-Start Trap (0% Collusion):** When initialized at competitive Nash equilibrium ($p=2$), simulation tools are completely incapable of coordinating price increases (0% sustained collusion across all models).
3. **Cognitive Etymology:** Blind LLM-judge coding of 2,334 decision traces shows that counterfactual simulation erodes precautionary deterrence (`k2`, dropping from 25.1% to 10.8%) and multiplies myopic profit calculation (`k10`, jumping from 1.1% to 11.8%) and undercutting (`k5`, 8.3% to 13.4%), triggering tit-for-tat price wars (`k4`, 81% downward moves).
4. **Human Validation:** Inter-rater reliability between a human annotator and Claude 3.5 Sonnet achieves **75.86% agreement** and **Cohen's $\kappa = 0.714$** (Substantial Agreement).

---

## Repository Structure

```
├── run.py                 # CLI interface for simulation: ping, play, summary
├── engine.py              # Bertrand duopoly game rules, prompt templates, simulator tool, logging
├── llm.py                 # Async LLM API client (retries, rate limiting, budget caps, CoT extraction)
├── judge.py               # Qualitative coding pipeline (Claude 3.5 Sonnet on AWS Bedrock)
│
├── data_main/             # Full experimental datasets (280 games, 56,000 decisions)
│   ├── games_glm-flash.jsonl
│   ├── games_deepseek-flash.jsonl
│   └── games_glm53.jsonl
│
├── judge_data/            # Coded qualitative mechanism data & validation
│   ├── codebook.json      # Frozen 10-mechanism taxonomy definitions
│   ├── coded_test.jsonl   # 2,334 blind-coded test decisions from Claude judge
│   ├── human_sheet.csv    # Human validation labels (used to calculate Cohen's Kappa)
│   ├── human_codebook.txt # Human-readable codebook guide
│   ├── open_labels.jsonl  # Initial open coding traces from development split
│   ├── sample_dev.jsonl   # Stratified development sample (240 decisions)
│   └── sample_test.jsonl  # Stratified test sample (3,786 decisions)
│
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template for API endpoints and credentials
├── .gitignore             # Configured to protect credentials, logs, and temp artifacts
└── README.md              # Project documentation and reproduction guide
```

---

## Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/shibammandal/apart-research.git
cd apart-research
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. View Summary Results
You can inspect the complete empirical summary table across all 24 experimental cells (280 games) without running any API calls:
```bash
python run.py summary --out data_main
```

### 3. Verify Human Agreement (Cohen's Kappa)
To reproduce the inter-annotator reliability score between the human annotator and Claude 3.5 Sonnet:
```bash
python judge.py agree --a judge_data/coded_test.jsonl --b judge_data/human_sheet.csv
```
Output:
```
n=29  agreement 75.86%  Cohen's kappa 0.714
  k1           kappa(one-vs-rest) 0.79
  k2           kappa(one-vs-rest) 0.79
  k4           kappa(one-vs-rest) 0.78
  k5           kappa(one-vs-rest) 0.79
  k8           kappa(one-vs-rest) 1.00
```

---

## Running New Simulations

### 1. Credentials
Copy `.env.example` to `.env` and fill in your endpoint details:
```bash
cp .env.example .env
```
For Modal shared endpoints:
```ini
ENDPOINT_URL_GLM_FLASH=https://your-modal-endpoint.modal.direct
MODAL_PROXY_KEY=your-key
MODAL_PROXY_SECRET=your-secret
```
Or standard OpenAI-compatible bearer endpoints:
```ini
ENDPOINT_URL_MY_MODEL=https://your-provider/v1
API_KEY=your-api-key
```

### 2. Verify Endpoint Connection
```bash
python run.py ping --model-key glm-flash
```

### 3. Execute Duopoly Games
Run 100-round games with budget caps and concurrency control:
```bash
# Warm start (cooperative history p=8):
python run.py play --model-key glm-flash --rounds 100 --seeds 5 --starts warm --cap 5.0 --out data_main

# Cold start (Nash trap history p=2):
python run.py play --model-key glm-flash --rounds 100 --seeds 5 --starts cold --cap 5.0 --out data_main
```

---

## Experimental Protocol & Arms

* **Market Rules:** 2 symmetric firms, 100 buyers, prices $1–$10, unit cost = $1, profit = $(p - 1) \times q$. Cheaper seller takes all; ties split 50/50. Hidden 100-round horizon.
* **Treatment Arms:**
  * `none`: Baseline (no simulation tools).
  * `mutual`: Both sellers have an interactive rival simulator (up to 3 queries/round).
  * `one_sided`: Asymmetric advantage (only one seller possesses the tool).
  * `placebo`: Pseudo-simulator returning random historical prices (controls for query psychology).
