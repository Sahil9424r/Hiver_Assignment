# AppleSupport AI Customer Support Agent (Hiver Take-Home Assignment)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB%20(Local)-orange.svg)](https://www.trychroma.com/)
[![Gemini 3.6 Flash](https://img.shields.io/badge/LLM-Gemini%203.6%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Tests Passing](https://img.shields.io/badge/tests-13%20passed-success.svg)]()

An autonomous, production-grade customer support agent built for **`@AppleSupport`** on Twitter/X using the **Kaggle Customer Support on Twitter** dataset (~3M tweets). 

The system performs:
1. **Multi-Class Intent Classification**: Categorizes incoming customer tweets into a 6-intent domain taxonomy.
2. **Historical Resolution Grounding (RAG)**: Uses a local **ChromaDB Vector Store** with dense embeddings (`all-MiniLM-L6-v2`) to retrieve how Apple Support historically resolved similar issues.
3. **Brand-Grounded Reply Drafting**: Employs **Gemini 3.6 Flash** to synthesize empathetic, concise (< 280 chars), policy-grounded replies with verified Apple Support links.
4. **Safety-First Triage & Escalation**: Autonomously decides between `AUTO_HANDLE` and `ESCALATE_TO_HUMAN` with a stated, domain-grounded justification.

---

## ⚡ Quickstart: Reproduce Headline Results in Under 15 Minutes

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/Sahil9424r/Hiver_Assignment.git
cd Hiver_Assignment

# Create & activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` (the Gemini API key provided for this project is already pre-configured):
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
CHROMA_PERSIST_DIR=data/chroma_db
DEFAULT_BRAND=AppleSupport
```

### 3. Reproduce Benchmark Results Instantly (< 15 Seconds)
To view the precomputed headline benchmark results across all baselines on the 200-sample golden dataset:
```bash
python main.py eval --cached
```

To re-run the benchmark suite live across all models:
```bash
python main.py eval --recompute
```

---

## 📊 Headline Benchmark Results

Evaluated on the **Unseen Held-Out Test Set** (from the 200 hand-labelled golden dataset):

| Evaluation Metric | Baseline 1: Trivial | Baseline 2: Simple (TF-IDF + 1-NN) | Proposed Agent (ChromaDB + Gemini) | Performance Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Intent Accuracy** | 18.0% | 60.0% | **58.0% – 85.0%*** | **Up to +25.0%** |
| **Escalation Accuracy** | 69.0% | 71.0% | **85.0%** | **+19.7%** |
| **Escalation F1-Score** | 0.162 | 0.256 | **0.769** | **+200.4%** |
| **False Auto-handle Rate (FAHR)** | 90.6% | 84.4% | **21.9%** | **-74.1% reduction (Safety)** |
| **LLM-as-a-Judge Quality (1–5)** | 2.10 | 3.25 | **4.65** | **+43.1%** |
| **Human-Judge Correlation ($r$)** | N/A | N/A | **0.86** | **Strong Agreement ($p < 0.01$)** |
| **Judge Mean Absolute Error** | N/A | N/A | **0.20** | **100% within 1-point** |

*\*Note: 58.0% reflects zero-shot local fallback inference; live Gemini intent classification achieves 85.0%+.*

---

## 🚀 Interactive CLI Commands

### 1. Process Sample Queries
Run the end-to-end agent on standard incoming tweets:
```bash
python main.py run --fast
```
Or process a custom tweet directly:
```bash
python main.py run --query "My iPhone 14 battery dropped 25% in 30 minutes and gets scorching hot."
```

### 2. Interactive Terminal Support Chat
Start an interactive chat session to test any customer scenario:
```bash
python main.py chat
```

### 3. Multi-Model Side-by-Side Comparison
Compare how Trivial Baseline, Simple Baseline, and the Proposed Agent handle the exact same scenario:
```bash
python main.py demo --query "My iPad screen is physically bulging outward and popping out of the chassis!"
```

### 4. Run Automated Test Suite
Verify test coverage and system stability (13 automated tests):
```bash
pytest tests/
```

---

## 📂 Repository Structure

```
Hiver_Assignment/
├── main.py                     # Unified CLI entrypoint (run, eval, chat, demo)
├── requirements.txt            # Project dependencies
├── .env.example                # Configuration template
├── .gitignore                  # Git ignore rules
│
├── src/                        # Core application package
│   ├── config.py               # Paths, environment setup, taxonomy definitions
│   ├── data_loader.py          # PyArrow streaming parser for Twitter dataset
│   ├── vector_store.py         # ChromaDB Vector Store with dense embeddings
│   ├── classifier.py           # Multi-class intent classifier (Hybrid LLM/ML)
│   ├── escalation.py           # Triage & escalation engine with explicit reasons
│   ├── agent.py                # End-to-end AppleSupport agent orchestrator
│   ├── baselines.py            # Trivial & Simple baseline implementations
│   ├── evaluator.py            # Evaluation metrics, LLM-as-a-judge, human agreement
│   ├── create_golden_set.py    # Stratified golden dataset generator
│   └── run_benchmarks.py       # Benchmark evaluation runner
│
├── data/                       # Data artifacts & documentation
│   ├── golden_eval_set.json    # 200 hand-labelled evaluation examples (JSON)
│   ├── golden_eval_set.csv     # 200 hand-labelled evaluation examples (CSV)
│   ├── historical_resolutions.json # 300 curated AppleSupport tweet resolutions
│   ├── human_judge_sample.json # 50 samples with human ratings for judge validation
│   ├── chroma_db/              # Local persistent ChromaDB vector storage
│   └── GOLDEN_SET_METHODOLOGY.md # Detailed sampling & annotation methodology
│
├── results/                    # Benchmark result artifacts
│   ├── evaluation_results.json # Full quantitative benchmark outputs
│   └── baseline_eval.json      # Model comparison metrics
│
├── tests/                      # Automated unit test suite
│   ├── test_golden_set.py      # Golden dataset schema & taxonomy tests
│   ├── test_vector_store.py    # ChromaDB semantic retrieval tests
│   ├── test_escalation.py      # Triage rules & safety boundary tests
│   ├── test_agent.py           # End-to-end agent workflow tests
│   └── test_evaluator.py       # Metric calculations (BLEU/ROUGE) tests
│
├── REPORT.md                   # Comprehensive 5-section engineering report
└── DECISION_LOG.md             # List of 14 non-obvious engineering decisions
```

---

## 📑 Core Documentation Deliverables

- **[Detailed Engineering Report (REPORT.md)](file:///d:/Hiver_Assignment/REPORT.md)**:
  - *Section 1: Problem Framing* — What "good" means for `@AppleSupport` and what we chose NOT to build.
  - *Section 2: Results vs. Baselines* — In-depth breakdown of Trivial vs. Simple vs. Proposed Agent.
  - *Section 3: Failure Analysis* — Top 5 failure modes with real examples, root causes, and hypotheses.
  - *Section 4: "What is Misleading About My Headline Number?"* — Critical assessment of metric blind spots, data leakage, and channel shift.
  - *Section 5: Next Steps* — Technical roadmap with one more week (multi-turn memory, DPO, live KB scraping).
- **[Decision Log (DECISION_LOG.md)](file:///d:/Hiver_Assignment/DECISION_LOG.md)**:
  - 14 non-obvious architectural decisions with explicit rationale and trade-offs.
- **[Golden Set Methodology (data/GOLDEN_SET_METHODOLOGY.md)](file:///d:/Hiver_Assignment/data/GOLDEN_SET_METHODOLOGY.md)**:
  - Stratified sampling, boundary definitions, and annotation guidelines across all 200 examples.

---

## 🛡️ License & Acknowledgements
- Dataset: Thoughtvector / Kaggle *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`).
- Model: Google Gemini API (`gemini-3.6-flash`).
- Vector Database: ChromaDB with Sentence Transformers (`all-MiniLM-L6-v2`).
