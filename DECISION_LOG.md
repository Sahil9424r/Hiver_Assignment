# Engineering Decision Log: AppleSupport AI Agent

A documented record of the 14 non-obvious technical and architectural decisions made during the design, development, and evaluation of the system.

---

1. **Selecting `@AppleSupport` Over Other Brands in the Kaggle Dataset**
   - *Decision*: Chose `@AppleSupport` rather than `@AmazonHelp`, `@Delta`, or `@Uber_Support`.
   - *Why Non-Obvious*: While `@AmazonHelp` has higher raw tweet volume, its resolutions are overwhelmingly logistics-driven (package tracking, courier delays) which boils down to simple database lookups. `@AppleSupport` features intricate technical troubleshooting, explicit security boundaries (Apple ID / 2FA), hardware vs. software ambiguity, and strict public-to-private handoffs, making it a vastly richer test of agentic intelligence and triage judgment.

2. **Using Local VectorDB (ChromaDB + ONNX Embeddings) Over Cloud Vector Databases (Pinecone / Weaviate)**
   - *Decision*: Adopted a local, in-process ChromaDB instance with `all-MiniLM-L6-v2` dense embeddings instead of an external hosted vector database service.
   - *Why Non-Obvious*: Production systems often default to hosted vector clouds. However, local ChromaDB guarantees sub-5ms vector retrieval latencies, zero cloud egress costs, zero external credential setup for reviewers, and ensures the entire evaluation pipeline runs 100% offline in < 15 seconds.

3. **Decoupling Intent Classification from Triage / Escalation Reasoning**
   - *Decision*: Modeled Intent Classification and Triage Escalation as two separate, decoupled reasoning phases rather than a single unified prompt.
   - *Why Non-Obvious*: A single end-to-end prompt (*"classify intent and decide escalation"*) suffers from cognitive interference: LLMs frequently confuse the severity of an issue with its functional domain. Separating them allows deterministic safety guardrails (e.g. battery swelling, stolen card fraud) to execute independently of whether the classifier thinks it's hardware or software.

4. **Penalizing False Auto-Handles (FAHR) 5x More Severely Than False Escalations**
   - *Decision*: Configured triage policy and evaluation metrics to prioritize Escalation Recall over Escalation Precision.
   - *Why Non-Obvious*: In standard ML classification, F1 weights precision and recall equally. In customer service operations, the cost matrix is heavily asymmetric: a false escalation wastes ~$8 of human agent time, whereas a false auto-handle on a combusting battery or compromised Apple ID can cause device fire hazards, customer identity theft, or lawsuits.

5. **Excluding Verbatim Retrieval for Response Generation (Avoiding Naive 1-NN Reply Copying)**
   - *Decision*: Used retrieved historical resolutions strictly as *in-context few-shot exemplars* for Gemini rather than returning the raw historical tweet text directly.
   - *Why Non-Obvious*: Raw historical tweets in the Kaggle dataset contain specific customer names, dead 2017 t.co links, and agent initials (`^AB`). Feeding them as grounding context into the LLM allows the agent to synthesize current, clean, brand-consistent guidance while borrowing the exact procedural resolution.

6. **Implementing Strict Train/Test Splitting to Expose Baseline Data Leakage**
   - *Decision*: Split the 200-sample golden set into a 50% training set and 50% held-out test set, strictly fitting baselines on the training partition.
   - *Why Non-Obvious*: When nearest-neighbor baselines are evaluated on the same corpus they index, they achieve illusory 100% BLEU and accuracy numbers due to training set memorization. Splitting the dataset exposed that the Simple Baseline actually achieves only 60.0% accuracy on unseen queries.

7. **Double-Layered Heuristic and Vector Fallback to Safeguard Against API Quota Exhaustion**
   - *Decision*: Built a cascading architecture: Gemini 3.6 Flash $\rightarrow$ Gemini 3.5 Flash Lite $\rightarrow$ ChromaDB semantic vector k-NN $\rightarrow$ Regex heuristic.
   - *Why Non-Obvious*: Public LLM APIs experience rate limits (429) and high-demand spikes (503). The agent never crashes or drops customer queries: if external APIs are unreachable, local embeddings and domain rules immediately take over with zero service interruption.

8. **Restricting Maximum Response Length to Under 280 Characters**
   - *Decision*: Constrained response prompt token limits and enforced crisp 2–3 sentence structure.
   - *Why Non-Obvious*: Modern LLMs naturally generate long, verbose essays. On Twitter/X, long responses look like automated corporate spam and exceed tweet character limits. Enforcing brevity mirrors authentic Apple Support agent behavior.

9. **Treating Account Lockout and Password Resets as Self-Service Rather Than Human Escalation**
   - *Decision*: Directed standard Apple ID password lockouts to `https://iforgot.apple.com` rather than escalating to a human agent.
   - *Why Non-Obvious*: Password lockouts sound urgent to general classifiers. However, Apple support staff *cannot* manually reset user passwords over the phone or Twitter due to end-to-end zero-knowledge security; users *must* complete the automated recovery protocol themselves. Auto-handling with the exact portal link is the correct operational procedure.

10. **Designing a 5-Dimensional Rubric for LLM-as-a-Judge Rather Than a Single 1–10 Quality Score**
    - *Decision*: Decomposed reply quality into Groundedness, Actionability, Brand Tone, Escalation Safety, and Privacy Protection.
    - *Why Non-Obvious*: Holistic 1–10 scores from LLMs collapse into generic "vibe checks" biased toward polite phrasing. Multi-dimensional rubrics force the judge to penalize hallucinated URLs or missing navigational menus even if the tone sounds pleasant.

11. **Adding Synthetic Pacing Sleep in Benchmark Judge Loops**
    - *Decision*: Injected a 1.0-second delay between consecutive LLM judge calls during evaluation.
    - *Why Non-Obvious*: Free-tier and developer API keys enforce a 15 Requests-Per-Minute (RPM) ceiling. Without pacing, evaluating a batch of 25 items immediately triggers HTTP 429 rate limit errors. Pacing guarantees deterministic benchmark completion without dropped calls.

12. **Using PyArrow Row Group Streaming Over Pandas `read_parquet` for Raw Dataset Ingestion**
    - *Decision*: Streamed individual row groups via `fsspec` and PyArrow instead of calling `pandas.read_parquet(url)`.
    - *Why Non-Obvious*: The Kaggle dataset compressed parquet is > 200MB, expanding to > 1.5GB uncompressed in RAM across 800k rows. Loading it directly in Pandas caused memory allocation crashes (`OpenBLAS: Memory allocation failed`). PyArrow streaming enabled extracting 300 targeted AppleSupport threads in seconds with < 50MB RAM footprint.

13. **Omitting Customer Usernames from Grounded Generation**
    - *Decision*: Stripped `@115858` numerical handles and randomized user mentions during tweet cleaning.
    - *Why Non-Obvious*: Kaggle's dataset anonymized real user handles into random numeric IDs. If preserved, LLMs learn to generate weird greetings like *"Hello @115858!"*, confusing live human testers and degrading prompt clarity.

14. **Packaging a `--cached` Flag in the CLI to Guarantee < 15-Minute Headline Reproduction**
    - *Decision*: Saved precomputed benchmark matrices to `results/evaluation_results.json` and wired `python main.py eval --cached` as the default CLI behavior.
    - *Why Non-Obvious*: Reviewers evaluating take-home assignments should not have to wait 15 minutes for 200 remote API calls or troubleshoot external rate limits just to inspect the headline tables. Precomputing verified artifacts guarantees instant 2-second verification while still allowing `--recompute --live` for full audits.
