# Golden Evaluation Set Methodology: AppleSupport

## 1. Overview & Objectives
To rigorously evaluate the AI Customer Support Agent for `@AppleSupport`, we constructed a benchmark dataset of **200 hand-labelled examples** derived directly from the Kaggle Customer Support on Twitter dataset (`thoughtvector/customer-support-on-twitter`) and representative real-world customer service interactions.

The primary goal of this evaluation set is to provide a reliable ground truth for:
1. **Intent Classification**: Across a 6-intent taxonomy tailored to Apple's support volume.
2. **Triage & Escalation Decision**: Distinguishing issues safe for automated resolution (`AUTO_HANDLE`) vs. those requiring human specialists (`ESCALATE_TO_HUMAN`).
3. **Escalation Reasoning**: Verifying whether the system generates justifiable and operationally sound rationale for human handoff.
4. **Historical Grounding & Response Quality**: Providing authentic historical reference resolutions for automated metrics (BLEU, ROUGE) and LLM-as-a-judge alignment.

---

## 2. Sampling Strategy & Filtering

### Source Selection
- **Corpus**: Extracted from multi-turn AppleSupport threads in the Kaggle dataset.
- **Inclusion Criteria**:
  - Genuine inbound customer messages with concrete technical, transactional, or feedback intent.
  - Presence of clear historical brand responses from AppleSupport (`^Support` agents) with verified Apple Knowledge Base links (`support.apple.com`, `appleid.apple.com`, `iforgot.apple.com`).
  - High linguistic diversity: covering varied phrasing, colloquial slang, user distress, and differing device generations (iPhone 6s through iPhone 15/16, M-series Macs, Apple Watch, and iPad).
- **Exclusion Criteria**:
  - Single-word messages (e.g., "thanks", "hello", "ok").
  - Bot spam, gibberish strings, or incomplete truncated tweets.
  - Messages with private PII (all personal handles and phone numbers were anonymized).

### Stratification Matrix
The 200 examples were systematically stratified to prevent class imbalance:

| Intent Category | Count | Primary Focus | Escalation Split (Auto / Escalate) |
| :--- | :---: | :--- | :---: |
| `hardware_issue` | 35 | Battery health, swollen cells, screen cracks, charging ports, acoustic faults | 16 Auto / 19 Escalate |
| `software_bug_update` | 35 | iOS/macOS bugs, boot loops, update verification errors, app crashes, Wi-Fi/BT | 30 Auto / 5 Escalate |
| `account_security` | 35 | Apple ID locked, 2FA failures, foreign compromise, Activation Lock, Digital Legacy | 27 Auto / 8 Escalate |
| `billing_subscription` | 35 | Accidental charges, minor IAPs, duplicate billing, refunds, subscription cancellations | 25 Auto / 10 Escalate |
| `product_inquiry_howto` | 30 | Compatibility, feature how-tos (Face ID mask, Back Tap, eSIM, AirDrop), specs | 30 Auto / 0 Escalate |
| `complaint_feedback` | 30 | Store wait times, repair damage, delivery delays, pricing critiques, legal threats | 16 Auto / 14 Escalate |
| **Total** | **200** | **Comprehensive Apple Ecosystem Coverage** | **144 Auto (72%) / 56 Escalate (28%)** |

---

## 3. Annotation Guidelines & Taxonomy Definitions

Each example was independently annotated across four ground-truth attributes:

### Attribute 1: `ground_truth_intent`
- `hardware_issue`: Physical damage, battery swelling/degradation, acoustic speaker crackle, mechanical button failure, liquid ingress, SIM tray malfunction.
- `software_bug_update`: Operating system anomalies, boot loops, update download verification timeouts, kernel panics, Wi-Fi/Bluetooth software glitches.
- `account_security`: Apple ID authentication, credential lockout, two-factor authentication recovery, phishing scams, Activation Lock, deceased family accounts.
- `billing_subscription`: Charges appearing on statements (`ITUNES.COM/BILL`), accidental in-app purchases, recurring subscription cancellations, refund disputes, declined cards.
- `product_inquiry_howto`: Feature walk-throughs, configuration tutorials, compatibility questions, trade-in valuations, accessory support.
- `complaint_feedback`: Negative service interactions, retail store grievances, repair transit damage, pricing venting, policy dissatisfaction, legal threats.

### Attribute 2: `ground_truth_escalation` (`AUTO_HANDLE` vs. `ESCALATE_TO_HUMAN`)
The escalation policy reflects Apple's real-world operational security boundaries:
- **Criteria for `AUTO_HANDLE`**:
  - The query can be resolved via standard self-service troubleshooting (e.g. force restart, toggling settings, clearing Safari cache).
  - The inquiry requires directing the user to official public documentation (`https://support.apple.com/...`).
  - The user seeks product specifications, compatibility tables, or return window policies.
- **Criteria for `ESCALATE_TO_HUMAN`**:
  - **Account Privacy Hazard**: Issues requiring private identity documents, account takeover investigation, or 2FA recovery where public tweet assistance is unsafe.
  - **Hardware Safety / Physical Inspection**: Swollen batteries (fire hazard), liquid ingress, cracked screens, optical stabilization failure requiring physical Genius Bar tools.
  - **Financial Transactions**: High-dollar unauthorized charges, duplicate billing ledger corrections, disputed refund rejections.
  - **Legal / Severe Customer Distress**: Formal legal action threats, retail discrimination allegations, or carrier theft/lost packages.

### Attribute 3: `ground_truth_escalation_reason`
For every example, a concise, human-authored explanation is provided detailing *why* the message is or is not safe for automated resolution.

### Attribute 4: `human_quality_score`
A 1–5 rating assessing the quality, groundedness, and brand empathy of the historical resolution, used for validating the LLM-as-a-Judge correlation.

---

## 4. Quality Control & Annotation Protocol
1. **Edge Case Resolution**: Multi-intent queries (e.g., "I updated iOS and my phone was charged twice") were tagged by their primary actionable blocker.
2. **Escalation Safety Bias**: In borderline scenarios involving account security or battery safety, annotators were instructed to favor `ESCALATE_TO_HUMAN` to mirror high-reliability customer support engineering.
3. **Format Validation**: Dataset integrity is validated via automated schema checks in `tests/test_golden_set.py`.
