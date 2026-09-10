"""
Configuration module for the AI Customer Support Agent.
Loads environment variables, manages paths, defines taxonomy, and configures Gemini LLM & ChromaDB.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHROMA_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
GOLDEN_SET_PATH = DATA_DIR / "golden_eval_set.json"
GOLDEN_SET_CSV_PATH = DATA_DIR / "golden_eval_set.csv"
HISTORICAL_RESOLUTIONS_PATH = DATA_DIR / "historical_resolutions.json"
HUMAN_JUDGE_PATH = DATA_DIR / "human_judge_sample.json"
BASELINE_RESULTS_PATH = RESULTS_DIR / "baseline_eval.json"
FINAL_EVAL_RESULTS_PATH = RESULTS_DIR / "evaluation_results.json"

# API & Model Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODEL = "gemini-3.5-flash-lite"
DEFAULT_BRAND = os.getenv("DEFAULT_BRAND", "AppleSupport")

# Intent Taxonomy for AppleSupport
INTENTS = [
    "hardware_issue",
    "software_bug_update",
    "account_security",
    "billing_subscription",
    "product_inquiry_howto",
    "complaint_feedback",
]

INTENT_DESCRIPTIONS = {
    "hardware_issue": "Physical device damage, battery health/drain, charging port, cracked screen, speaker, microphone, or SIM card reader defects.",
    "software_bug_update": "iOS or macOS update glitches, app crashing, device freezing, reboot loops, Wi-Fi or Bluetooth software connectivity problems.",
    "account_security": "Apple ID locked, two-factor authentication issues, forgotten password, iCloud syncing errors, account recovery.",
    "billing_subscription": "App Store charges, in-app purchases, recurring subscription cancellations, refund disputes, declined payment methods.",
    "product_inquiry_howto": "Device specifications, release schedules, feature how-tos, settings configuration, compatibility, trade-in values.",
    "complaint_feedback": "Customer dissatisfaction with Apple retail store experience, wait times, repair costs, or general company policies without a technical ask."
}

# Triage & Escalation Categories
ESCALATION_ACTIONS = ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]

# Criteria requiring escalation to human
ESCALATION_TRIGGERS = [
    "Requires private user credentials, password reset, or account ownership verification (Apple ID security hazard)",
    "Physical hardware defect requiring hands-on Genius Bar appointment or mail-in hardware repair",
    "Financial dispute, disputed credit card charges, or refund requests over automated policy limit",
    "Severe customer frustration, repeated failed self-service troubleshooting, or threat of legal/executive complaint"
]
