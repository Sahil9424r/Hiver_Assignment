"""
Intent Classifier Module for AppleSupport.
Classifies customer messages into the 6-intent taxonomy with confidence scores.
Supports Gemini LLM-based classification with few-shot guidance, with fallback.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from google import genai

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    FALLBACK_MODEL,
    INTENTS,
    INTENT_DESCRIPTIONS
)

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """You are an expert customer-support triage specialist for Apple Support on Twitter.
Your task is to classify an incoming customer tweet into EXACTLY ONE of the following 6 intents:

1. hardware_issue: Physical damage, battery health/drain/swelling, screen, camera, speaker, microphone, charging port, button, SIM card issues.
2. software_bug_update: iOS/macOS update glitches, boot loops, crashes, freezing, Wi-Fi/Bluetooth software bugs, system data bloat.
3. account_security: Apple ID locked, 2FA codes, forgotten passwords, iCloud access, phishing, Activation Lock, hacked accounts.
4. billing_subscription: Charges, accidental purchases, refunds, subscription cancellations, declined payment methods.
5. product_inquiry_howto: How-to guides, feature instructions, device specifications, release dates, trade-in values, compatibility.
6. complaint_feedback: Dissatisfaction with store staff, service delays, repair costs, company policies, or legal complaints without technical asks.

Output ONLY valid JSON with keys:
- "intent": one of the 6 exact intent strings above
- "confidence": float between 0.0 and 1.0
- "reasoning": brief one-sentence justification
"""

# Few-shot representative examples to ensure prompt stability
FEW_SHOT_PROMPT = """
Examples:
Tweet: "My iPhone 7 battery percentage drops from 80% to 15% in twenty minutes and gets hot."
Response: {"intent": "hardware_issue", "confidence": 0.98, "reasoning": "Battery drain and physical thermal heat indicate hardware battery cell degradation."}

Tweet: "Ever since updating to iOS 17.2, Messages crashes every time I tap on a group chat."
Response: {"intent": "software_bug_update", "confidence": 0.96, "reasoning": "App crash triggered directly following an iOS operating system update."}

Tweet: "My Apple ID is locked for security reasons and I can't access my iCloud email."
Response: {"intent": "account_security", "confidence": 0.99, "reasoning": "Apple ID account lockout and authentication credential issue."}

Tweet: "Apple charged my card $79.99 for ITUNES.COM/BILL and I don't recognize it!"
Response: {"intent": "billing_subscription", "confidence": 0.98, "reasoning": "Unexpected financial charge on bank statement from Apple billing."}

Tweet: "How do I take a scrolling full-page screenshot in Safari on iPhone?"
Response: {"intent": "product_inquiry_howto", "confidence": 0.99, "reasoning": "Request for operational how-to tutorial regarding built-in iOS feature."}

Tweet: "I waited 2 hours at the Regent Street Apple Store and the staff was extremely rude!"
Response: {"intent": "complaint_feedback", "confidence": 0.97, "reasoning": "Customer complaint regarding retail store wait time and employee attitude."}
"""


class IntentClassifier:
    """
    Classifies incoming customer tweets into defined AppleSupport taxonomy.
    Supports LLM classification, trained ML classification (TF-IDF + LogReg), and heuristics.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL
        self.client = None
        self.tfidf = None
        self.lr_model = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client: {e}")

    def fit(self, examples: List[Dict]):
        """Trains internal ML classifier on provided examples."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        texts = [ex["customer_message"] for ex in examples if "customer_message" in ex]
        labels = [ex.get("ground_truth_intent", ex.get("intent")) for ex in examples]
        if texts and labels:
            self.tfidf = TfidfVectorizer(max_features=1500, stop_words="english", ngram_range=(1, 2))
            X = self.tfidf.fit_transform(texts)
            self.lr_model = LogisticRegression(max_iter=300, class_weight="balanced")
            self.lr_model.fit(X, labels)
            logger.info(f"IntentClassifier trained on {len(texts)} samples.")

    def classify(self, message: str, fast_mode: bool = False) -> Dict:
        """
        Classifies the incoming message into an intent.
        Returns a dict: {"intent": str, "confidence": float, "reasoning": str}
        """
        if not message or not message.strip():
            return {"intent": "product_inquiry_howto", "confidence": 0.5, "reasoning": "Empty input fallback."}

        # Fast mode or trained model priority
        if fast_mode:
            if self.lr_model and self.tfidf:
                vec = self.tfidf.transform([message])
                pred = self.lr_model.predict(vec)[0]
                prob = float(np.max(self.lr_model.predict_proba(vec)))
                return {
                    "intent": pred,
                    "confidence": round(prob, 2),
                    "reasoning": f"Classified via fine-tuned model (confidence: {round(prob, 2)})."
                }
            return self._heuristic_classify(message)

        # Try LLM classification first
        if self.client:
            try:
                return self._llm_classify(message)
            except Exception as e:
                logger.warning(f"Gemini intent classification failed: {e}. Falling back to ML/heuristic classifier.")

        # Fallback to trained ML model if available
        if self.lr_model and self.tfidf:
            vec = self.tfidf.transform([message])
            pred = self.lr_model.predict(vec)[0]
            prob = float(np.max(self.lr_model.predict_proba(vec)))
            return {
                "intent": pred,
                "confidence": round(prob, 2),
                "reasoning": f"Classified via fine-tuned model (confidence: {round(prob, 2)})."
            }

        # Fallback to heuristic rule-based classification
        return self._heuristic_classify(message)

    def _llm_classify(self, message: str) -> Dict:
        """Calls Gemini to predict intent with fallback model and retry."""
        import time
        prompt = f"{INTENT_SYSTEM_PROMPT}\n{FEW_SHOT_PROMPT}\nTweet: \"{message}\"\nResponse:"
        
        models_to_try = [self.model, FALLBACK_MODEL]
        last_error = None

        for m in models_to_try:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=m,
                        contents=prompt
                    )
                    raw_text = response.text.strip()
                    cleaned_json = re.sub(r"^```json\s*", "", raw_text)
                    cleaned_json = re.sub(r"\s*```$", "", cleaned_json).strip()

                    data = json.loads(cleaned_json)
                    intent = data.get("intent", "").lower().strip()
                    confidence = float(data.get("confidence", 0.90))
                    reasoning = data.get("reasoning", "")

                    if intent in INTENTS:
                        return {
                            "intent": intent,
                            "confidence": round(confidence, 2),
                            "reasoning": reasoning
                        }
                    else:
                        for candidate in INTENTS:
                            if candidate in intent:
                                return {"intent": candidate, "confidence": 0.80, "reasoning": reasoning}
                except Exception as e:
                    last_error = e
                    time.sleep(0.5)

        logger.warning(f"All LLM attempts failed ({last_error}). Using heuristic classifier.")
        return self._heuristic_classify(message)

    def _heuristic_classify(self, message: str) -> Dict:
        """
        Robust heuristic rule-based intent classifier with regex patterns.
        """
        msg = message.lower()

        # Account Security
        if any(w in msg for w in ["apple id", "icloud lock", "iforgot", "password", "passcode", "2fa", "two-factor", "verification code", "activation lock", "hacked", "stolen id", "security question", "recovery contact", "disabled in the app store", "safety check", "phishing"]):
            return {"intent": "account_security", "confidence": 0.88, "reasoning": "Detected account security, credential, or authentication keywords."}

        # Billing & Subscription
        if any(w in msg for w in ["charged", "charge", "billed", "bill", "subscription", "refund", "itunes.com/bill", "payment declined", "apple music", "in-app purchase", "robux", "renew", "invoice", "debit card", "credit card", "pro-rated", "pro rated"]):
            return {"intent": "billing_subscription", "confidence": 0.88, "reasoning": "Detected financial, subscription, or billing keywords."}

        # Hardware Issue
        if re.search(r"\b(battery|screen|display|camera|speaker|microphone|mic|charging port|charger|lightning|magsafe|cable|button|chassis|jack|fan|fans|trackpad|keyboard|lens|digitizer)\b", msg) and any(w in msg for w in ["drain", "drop", "dropped", "die", "hot", "heat", "scorch", "shatter", "crack", "static", "muffle", "stuck", "loose", "wobbly", "swollen", "bulg", "dent", "broken", "black line", "green line", "water", "liquid", "buzz"]):
            return {"intent": "hardware_issue", "confidence": 0.88, "reasoning": "Detected physical device or hardware component defect."}
        if any(w in msg for w in ["hardware diagnostic", "genius bar repair", "swollen battery", "shattered screen", "earpiece speaker", "no sim"]):
            return {"intent": "hardware_issue", "confidence": 0.85, "reasoning": "Detected hardware repair / component failure."}

        # Software Bug / Update
        if any(w in msg for w in ["ios", "macos", "watchos", "ipados", "update", "updated", "updating", "crashes", "crashing", "crash", "boot loop", "frozen", "freezing", "freeze", "spinning wheel", "wi-fi", "wifi", "bluetooth greyed", "system data", "glitch", "kernel panic", "restore", "error 54", "error -54", "recovery mode"]):
            return {"intent": "software_bug_update", "confidence": 0.85, "reasoning": "Detected operating system update, firmware, or software bug terms."}

        # Complaint & Feedback
        if any(w in msg for w in ["worst", "rude", "unhelpful", "unacceptable", "terrible service", "scam", "rip-off", "rip off", "lawsuit", "complaint", "store staff", "hung up", "disgusted", "thieves", "corporate greed", "stole my", "lied", "hate"]):
            return {"intent": "complaint_feedback", "confidence": 0.82, "reasoning": "Detected high-sentiment customer complaint or dissatisfaction language."}

        # Semantic Vector Store Fallback (ChromaDB)
        try:
            from src.vector_store import get_vector_store
            store = get_vector_store()
            if store.count() > 0:
                hits = store.search_similar_resolutions(message, top_k=1)
                if hits and hits[0].get("intent") in INTENTS:
                    return {
                        "intent": hits[0]["intent"],
                        "confidence": round(hits[0].get("similarity_score", 0.80), 2),
                        "reasoning": f"Semantically matched historical resolution in ChromaDB (similarity: {hits[0].get('similarity_score', 0.80)})."
                    }
        except Exception as e:
            logger.debug(f"ChromaDB fallback in classifier failed: {e}")

        # Default to product inquiry / how-to
        return {"intent": "product_inquiry_howto", "confidence": 0.75, "reasoning": "General product inquiry, compatibility question, or tutorial request."}


# Global singleton
_classifier_instance = None


def get_classifier() -> IntentClassifier:
    """Singleton getter for IntentClassifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    classifier = get_classifier()
    test_msgs = [
        "My iPhone 14 battery dropped 25% in 30 minutes and gets burning hot.",
        "How do I scan documents using Notes app?",
        "Why was my card billed $9.99 for Apple TV?",
        "Someone hacked my Apple ID and changed my recovery number!"
    ]
    for m in test_msgs:
        res = classifier.classify(m)
        print(f"Message: {m}\nResult: {res}\n" + "-"*40)
