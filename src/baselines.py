"""
Baseline models for comparative benchmarking:
1. Trivial Baseline: Majority-class intent predictor, static canned macro reply, naive rule escalation.
2. Simple Baseline: TF-IDF intent classifier, raw nearest-neighbor resolution without LLM grounding, basic sentiment escalation.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import INTENTS

logger = logging.getLogger(__name__)

TRIVIAL_CANNED_REPLY = (
    "Thank you for contacting Apple Support! Please try restarting your device "
    "and ensuring it is updated to the latest iOS version. Let us know if that helps!"
)


class TrivialBaseline:
    """
    Trivial baseline:
    - Intent: Constant majority class ('hardware_issue')
    - Reply: Static canned macro template
    - Escalation: Naive character-length rule
    """

    def __init__(self, majority_intent: str = "hardware_issue"):
        self.majority_intent = majority_intent

    def process(self, customer_message: str) -> Dict:
        # Naive rule: escalate if tweet is longer than 140 chars or contains 'broken'
        msg = customer_message.lower()
        if len(customer_message) > 140 or "broken" in msg:
            action = "ESCALATE_TO_HUMAN"
            reason = "Message exceeded length threshold or contained 'broken'."
        else:
            action = "AUTO_HANDLE"
            reason = "Standard message length auto-handled."

        return {
            "intent": self.majority_intent,
            "intent_confidence": 0.50,
            "drafted_reply": TRIVIAL_CANNED_REPLY,
            "escalation_action": action,
            "escalation_confidence": 0.50,
            "escalation_reason": reason,
            "model_name": "Trivial Baseline"
        }


class SimpleBaseline:
    """
    Simple baseline:
    - Intent: TF-IDF Bag-of-Words + Logistic Regression
    - Reply: Raw 1-NN historical resolution via TF-IDF cosine similarity (no LLM rewriting)
    - Escalation: Basic keyword match
    """

    def __init__(self, training_data: Optional[List[Dict]] = None):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.classifier = LogisticRegression(max_iter=200)
        self.historical_docs = []
        self.historical_replies = []
        self.tfidf_matrix = None
        self.is_fitted = False

        if training_data:
            self.fit(training_data)

    def fit(self, examples: List[Dict]):
        """Fits TF-IDF classifier and corpus memory."""
        texts = []
        labels = []
        for ex in examples:
            msg = ex.get("customer_message", "")
            lbl = ex.get("ground_truth_intent", "")
            if msg and lbl:
                texts.append(msg)
                labels.append(lbl)

            # Store for 1-NN reply retrieval
            if msg and ex.get("historical_reference_reply"):
                self.historical_docs.append(msg)
                self.historical_replies.append(ex.get("historical_reference_reply"))

        if texts and labels:
            X = self.vectorizer.fit_transform(texts)
            self.classifier.fit(X, labels)
            self.tfidf_matrix = self.vectorizer.transform(self.historical_docs)
            self.is_fitted = True
            logger.info(f"Fitted SimpleBaseline on {len(texts)} samples.")

    def process(self, customer_message: str) -> Dict:
        if not self.is_fitted:
            return TrivialBaseline().process(customer_message)

        # 1. Intent via TF-IDF + Logistic Regression
        vec = self.vectorizer.transform([customer_message])
        pred_intent = self.classifier.predict(vec)[0]
        probs = self.classifier.predict_proba(vec)[0]
        confidence = float(np.max(probs))

        # 2. Reply via 1-NN Cosine Similarity on TF-IDF
        if self.tfidf_matrix is not None and self.tfidf_matrix.shape[0] > 0:
            similarities = (self.tfidf_matrix * vec.T).toarray().flatten()
            best_idx = int(np.argmax(similarities))
            reply = self.historical_replies[best_idx]
        else:
            reply = TRIVIAL_CANNED_REPLY

        # 3. Simple Escalation: Basic keyword dictionary check
        msg_low = customer_message.lower()
        if any(w in msg_low for w in ["stolen", "lawsuit", "refund", "swollen", "shattered", "hacked", "police"]):
            action = "ESCALATE_TO_HUMAN"
            reason = "Matched urgent keyword in dictionary."
        else:
            action = "AUTO_HANDLE"
            reason = "No urgent keywords found in dictionary."

        return {
            "intent": pred_intent,
            "intent_confidence": round(confidence, 2),
            "drafted_reply": reply,
            "escalation_action": action,
            "escalation_confidence": 0.70,
            "escalation_reason": reason,
            "model_name": "Simple Baseline (TF-IDF + 1-NN)"
        }
