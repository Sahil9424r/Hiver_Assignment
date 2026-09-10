"""
End-to-End AI Customer Support Agent for AppleSupport.
Orchestrates:
1. Intent Classification
2. Semantic Retrieval of Historical Resolutions from ChromaDB Vector Store
3. Grounded Reply Generation using Gemini (with brand-tone alignment & safety guardrails)
4. Triage & Escalation Decision with Stated Rationale
"""

import json
import logging
import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from google import genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL, FALLBACK_MODEL
from src.classifier import get_classifier, IntentClassifier
from src.vector_store import get_vector_store, ChromaResolutionStore
from src.escalation import get_escalation_engine, EscalationEngine

logger = logging.getLogger(__name__)

DRAFT_REPLY_SYSTEM_PROMPT = """You are the official Twitter customer support agent for @AppleSupport.
Your goal is to draft a helpful, concise, and empathetic reply grounded in how Apple Support has historically resolved similar issues.

CRITICAL GUIDELINES:
1. Brand Voice: Calm, empathetic, professional, and courteous (e.g. "We'd like to help with this.", "We're happy to help.").
2. Conciseness: Keep responses crisp and actionable (ideally under 280 characters or 2-3 short sentences, matching Twitter format).
3. Grounding: You are provided with retrieved historical Apple resolutions. Ground your response in their advice and links.
4. Official Apple Links: Use official Apple Support domains (e.g. https://support.apple.com/..., https://iforgot.apple.com, https://reportaproblem.apple.com, https://appleid.apple.com).
5. Privacy & Security: NEVER ask for passwords, credit card numbers, or full Apple IDs in public. If sensitive assistance or device inspection is required, invite them to DM or visit an Apple Store Genius Bar.
6. If the issue is escalated to a human or requires private info, politely invite the user to send a Direct Message (DM).
"""


class AgentOutput(BaseModel):
    customer_message: str
    intent: str
    intent_confidence: float
    intent_reasoning: str
    drafted_reply: str
    escalation_action: str
    escalation_confidence: float
    escalation_reason: str
    retrieved_resolutions: List[Dict] = Field(default_factory=list)


class AppleSupportAgent:
    """
    Unified AI Support Agent combining intent classification, vector retrieval,
    response generation, and triage escalation.
    """

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        vector_store: Optional[ChromaResolutionStore] = None,
        escalation_engine: Optional[EscalationEngine] = None,
        model: Optional[str] = None
    ):
        self.classifier = classifier or get_classifier()
        self.vector_store = vector_store or get_vector_store()
        self.escalator = escalation_engine or get_escalation_engine()
        self.model = model or GEMINI_MODEL
        self.client = None
        if GEMINI_API_KEY:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client in agent: {e}")

    def process(
        self,
        customer_message: str,
        top_k: int = 2,
        fast_mode: bool = False
    ) -> AgentOutput:
        """
        Processes an incoming customer message end-to-end.
        """
        # 1. Intent Classification
        clf_result = self.classifier.classify(customer_message, fast_mode=fast_mode)

        intent = clf_result["intent"]
        intent_conf = clf_result["confidence"]
        intent_reasoning = clf_result["reasoning"]

        # 2. Retrieve Historical Resolutions from ChromaDB Vector Store
        retrieved = self.vector_store.search_similar_resolutions(
            query=customer_message,
            top_k=top_k
        )

        # 3. Triage & Escalation Decision
        esc_result = self.escalator.evaluate(
            message=customer_message,
            intent=intent,
            confidence=intent_conf
        )
        escalation_action = esc_result["action"]
        escalation_conf = esc_result["confidence"]
        escalation_reason = esc_result["reason"]

        # 4. Draft Grounded Response
        if fast_mode and retrieved:
            drafted_reply = retrieved[0]["support_reply"]
        else:
            drafted_reply = self._draft_reply(
                customer_message=customer_message,
                intent=intent,
                escalation_action=escalation_action,
                retrieved=retrieved
            )

        return AgentOutput(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=intent_conf,
            intent_reasoning=intent_reasoning,
            drafted_reply=drafted_reply,
            escalation_action=escalation_action,
            escalation_confidence=escalation_conf,
            escalation_reason=escalation_reason,
            retrieved_resolutions=retrieved
        )

    def _draft_reply(
        self,
        customer_message: str,
        intent: str,
        escalation_action: str,
        retrieved: List[Dict]
    ) -> str:
        """
        Drafts a response using Gemini grounded in retrieved resolutions, or fallback template.
        """
        # Build prompt context from retrieved resolutions
        evidence_text = ""
        if retrieved:
            evidence_text = "Historical Similar Resolutions from AppleSupport:\n"
            for i, r in enumerate(retrieved, 1):
                evidence_text += f"{i}. Customer: {r['matched_customer_query']}\n   Support: {r['support_reply']}\n"

        prompt = f"""{DRAFT_REPLY_SYSTEM_PROMPT}

Customer Tweet: "{customer_message}"
Detected Intent: {intent}
Triage Action: {escalation_action}

{evidence_text}

Draft the official AppleSupport reply tweet:"""

        if self.client:
            models_to_try = [self.model, FALLBACK_MODEL]
            for m in models_to_try:
                try:
                    res = self.client.models.generate_content(
                        model=m,
                        contents=prompt
                    )
                    reply = res.text.strip()
                    # Strip any surrounding quotes
                    if reply.startswith('"') and reply.endswith('"'):
                        reply = reply[1:-1].strip()
                    return reply
                except Exception as e:
                    logger.warning(f"LLM draft generation failed on {m}: {e}")

        # Fallback template if LLM is unavailable
        if retrieved:
            return retrieved[0]["support_reply"]

        if escalation_action == "ESCALATE_TO_HUMAN":
            return "We'd like to gather more details to assist with this issue. Please send us a DM with your device model and iOS version so we can help: https://apple.co/DM"
        else:
            return "We're happy to help with this! Please check our support article for troubleshooting steps: https://support.apple.com or DM us if you need further guidance."


# Global singleton
_agent_instance = None


def get_agent() -> AppleSupportAgent:
    """Singleton getter for AppleSupportAgent."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AppleSupportAgent()
    return _agent_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = get_agent()

    test_queries = [
        "My iPhone 12 battery drains from 100% to 20% in an hour after the iOS update.",
        "Someone in Russia logged into my Apple ID account, please help!",
        "How do I set up Face ID with a mask?",
        "I was charged $4.99 for an in-app subscription I didn't authorize."
    ]

    for q in test_queries:
        out = agent.process(q)
        print(f"\n==========================================")
        print(f"Customer: {out.customer_message}")
        print(f"Intent: {out.intent} ({out.intent_confidence}) - {out.intent_reasoning}")
        print(f"Triage: {out.escalation_action} ({out.escalation_confidence})")
        print(f"Reason: {out.escalation_reason}")
        print(f"Reply: {out.drafted_reply}")
        if out.retrieved_resolutions:
            print(f"Top Retrieved Evidence: {out.retrieved_resolutions[0]['support_reply'][:100]}...")
