"""
Tests for AppleSupport Agent pipeline.
"""

import pytest
from src.agent import AppleSupportAgent, get_agent
from src.config import INTENTS, ESCALATION_ACTIONS


def test_agent_end_to_end_fast():
    agent = get_agent()
    out = agent.process("How do I turn on Closed Captions on Apple TV?", fast_mode=True)

    assert out.customer_message == "How do I turn on Closed Captions on Apple TV?"
    assert out.intent in INTENTS
    assert out.intent_confidence > 0.0
    assert out.escalation_action in ESCALATION_ACTIONS
    assert len(out.drafted_reply) > 5
    assert len(out.escalation_reason) > 5


def test_agent_hazardous_query():
    agent = get_agent()
    out = agent.process("My MacBook battery is smoking and burning hot!", fast_mode=True)

    assert out.escalation_action == "ESCALATE_TO_HUMAN"
    assert "safety" in out.escalation_reason.lower() or "thermal" in out.escalation_reason.lower() or "battery" in out.escalation_reason.lower()
