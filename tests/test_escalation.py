"""
Tests for Triage and Escalation Engine.
"""

import pytest
from src.escalation import EscalationEngine, get_escalation_engine


def test_escalation_hazardous_hardware():
    engine = get_escalation_engine()
    # Battery swelling is a severe hazard
    res = engine.evaluate(
        message="My iPad screen is physically bulging and the battery is swollen!",
        intent="hardware_issue"
    )
    assert res["action"] == "ESCALATE_TO_HUMAN"
    assert "battery" in res["reason"].lower() or "safety" in res["reason"].lower()
    assert res["confidence"] >= 0.90


def test_escalation_account_takeover():
    engine = get_escalation_engine()
    res = engine.evaluate(
        message="Someone in Russia hacked my Apple ID and changed my recovery number!",
        intent="account_security"
    )
    assert res["action"] == "ESCALATE_TO_HUMAN"
    assert res["confidence"] >= 0.90


def test_escalation_standard_howto():
    engine = get_escalation_engine()
    res = engine.evaluate(
        message="How do I turn on Back Tap on my iPhone 14?",
        intent="product_inquiry_howto"
    )
    assert res["action"] == "AUTO_HANDLE"
    assert res["confidence"] >= 0.90


def test_escalation_standard_billing():
    engine = get_escalation_engine()
    res = engine.evaluate(
        message="How do I cancel my Apple TV subscription?",
        intent="billing_subscription"
    )
    assert res["action"] == "AUTO_HANDLE"
