"""
Tests for Golden Evaluation Set integrity and formatting.
"""

import json
from pathlib import Path
import pytest
from src.config import GOLDEN_SET_PATH, INTENTS, ESCALATION_ACTIONS


def test_golden_set_exists_and_count():
    assert GOLDEN_SET_PATH.exists(), f"File {GOLDEN_SET_PATH} does not exist"
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Assignment requires 150-250 hand-labelled examples
    assert 150 <= len(data) <= 250, f"Expected between 150 and 250 examples, found {len(data)}"
    assert len(data) == 200, f"Expected exactly 200 examples, got {len(data)}"


def test_golden_set_schema_and_taxonomy():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {
        "id",
        "customer_message",
        "ground_truth_intent",
        "ground_truth_escalation",
        "ground_truth_escalation_reason",
        "historical_reference_reply"
    }

    intent_counts = {}
    for item in data:
        # Check required fields
        assert required_keys.issubset(item.keys()), f"Missing keys in item: {item.get('id')}"

        # Check intent is in taxonomy
        intent = item["ground_truth_intent"]
        assert intent in INTENTS, f"Invalid intent: {intent} in item {item.get('id')}"
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # Check escalation action is valid
        esc = item["ground_truth_escalation"]
        assert esc in ESCALATION_ACTIONS, f"Invalid escalation: {esc} in item {item.get('id')}"

        # Check non-empty strings
        assert len(item["customer_message"].strip()) > 5
        assert len(item["ground_truth_escalation_reason"].strip()) > 5
        assert len(item["historical_reference_reply"].strip()) > 5

    # Check that all 6 intents are represented
    assert len(intent_counts) == 6
    for intent in INTENTS:
        assert intent_counts[intent] >= 25, f"Intent {intent} has fewer than 25 examples"
