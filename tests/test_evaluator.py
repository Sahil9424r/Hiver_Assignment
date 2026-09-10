"""
Tests for Evaluation metrics: BLEU-4, ROUGE-L, and harness functions.
"""

import pytest
from src.evaluator import compute_bleu_4, compute_rouge_l, get_evaluator


def test_bleu_and_rouge_identical():
    text = "We'd like to help you with this. Please DM us your device model: https://apple.co"
    bleu = compute_bleu_4(text, text)
    rouge = compute_rouge_l(text, text)

    assert bleu >= 0.99
    assert rouge >= 0.99


def test_bleu_and_rouge_different():
    ref = "We'd like to help you with this. Please DM us your device model: https://apple.co"
    cand = "Completely unrelated text about baking cakes and buying flowers."
    bleu = compute_bleu_4(ref, cand)
    rouge = compute_rouge_l(ref, cand)

    assert bleu < 0.10
    assert rouge < 0.20


def test_evaluator_singleton():
    harness = get_evaluator()
    assert harness is not None
