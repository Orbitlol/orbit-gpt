"""Checks for the deterministic arithmetic skill (run via tests/test_smoke.py)."""

from __future__ import annotations

import random
import re


def test_answer_arithmetic_is_actually_correct():
    from orbit_gpt.skills import answer_arithmetic

    rng = random.Random(0)
    for _ in range(400):
        x, y = rng.randint(1, 999), rng.randint(1, 99)
        for question, want in (
            (f"What is {x} + {y}?", x + y),
            (f"What is {x} - {y}?", x - y),
            (f"What is {x} * {y}?", x * y),
            (f"What is {x * y} / {y}?", x),
            (f"{x} plus {y}", x + y),
            (f"{x} times {y}", x * y),
            (f"add {x} and {y}", x + y),
            (f"subtract {y} from {x}", x - y),
            (f"divide {x * y} by {y}", x),
        ):
            got = answer_arithmetic(question)
            assert got is not None, question
            numbers = re.findall(r"-?[0-9]+(?:\.[0-9]+)?", got)
            assert numbers, (question, got)
            assert abs(float(numbers[-1]) - want) < 0.02, (question, got, want)


def test_answer_arithmetic_handles_percentages_powers_and_roots():
    from orbit_gpt.skills import answer_arithmetic

    assert answer_arithmetic("What is 25% of 900?") is not None
    assert "225" in answer_arithmetic("What is 25% of 900?")
    assert "144" in answer_arithmetic("12 squared")
    assert "12" in answer_arithmetic("square root of 144")
    assert "27" in answer_arithmetic("3 cubed")
    assert "zero" in answer_arithmetic("What is 5 / 0?").lower()


def test_answer_arithmetic_converts_units():
    from orbit_gpt.skills import answer_arithmetic

    assert "466" in answer_arithmetic("Convert 750 kilometres to miles.")
    assert "264" in answer_arithmetic("How many pounds is 120 kilograms?")
    assert "37.8" in answer_arithmetic("100 f to c") or "37.78" in answer_arithmetic("100 f to c")
    assert "212" in answer_arithmetic("100 c to f")


def test_answer_arithmetic_stays_out_of_the_way():
    """Anything that is not clearly a calculation belongs to the model."""
    from orbit_gpt.skills import answer_arithmetic

    for text in (
        "tell me a joke",
        "who made you?",
        "explain recursion",
        "what is the capital of France?",
        "I have 2 apples and 3 oranges",
        "what is 2 + 2 in Python?",
        "hi",
        "thanks",
        "Can you explain more?",
        "how do you work?",
        "",
    ):
        assert answer_arithmetic(text) is None, text
