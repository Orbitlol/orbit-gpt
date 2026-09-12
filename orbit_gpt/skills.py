"""Deterministic answers for questions a tiny model cannot work out.

A 1M-parameter transformer cannot reliably add two 2-digit numbers in a single
forward pass - it will happily invent "45 + 37 = 134".  Rather than let it be
confidently wrong, ``orbit`` answers arithmetic and unit conversions with
plain Python and lets the model handle everything else.

Everything here is deliberately conservative: if a sentence is not clearly a
calculation, :func:`answer_arithmetic` returns ``None`` and the model answers.
"""

from __future__ import annotations

import random
import re
from typing import Optional

__all__ = ["answer_arithmetic"]

# unit conversions: (source names, target names, factor)  result = value * factor
UNIT_CONVERSIONS = (
    (("km", "kms", "kilometre", "kilometres", "kilometer", "kilometers"),
     ("mile", "miles"), 0.621371),
    (("mile", "miles"), ("km", "kms", "kilometre", "kilometres", "kilometer",
                         "kilometers"), 1.609344),
    (("kg", "kgs", "kilogram", "kilograms"), ("pound", "pounds", "lb", "lbs"),
     2.204623),
    (("pound", "pounds", "lb", "lbs"), ("kg", "kgs", "kilogram", "kilograms"),
     0.4535924),
    (("m", "metre", "metres", "meter", "meters"), ("foot", "feet", "ft"),
     3.28084),
    (("foot", "feet", "ft"), ("m", "metre", "metres", "meter", "meters"),
     0.3048),
    (("cm", "centimetre", "centimetres", "centimeter", "centimeters"),
     ("inch", "inches", "in"), 0.3937008),
    (("inch", "inches", "in"), ("cm", "centimetre", "centimetres",
                                "centimeter", "centimeters"), 2.54),
    (("litre", "litres", "liter", "liters", "l"), ("gallon", "gallons", "gal"),
     0.2641721),
    (("gallon", "gallons", "gal"), ("litre", "litres", "liter", "liters", "l"),
     3.785412),
)

_OP_WORDS = {
    "plus": "+", "minus": "-", "times": "*", "multiplied by": "*", "x": "*",
    "divided by": "/", "over": "/",
}

# filler that surrounds the actual calculation
_FILLER = re.compile(
    r"^(?:and\s+)?(?:hey|hi|hello|ok|okay|so|well)?\s*"
    r"(?:and\s+what\s+about|what\s+about|how\s+about|"
    r"what\s+is|what\s+are|what's|whats|what\s+does|"
    r"how\s+much\s+is|how\s+many\s+is|"
    r"calculate|compute|evaluate|solve|convert|"
    r"can\s+you\s+(?:tell\s+me\s+|work\s+out\s+|do\s+|calculate\s+|compute\s+)?|"
    r"could\s+you\s+(?:tell\s+me\s+|work\s+out\s+)?|"
    r"tell\s+me\s+|work\s+out\s+)\s+",
    re.IGNORECASE,
)

_TEMPERATURES = {
    "c": "degrees Celsius", "celsius": "degrees Celsius",
    "f": "degrees Fahrenheit", "fahrenheit": "degrees Fahrenheit",
    "k": "kelvin", "kelvin": "kelvin",
}


def _to_celsius(unit: str, value: float) -> float:
    if unit.startswith("f"):
        return (value - 32) / 1.8
    if unit.startswith("k"):
        return value - 273.15
    return value


def _from_celsius(unit: str, value: float) -> float:
    if unit.startswith("f"):
        return value * 1.8 + 32
    if unit.startswith("k"):
        return value + 273.15
    return value


def _temperature(text: str) -> Optional[str]:
    names = "|".join(sorted(_TEMPERATURES, key=len, reverse=True))
    match = re.match(
        rf"^({_NUMBER})\s*(?:degrees?\s*)?({names})\s*"
        rf"(?:to|in|into|as)\s*(?:degrees?\s*)?({names})$",
        text,
    )
    if not match:
        return None
    value, source, target = float(match.group(1)), match.group(2), match.group(3)
    result = _from_celsius(target, _to_celsius(source, value))
    return (f"{_fmt_number(value)} {_TEMPERATURES[source]} is about "
            f"{_fmt_number(result)} {_TEMPERATURES[target]}.")

# unit words that must survive the filler strip
_UNIT_WORDS = "|".join(sorted(
    {w for pair in UNIT_CONVERSIONS for group in pair[:2] for w in group},
    key=len, reverse=True,
))

_NUMBER = r"[0-9]+(?:\.[0-9]+)?"


def _clean_text(text: str) -> str:
    """Lowercase, expand words, drop question filler."""
    text = text.lower().strip()
    text = text.replace("\u00d7", " * ").replace("\u2212", "-")
    text = text.replace("\u00f7", " / ").replace("^", " ^ ")
    for word, op in _OP_WORDS.items():
        text = re.sub(rf"\b{re.escape(word)}\b", f" {op} ", text)
    text = re.sub(r"[?!.,]+$", "", text.strip())
    prev = None
    while prev != text:  # strip leading filler such as "what is"
        prev = text
        match = _FILLER.match(text)
        if match and not re.match(rf"({_UNIT_WORDS})\b", text[match.end():] or " "):
            text = text[match.end():].strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip("?!.").strip()


def _fmt_number(value: float) -> str:
    rounded = round(value, 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:g}"


def _phrase(answer: str) -> str:
    """Answer in the voice the model learned from the corpus."""
    style = random.random()
    if style < 0.45:
        return f"{answer}."
    if style < 0.75:
        return f"The answer is {answer}."
    if style < 0.9:
        return f"It's {answer}."
    return f"That's {answer}."


def _convert(text: str) -> Optional[str]:
    match = re.match(
        rf"^({_NUMBER})\s*({_UNIT_WORDS})\s*(?:to|in|into|as)\s*({_UNIT_WORDS})$",
        text,
    )
    if not match:
        return None
    value, source, target = float(match.group(1)), match.group(2), match.group(3)
    for sources, targets, factor in UNIT_CONVERSIONS:
        if source in sources and target in targets:
            return (f"{_fmt_number(value)} {source} is about {_fmt_number(value * factor)} "
                    f"{target}.")
    return None


def answer_arithmetic(text: str) -> Optional[str]:
    """Return a correct answer for a calculation, or ``None`` to let the model speak.

    >>> answer_arithmetic("What is 45 + 37?")
    '45 + 37 = 82.'
    >>> answer_arithmetic("25% of 900") is None
    False
    >>> answer_arithmetic("tell me a joke") is None
    True
    """
    if not text or len(text) > 200:
        return None
    cleaned = _clean_text(text)

    # "how many pounds is 120 kilograms?"
    match = re.match(
        rf"^how many ({_UNIT_WORDS}) is ({_NUMBER})\s*({_UNIT_WORDS})$", cleaned
    )
    if match:
        target, value, source = match.group(1), float(match.group(2)), match.group(3)
        for sources, targets, factor in UNIT_CONVERSIONS:
            if source in sources and target in targets:
                return (f"{_fmt_number(value)} {source} is about {_fmt_number(value * factor)} "
                        f"{target}.")

    converted = _convert(cleaned) or _temperature(cleaned)
    if converted:
        return converted

    # "add 120 and 275" / "subtract 5 from 20" / "multiply 6 by 7" / "divide 9 by 3"
    match = re.match(rf"^add ({_NUMBER}) and ({_NUMBER})$", cleaned)
    if match:
        return _phrase(_fmt_number(float(match.group(1)) + float(match.group(2))))
    match = re.match(rf"^subtract ({_NUMBER}) from ({_NUMBER})$", cleaned)
    if match:
        return _phrase(_fmt_number(float(match.group(2)) - float(match.group(1))))
    match = re.match(rf"^multiply ({_NUMBER}) (?:by|and) ({_NUMBER})$", cleaned)
    if match:
        return _phrase(_fmt_number(float(match.group(1)) * float(match.group(2))))
    match = re.match(rf"^divide ({_NUMBER}) by ({_NUMBER})$", cleaned)
    if match:
        divisor = float(match.group(2))
        if divisor == 0:
            return "You can't divide by zero."
        return _phrase(_fmt_number(float(match.group(1)) / divisor))

    # "45 + 37" / "45 - 37" / "45 * 37" / "45 / 37"
    match = re.match(rf"^({_NUMBER})\s*([-+*/])\s*({_NUMBER})$", cleaned)
    if match:
        left, op, right = float(match.group(1)), match.group(2), float(match.group(3))
        if op == "+" or op == "-":
            result: float = left + right if op == "+" else left - right
            return _phrase(f"{_fmt_number(left)} {op} {_fmt_number(right)} = {_fmt_number(result)}"
                           if random.random() < 0.5 else _fmt_number(result))
        if op == "*":
            return _phrase(f"{_fmt_number(left)} \u00d7 {_fmt_number(right)} = {_fmt_number(left * right)}"
                           if random.random() < 0.5 else _fmt_number(left * right))
        if right == 0:
            return "You can't divide by zero."
        return _phrase(_fmt_number(left / right))

    # "25% of 900" / "25 percent of 900"
    match = re.match(rf"^({_NUMBER})\s*(?:%|percent)\s*of\s*({_NUMBER})$", cleaned)
    if match:
        percent, base = float(match.group(1)), float(match.group(2))
        result = base * percent / 100
        return _phrase(f"{_fmt_number(percent)}% of {_fmt_number(base)} = {_fmt_number(result)}"
                       if random.random() < 0.5 else _fmt_number(result))

    # "12 squared" / "12 cubed" / "12 ^ 2"
    match = re.match(rf"^({_NUMBER})\s*(?:\^|to the power of)\s*([23])$", cleaned)
    if match:
        base, power = float(match.group(1)), int(match.group(2))
        return _phrase(_fmt_number(base ** power))
    match = re.match(rf"^({_NUMBER})\s*(squared|cubed)$", cleaned)
    if match:
        base = float(match.group(1))
        power = 2 if match.group(2) == "squared" else 3
        return _phrase(_fmt_number(base ** power))

    # square root
    match = re.match(rf"^(?:square root of|sqrt)\s*({_NUMBER})$", cleaned)
    if match:
        base = float(match.group(1))
        if base < 0:
            return "The square root of a negative number is not a real number."
        return _phrase(_fmt_number(base ** 0.5))
    return None
