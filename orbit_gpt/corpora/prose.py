"""Plain-prose corpus for the pre-training stage of SFT.

The dialogue corpus (`conversation.py`) teaches *format*: it is all
"User: ... / Assistant: ...".  A tiny model trained only on that format never
learns how English sentences are built, which is why it can sound like word
salad the moment it has to write something longer than a canned reply.

This module re-uses the same knowledge (the fact / how-to / chit-chat answers)
but writes it as **flowing paragraphs** with subject-verb sentences,
punctuation, and connectors.  The two stages are:

    stage 1  pre-train on `prose`     -> sentence structure, grammar, fluency
    stage 2  SFT on `conversation`    -> the User:/Assistant: chat format

Deterministic, offline, and cheap: a few hundred kB to a few MB in well under
a second.
"""

from __future__ import annotations

import random
import re
from typing import List, Sequence

from orbit_gpt.corpora.conversation import (
    ADVICE,
    CHITCHAT,
    FACTS,
    FOLLOWUPS,
    GREETINGS,
    HOWTO,
    IDENTITY,
    META,
    UNKNOWNS,
    _pick,
)

__all__ = ["build_prose_corpus", "prose_stats"]

# connectors that glue sentences together (this is what teaches "real"
# sentence flow: capital letter, clause, full stop, next sentence)
_OPENERS = (
    "In practice, ", "Usually, ", "Most of the time, ", "In short, ",
    "To be precise, ", "Put simply, ", "On most projects, ", "As a rule, ",
    "If you are new to this, ", "A good way to think about it is this: ",
)
_LINKS = (
    " That said, ", " However, ", " Also, ", " In other words, ",
    " For example, ", " As a result, ", " On top of that, ", " Even so, ",
    " The reason is simple: ", " What matters most is this: ",
)
_CLOSERS = (
    " It is worth knowing because it comes up again and again.",
    " Once you have seen it a few times it becomes obvious.",
    " That is the whole idea; everything else is detail.",
    " Keep it in mind the next time you run into it.",
    " It sounds small, but it saves a lot of time later.",
    " You will see the same pattern in most other languages too.",
    " Practising it for ten minutes teaches more than reading about it.",
    " There are exceptions, but this is the normal case.",
)

def _paragraph(text: str) -> str:
    """Tidy a finished paragraph: capital start, single spaces."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    return (text[0].upper() + text[1:]) if text else text


def _s(text: str) -> str:
    """One clean sentence: capitalised, single full stop, no '..'."""
    text = (text or "").strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    text = re.sub(r"\s+", " ", text)
    while text.endswith(".."):
        text = text[:-1]
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def _two(rng: random.Random, answers: Sequence[str]) -> List[str]:
    """Two *different* sentences from the same answer bank."""
    pool = [a for a in answers if a]
    if not pool:
        return []
    first = _pick(rng, pool)
    rest = [a for a in pool if a != first] or pool
    return [_s(first), _s(_pick(rng, rest))]


_TOPIC_LEADS = (
    "{topic} is worth knowing about.",
    "Let's talk about {topic}.",
    "{topic} comes up all the time.",
    "People ask about {topic} a lot.",
    "{topic} sounds harder than it is.",
)


def _sentences(rng: random.Random, topic: str, answers: List[str]) -> str:
    """Turn a topic + its answer variants into a short, grammatical paragraph."""
    parts = [rng.choice(_TOPIC_LEADS).format(topic=topic)]
    parts.extend(_two(rng, answers))
    paragraph = " ".join(p for p in parts if p)
    if rng.random() < 0.25:
        paragraph += " " + rng.choice(_CLOSERS)
    return _paragraph(paragraph)


def _howto(rng: random.Random, task: str, answers: List[str]) -> str:
    parts = [f"To {task}, start with the basics."]
    parts.extend(_two(rng, answers))
    paragraph = " ".join(p for p in parts if p)
    if rng.random() < 0.25:
        paragraph += " " + rng.choice(_CLOSERS)
    return _paragraph(paragraph)


def _social(rng: random.Random) -> str:
    group = rng.choice((GREETINGS, IDENTITY, CHITCHAT, FOLLOWUPS, UNKNOWNS,
                        ADVICE, META))
    _, answers = rng.choice(group)
    paragraph = " ".join(p for p in _two(rng, answers) if p)
    if rng.random() < 0.3:
        paragraph += " " + rng.choice(_CLOSERS)
    return _paragraph(paragraph)


def build_prose_corpus(
    seed: int = 1337,
    n_facts: int = 6000,
    n_howto: int = 4000,
    n_social: int = 4000,
    paragraphs_per_block: int = 3,
) -> str:
    """Build a prose corpus.  Same ``seed`` -> same text, every time."""
    rng = random.Random(seed)
    blocks: List[str] = []

    # definition paragraphs
    for _ in range(n_facts):
        topic, answers = rng.choice(list(FACTS.items()))
        blocks.append(_sentences(rng, topic, answers))

    # how-to paragraphs
    for _ in range(n_howto):
        task, answers = rng.choice(list(HOWTO.items()))
        blocks.append(_howto(rng, task, answers))

    # chit-chat / advice paragraphs (teaches an informal, first-person voice)
    for _ in range(n_social):
        blocks.append(_social(rng))

    rng.shuffle(blocks)

    # group a few paragraphs per block: language models need to learn that
    # a paragraph ends and a new one begins
    out: List[str] = []
    for i in range(0, len(blocks), paragraphs_per_block):
        out.append("\n\n".join(blocks[i:i + paragraphs_per_block]))
    return "\n\n".join(out) + "\n"


def prose_stats(text: str) -> str:
    return f"{len(text):,} characters, ~{text.count('. '):,} sentences"
