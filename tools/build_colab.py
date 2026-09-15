#!/usr/bin/env python3
"""Bundle the orbit_gpt package into one self-contained Colab file.

``colab/orbit_gpt_colab.py`` is generated from the real package sources so the
copy-paste version can never drift from the repo version:

    python tools/build_colab.py

The generated file only needs PyTorch, embeds the built-in assistant corpus,
and can be dropped into a Google Colab cell (or run locally with
``python colab/orbit_gpt_colab.py --help``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "orbit_gpt_colab.py"

MODULES = [
    ("configuration", ROOT / "orbit_gpt" / "config.py"),
    ("tokenizer (byte-level BPE, pure Python)", ROOT / "orbit_gpt" / "tokenizer.py"),
    ("model (GPT transformer)", ROOT / "orbit_gpt" / "model.py"),
    ("data (corpora + batching)", ROOT / "orbit_gpt" / "data.py"),
    ("optional web search", ROOT / "orbit_gpt" / "search.py"),
    ("checkpoint locations", ROOT / "orbit_gpt" / "checkpoints.py"),
    ("generated conversation corpus", ROOT / "orbit_gpt" / "corpora" / "conversation.py"),
    ("generated prose corpus (stage 1)", ROOT / "orbit_gpt" / "corpora" / "prose.py"),
    ("deterministic arithmetic skill", ROOT / "orbit_gpt" / "skills.py"),
    ("training loop", ROOT / "orbit_gpt" / "train.py"),
    ("generation + chat", ROOT / "orbit_gpt" / "generate.py"),
]

HEADER = '''"""
OrbitGPT - a tiny GPT you can train in Google Colab in a couple of minutes.
==========================================================================

HOW TO USE (30 seconds, no install):

  1. Open https://colab.research.google.com and create a new notebook.
  2. Runtime -> Change runtime type -> T4 GPU  (CPU also works, just slower).
  3. Paste this whole file into ONE cell and run it.
     (Or upload this file and run:  !python orbit_gpt_colab.py)
  4. Wait for training, then chat with your model in the box at the bottom.

Edit the CONFIG block below to change the corpus, model size or chat style.

This file is generated from the orbit-gpt package by tools/build_colab.py -
edit the package, not this file.
"""

# ---------------------------------------------------------------------------
# CONFIG - change these, then re-run the cell
# ---------------------------------------------------------------------------
CONFIG = dict(
    # ---- what to learn -------------------------------------------------
    preset="micro",       # nano|micro|mini|small|base  (micro = 4.8M params)
    vocab_size=None,      # None = the preset default (1024 nano / 2048 micro)
    block_size=None,      # context length (None = preset default)
    n_layer=None, n_head=None, n_embd=None,   # override the preset if you like
    batch_size=None,      # None = auto for your hardware
    max_steps=None,       # None = auto (2000)
    seed=1337,

    # ---- supervised fine-tuning (two stages, one run) -------------------
    # Stage 1 teaches sentence structure on plain prose, stage 2 teaches the
    # User:/Assistant: chat format.  Skipping stage 1 is what makes a tiny
    # model sound like word salad.
    pretrain_corpus="prose",      # stage 1 corpus (generated, offline)
    pretrain_epochs=2,            # passes over the prose
    pretrain_fraction=0.4,        # share of the step budget for stage 1
    pretrain_steps=None,          # or set it directly
    sft_corpus="conversation",    # stage 2 corpus (generated, offline)
    sft_epochs=8,                 # cap so it does not memorise the dialogues
    sft_steps=None,               # or set it directly
    sft_lr=None,                  # None = preset lr / 3
    weight_decay=0.1,
    grad_clip=1.0,
    dropout=0.1,

    # ---- checkpoints (Google Drive is never used) -----------------------
    # "" = automatic: <repo>/checkpoints/orbit, or /content/checkpoints/orbit
    # when this file runs standalone in Colab.
    out_dir="",
    save_interval=250,    # also write model-latest.pt every N steps
    retrain=False,        # True = ignore the saved model and train again

    # ---- inference ------------------------------------------------------
    use_web_search=True,  # False (or --no-search) = never touch the network
    web_results=5,        # how many search results to put in the prompt
    sample_after_train=True,
    chat_after_train=True,
    chat_temperature=0.7,
    chat_tokens=160,
    chat_repetition_penalty=1.15,   # >1 stops it looping on the same words
)

# ---------------------------------------------------------------------------
# Imports - PyTorch is the only dependency (Colab already has it)
# ---------------------------------------------------------------------------
import argparse
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    sys.exit("PyTorch is not installed. In Colab run: !pip install torch")
'''

FOOTER = (Path(__file__).parent / "colab_footer.py").read_text(encoding="utf-8")


# Names that are allowed to appear in more than one module of the single-file
# build.  ``load_source`` is deliberately replaced by the footer; ``__all__``
# is inert once everything lives in one namespace.
ALLOWED_COLLISIONS = {"load_source", "_original_load_source", "__all__", "main"}


def top_level_names(source: str) -> set:
    """Every name a module defines at the top level."""
    import ast

    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def check_for_collisions(bodies, footer_source: str) -> None:
    """The build flattens every module into ONE namespace: a shared name means
    the later module silently replaces the earlier one's function or table."""
    from collections import defaultdict

    owners: dict = defaultdict(set)
    for title, body in bodies.items():
        for name in top_level_names(body):
            owners[name].add(title)
    for name in top_level_names(footer_source):
        owners[name].add("<footer>")

    clashes = {
        name: sorted(titles)
        for name, titles in owners.items()
        if len(titles) > 1 and name not in ALLOWED_COLLISIONS
    }
    if clashes:
        detail = "; ".join(f"{n} in {', '.join(t)}" for n, t in sorted(clashes.items()))
        raise SystemExit(
            "the single-file build flattens all modules, so these names clash: "
            f"{detail}\nRename them in the package and rebuild."
        )


def strip_imports(source: str) -> str:
    """Drop top-level import statements (the merged header provides them)."""
    lines = source.splitlines()
    out, skipping = [], False
    for line in lines:
        if skipping:
            # end of "from x import (a,\n b,\n)" - or of a "\" continuation
            if line.split("#", 1)[0].rstrip().endswith(("\\", ")", "]")):
                skipping = False
            continue
        if re.match(r"^(import |from )\S", line):
            stripped = line.split("#", 1)[0].rstrip()
            if stripped.endswith("\\") or stripped.count("(") > stripped.count(")"):
                skipping = True
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


def main() -> int:
    parts = [HEADER]
    bodies = {}
    for title, path in MODULES:
        body = strip_imports(path.read_text(encoding="utf-8"))
        bodies[title] = body
        banner = f"\n\n# {'=' * 68}\n# {title}\n# {'=' * 68}\n"
        parts.append(banner + "\n" + body)
    check_for_collisions(bodies, FOOTER)
    corpus = (ROOT / "orbit_gpt" / "corpora" / "orbit_assistant.txt").read_text(encoding="utf-8")
    assert '"""' not in corpus and "\\" not in corpus, "corpus must be raw-string safe"
    parts.append(FOOTER.replace("@@CHAT_CORPUS@@", corpus))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1024:.0f} KB, "
          f"{len(OUT.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
