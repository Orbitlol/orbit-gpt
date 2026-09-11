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
    ("generated conversation corpus", ROOT / "orbit_gpt" / "corpora" / "conversation.py"),
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
    corpus="conversation",  # generated dialogue corpus | builtin | file | folder | URL
    preset="auto",        # auto|nano|micro|mini|small|base  (auto: micro on GPU)
    vocab_size=2048,      # BPE vocabulary size
    block_size=None,      # context length (None = preset default)
    n_layer=None, n_head=None, n_embd=None,   # override the preset if you like
    batch_size=None,      # None = auto for your hardware
    max_steps=None,       # None = auto (2000 on both GPU and CPU)
    max_epochs=8,         # never train more than this many passes over the corpus
    learning_rate=2e-3,
    dropout=0.1,
    seed=1337,
    # ---- train once, reuse forever -------------------------------------
    # On Colab the model is saved to Google Drive, so it survives session
    # restarts: the next run finds it, loads it, and starts chatting in
    # seconds instead of training again.
    out_dir="/content/orbit_model" if __import__("os").path.exists("/content") else "out/orbit",
    save_to_drive=True,   # Colab only: keep the checkpoint in MyDrive/orbit-gpt
    retrain=False,        # True = ignore the saved model and train again
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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    sys.exit("PyTorch is not installed. In Colab run: !pip install torch")
'''

FOOTER = (Path(__file__).parent / "colab_footer.py").read_text(encoding="utf-8")


def strip_imports(source: str) -> str:
    """Drop top-level import statements (the merged header provides them)."""
    lines = source.splitlines()
    out, skipping = [], False
    for line in lines:
        if skipping:
            skipping = line.rstrip().endswith("\\") or not line.strip()
            if not line.strip():
                skipping = False
            continue
        if re.match(r"^(import |from )\S", line):
            # multi-line "from x import (\n a,\n b,\n)" -> skip until ")"
            if line.rstrip().endswith("("):
                skipping = True
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


def main() -> int:
    parts = [HEADER]
    for title, path in MODULES:
        body = strip_imports(path.read_text(encoding="utf-8"))
        banner = f"\n\n# {'=' * 68}\n# {title}\n# {'=' * 68}\n"
        parts.append(banner + "\n" + body)
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
