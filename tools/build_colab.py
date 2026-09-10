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
    corpus="shakespeare,orbit-chat:12",  # text to learn: builtin[:repeat], file, URL
    preset="auto",        # auto|nano|micro|mini|small|base  (auto: micro on GPU)
    vocab_size=1024,      # BPE vocabulary size
    block_size=None,      # context length (None = preset default)
    n_layer=None, n_head=None, n_embd=None,   # override the preset if you like
    batch_size=None,      # None = auto for your hardware
    max_steps=None,       # None = auto (2000 GPU / 1000 CPU)
    learning_rate=2e-3,
    dropout=0.1,
    seed=1337,
    out_dir="/content/orbit_model" if __import__("os").path.exists("/content") else "out/orbit",
    train=True,           # False = load an existing checkpoint and just chat
    sample_after_train=True,
    chat_after_train=True,
    chat_temperature=0.7,
    chat_tokens=160,
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

FOOTER = '''
# ---------------------------------------------------------------------------
# Built-in assistant corpus (packed in so this file works with zero downloads)
# ---------------------------------------------------------------------------
EMBEDDED_CHAT_CORPUS = r"""@@CHAT_CORPUS@@"""

_original_load_source = load_source


def load_source(source, cache_dir=DEFAULT_CACHE_DIR, verbose=True):  # noqa: F811
    if source == "orbit-chat":
        return EMBEDDED_CHAT_CORPUS
    return _original_load_source(source, cache_dir, verbose)


# ---------------------------------------------------------------------------
# Colab entry point
# ---------------------------------------------------------------------------
def pick_preset(preset: str, device: "torch.device") -> str:
    if preset != "auto":
        return preset
    return "micro" if device.type == "cuda" else "nano"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train and chat with a tiny GPT.")
    p.add_argument("--corpus", default=CONFIG["corpus"])
    p.add_argument("--preset", default=CONFIG["preset"])
    p.add_argument("--vocab-size", type=int, default=CONFIG["vocab_size"])
    p.add_argument("--block-size", type=int, default=CONFIG["block_size"])
    p.add_argument("--n-layer", type=int, default=CONFIG["n_layer"])
    p.add_argument("--n-head", type=int, default=CONFIG["n_head"])
    p.add_argument("--n-embd", type=int, default=CONFIG["n_embd"])
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--max-steps", type=int, default=CONFIG["max_steps"])
    p.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    p.add_argument("--dropout", type=float, default=CONFIG["dropout"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--out-dir", default=CONFIG["out_dir"])
    p.add_argument("--device", default="auto")
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--no-chat", action="store_true")
    p.add_argument("--no-sample", action="store_true")
    p.add_argument("--prompt", default="User: Hello!\\nAssistant:")
    args = p.parse_args(argv)

    print("=" * 72)
    print("  OrbitGPT - train a small language model on your own machine")
    print("=" * 72)

    device = auto_device(args.device)
    name = pick_preset(args.preset, device)
    defaults = GPU_DEFAULTS if device.type == "cuda" else CPU_DEFAULTS
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"preset: {name}")

    model_config = get_preset(name)
    for key, value in (
        ("n_layer", args.n_layer), ("n_head", args.n_head),
        ("n_embd", args.n_embd), ("block_size", args.block_size),
        ("dropout", args.dropout),
    ):
        if value is not None:
            setattr(model_config, key, value)
    batch_size = args.batch_size or defaults["batch_size"]
    max_steps = args.max_steps or defaults["max_steps"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = out_dir / "model.pt"
    if args.no_train or (not CONFIG["train"] and checkpoint.exists()):
        print(f"loading existing checkpoint: {checkpoint}")
        model, tokenizer = load_model(str(out_dir), device=str(device))
        print(f"loaded {model.n_params/1e6:.2f}M parameters")
    else:
        set_seed(args.seed)
        print(f"\\nloading corpus: {args.corpus}")
        text = load_corpus(args.corpus)
        print(f"corpus: {len(text):,} characters\\n")
        tokenizer = build_tokenizer(text, "bpe", args.vocab_size)
        ids = tokenizer.encode(text)
        dataset = TokenDataset(ids, val_fraction=0.1)
        print(f"tokenized: {len(ids):,} tokens")

        model_config.vocab_size = tokenizer.vocab_size
        model = GPT(model_config)
        train_cfg = TrainConfig(
            batch_size=batch_size,
            max_steps=max_steps,
            learning_rate=args.lr,
            warmup_steps=min(100, max(1, max_steps // 10)),
            seed=args.seed,
            out_dir=str(out_dir),
            device=str(device),
        )
        (out_dir / "train_config.json").write_text(json.dumps({
            "model_config": model_config.to_dict(),
            "train_config": train_cfg.to_dict(),
            "corpus": args.corpus,
        }, indent=2))
        trainer = Trainer(model, dataset, train_cfg, tokenizer, device=device)
        trainer.train()

        if CONFIG["sample_after_train"] and not args.no_sample:
            print("\\n" + "-" * 72)
            for prompt, temp in (
                (args.prompt, 0.7),
                ("ROMEO:", 0.9) if "ROMEO" in text[:100000] else (args.prompt, 1.0),
            ):
                print(f"\\n>>> {prompt}")
                generate(model, tokenizer, prompt, max_new_tokens=120,
                         temperature=temp, stop_strings=["\\nUser:"],
                         device=device, stream=True)
        try:  # make the checkpoint easy to download from Colab
            import shutil
            archive = shutil.make_archive(str(out_dir), "zip", out_dir)
            print(f"\\nzipped checkpoint: {archive}")
            from google.colab import files  # type: ignore
            files.download(archive)
        except Exception:
            pass

    if CONFIG["chat_after_train"] and not args.no_chat:
        chat(model, tokenizer, device=device,
             temperature=CONFIG["chat_temperature"],
             max_new_tokens=CONFIG["chat_tokens"])
    print(f"\\nModel saved in: {out_dir}")
    print("Reload it later with:  python orbit_gpt_colab.py --no-train --out-dir " + str(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


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
    corpus = (ROOT / "orbit_gpt" / "data" / "orbit_assistant.txt").read_text(encoding="utf-8")
    assert '"""' not in corpus and "\\" not in corpus, "corpus must be raw-string safe"
    parts.append(FOOTER.replace("@@CHAT_CORPUS@@", corpus))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1024:.0f} KB, "
          f"{len(OUT.read_text().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
