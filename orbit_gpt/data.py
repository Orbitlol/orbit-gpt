"""Corpora and the (very simple) data pipeline.

A *corpus spec* is a comma separated list of sources, each optionally followed
by ``:repeat`` to oversample it::

    shakespeare                 # just Shakespeare
    orbit-chat                  # just the built-in assistant corpus
    shakespeare,orbit-chat:12   # Shakespeare once + the chat corpus 12 times
    ./my_notes.txt              # any local text file
    https://example.com/a.txt   # or any URL

Everything is cached on disk, so the second run is instant.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

DEFAULT_CACHE_DIR = Path(
    os.environ.get("ORBIT_GPT_CACHE", Path.home() / ".cache" / "orbit-gpt")
)

# Mirrors, tried in order, so a flaky CDN never ruins your evening.
DATASETS: Dict[str, Tuple[str, ...]] = {
    "shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "https://raw.githubusercontent.com/karpathy/nanoGPT/master/data/tinyshakespeare/input.txt",
    ),
    "shakespeare-sonnets": (
        "https://www.gutenberg.org/cache/epub/1041/pg1041.txt",
    ),
    "alice": (
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    ),
    "pride": (  # Pride and Prejudice - ~700KB of very learnable prose
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    ),
}

BUILTIN_DIR = Path(__file__).parent / "corpora"
BUILTIN_DATASETS: Dict[str, Path] = {
    "orbit-chat": BUILTIN_DIR / "orbit_assistant.txt",
}

# Generated on the fly (see orbit_gpt/corpora/conversation.py): a large, varied
# dialogue corpus.  This is the default because it is the only corpus big and
# varied enough that a small model has to generalise instead of memorise.
GENERATED_DATASETS = ("conversation", "prose")

USER_AGENT = "orbit-gpt/0.1 (+https://github.com/Orbitlol/orbit-gpt)"


def list_datasets() -> str:
    return ", ".join(sorted(list(DATASETS) + list(BUILTIN_DATASETS) + list(GENERATED_DATASETS)))


def generated_corpus(name: str) -> str:
    """Build one of the synthetic corpora (deterministic, no network)."""
    if name == "conversation":
        from orbit_gpt.corpora.conversation import build_conversation_corpus

        return build_conversation_corpus()
    if name == "prose":
        from orbit_gpt.corpora.prose import build_prose_corpus

        return build_prose_corpus()
    raise KeyError(f"unknown generated corpus {name!r}")


def _download(url: str, dest: Path, verbose: bool = True) -> Path:
    """Download ``url`` to ``dest`` (atomic: writes to a temp file first)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if verbose:
        print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if verbose and total:
                pct = 100 * done / total
                print(f"\r  {done/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.0f}%)", end="")
        if verbose and total:
            print()
    tmp.replace(dest)
    return dest


def _fetch_remote(name: str, cache_dir: Path, verbose: bool) -> str:
    urls = DATASETS[name]
    cache_file = cache_dir / f"{name}.txt"
    if cache_file.exists() and cache_file.stat().st_size > 0:
        if verbose:
            print(f"using cached {name} ({cache_file.stat().st_size/1e6:.2f} MB)")
        return cache_file.read_text(encoding="utf-8", errors="replace")
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            _download(url, cache_file, verbose=verbose)
            return cache_file.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # network hiccup, dead mirror, ...
            last_error = exc
            if verbose:
                print(f"  failed: {exc}")
    raise RuntimeError(
        f"could not download dataset {name!r} from any mirror ({last_error}). "
        "Pass a local file with --dataset path/to/text.txt instead."
    )


def load_source(source: str, cache_dir: Path = DEFAULT_CACHE_DIR, verbose: bool = True) -> str:
    """Load one source: a built-in name, a URL, or a path on disk."""
    if source in GENERATED_DATASETS:
        if verbose:
            print(f"  generating {source} corpus (no download needed)")
        return generated_corpus(source)
    if source in BUILTIN_DATASETS:
        path = BUILTIN_DATASETS[source]
        if not path.exists():
            raise FileNotFoundError(f"built-in corpus missing: {path}")
        return path.read_text(encoding="utf-8")
    if source.startswith(("http://", "https://")):
        key = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
        cache_file = cache_dir / f"url-{key}.txt"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            return cache_file.read_text(encoding="utf-8", errors="replace")
        _download(source, cache_file, verbose=verbose)
        return cache_file.read_text(encoding="utf-8", errors="replace")
    if source in DATASETS:
        return _fetch_remote(source, cache_dir, verbose)
    path = Path(source).expanduser()
    if path.is_dir():  # every text file in the folder, concatenated
        parts = []
        for f in sorted(path.glob("**/*.txt")):
            parts.append(f.read_text(encoding="utf-8", errors="replace"))
        if not parts:
            raise FileNotFoundError(f"no .txt files inside {path}")
        return "\n\n".join(parts)
    if not path.exists():
        raise FileNotFoundError(
            f"{source!r} is not a known dataset ({list_datasets()}) and not a file"
        )
    return path.read_text(encoding="utf-8", errors="replace")


def parse_spec(spec: str) -> List[Tuple[str, int]]:
    """``"shakespeare,orbit-chat:12"`` -> ``[("shakespeare", 1), ("orbit-chat", 12)]``"""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item and not item.startswith(("http://", "https://")):
            name, _, weight = item.rpartition(":")
            try:
                n = int(weight)
            except ValueError:  # a windows path like C:\... - treat as a name
                out.append((item, 1))
                continue
            out.append((name.strip(), max(1, n)))
        else:
            out.append((item, 1))
    return out or [("shakespeare", 1)]


def shuffle_blocks(text: str, seed: int = 0) -> str:
    """Shuffle the blank-line separated blocks of ``text`` (deterministic)."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    random.Random(seed).shuffle(blocks)
    return "\n\n".join(blocks)


def load_corpus(
    spec: str = "shakespeare",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = True,
    shuffle: bool = True,
) -> str:
    """Build the training text for a corpus spec (see the module docstring).

    Extra copies of a source (``name:3``) are shuffled block-wise unless
    ``shuffle`` is False: repeating a corpus in the same order teaches a small
    model the *document order*, and it then answers the exchange that follows
    the previous one instead of the question you actually asked.
    """
    cache_dir = Path(cache_dir)
    parts: List[str] = []
    for source, repeat in parse_spec(spec):
        text = load_source(source, cache_dir, verbose=verbose).strip()
        if verbose:
            print(
                f"  + {source}: {len(text):,} characters"
                + (f" (x{repeat}{', shuffled' if shuffle and repeat > 1 else ''})"
                   if repeat > 1 else "")
            )
        parts.append(text)
        for copy in range(repeat - 1):
            parts.append(shuffle_blocks(text, seed=copy) if shuffle else text)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# tokenized dataset
# ---------------------------------------------------------------------------
class TokenDataset:
    """Flat array of token ids + a train/val split + random block sampling."""

    def __init__(self, ids: List[int], val_fraction: float = 0.1):
        data = torch.tensor(ids, dtype=torch.long)
        cut = int(len(data) * (1.0 - val_fraction))
        self.train = data[:cut]
        self.val = data[cut:] if cut < len(data) else data[:1]
        if len(self.val) < 2:  # tiny corpus: fall back to using train as val
            self.val = self.train

    @property
    def n_train(self) -> int:
        return int(self.train.numel())

    @property
    def n_val(self) -> int:
        return int(self.val.numel())

    def get_batch(
        self, split: str, batch_size: int, block_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        if data.numel() <= block_size + 1:
            raise ValueError(
                f"the {split} split has only {data.numel()} tokens but block_size is "
                f"{block_size}. Use a larger corpus or a smaller --block-size."
            )
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x = x.pin_memory() if device.type == "cuda" else x
        y = y.pin_memory() if device.type == "cuda" else y
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def tokens(self) -> int:
        return self.n_train + self.n_val
