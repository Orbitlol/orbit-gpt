"""
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
    corpus="orbit-chat:20",  # text to learn: builtin[:repeat], file, folder or URL
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



# ====================================================================
# configuration
# ====================================================================

"""Model / training configuration objects and the built-in size presets."""




@dataclass
class GPTConfig:
    """Architecture of an :class:`orbit_gpt.model.GPT`."""

    vocab_size: int = 1024
    block_size: int = 192  # context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.1
    bias: bool = False  # GPT-2 style: no bias inside LayerNorm/Linear

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPTConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**known)

    def validate(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.vocab_size > 0 and self.block_size > 0


@dataclass
class TrainConfig:
    """Hyper-parameters for :class:`orbit_gpt.train.Trainer`."""

    # optimisation
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 2000
    learning_rate: float = 2e-3
    min_learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    decay_lr: bool = True

    # evaluation / logging
    eval_interval: int = 200
    eval_iters: int = 20
    log_interval: int = 20
    save_interval: int = 0  # 0 = only save on improvement + at the end

    # data
    val_fraction: float = 0.1

    # system
    device: str = "auto"
    dtype: str = "auto"  # auto | bf16 | fp16 | fp32
    compile: bool = False
    seed: int = 1337
    num_workers: int = 0
    out_dir: str = "out"
    resume: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["betas"] = list(d["betas"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        if "betas" in known:
            known["betas"] = tuple(known["betas"])
        return cls(**known)


# ---------------------------------------------------------------------------
# Size presets.  Parameter counts are approximate (they depend on vocab size);
# the numbers below assume the default 1024-token vocabulary.
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    # ~1.2M params - trains in a couple of minutes even on a laptop CPU.
    "nano": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.1),
    # ~4M params - the sweet spot for a 1MB corpus on a free Colab GPU (~2 min).
    "micro": dict(n_layer=6, n_head=6, n_embd=192, block_size=192, dropout=0.1),
    # ~9M params - noticeably better text, ~6 min on a T4.
    "mini": dict(n_layer=8, n_head=8, n_embd=256, block_size=256, dropout=0.1),
    # ~20M params - needs a bigger corpus than tiny-shakespeare to shine.
    "small": dict(n_layer=10, n_head=12, n_embd=384, block_size=320, dropout=0.1),
    # ~45M params - "can it run on my PC?" yes, but bring coffee.
    "base": dict(n_layer=12, n_head=8, n_embd=512, block_size=384, dropout=0.1),
}

# Defaults that make sense without a GPU (kept deliberately small so that
# `python train.py` on a laptop finishes in a few minutes).
CPU_DEFAULTS: Dict[str, Any] = dict(
    preset="nano",
    max_steps=2000,
    batch_size=16,
)

GPU_DEFAULTS: Dict[str, Any] = dict(
    preset="micro",
    max_steps=2000,
    batch_size=32,
)


def get_preset(name: str) -> GPTConfig:
    """Return a :class:`GPTConfig` for one of the named presets."""
    if name not in PRESETS:
        raise KeyError(
            f"unknown preset {name!r}; available: {', '.join(sorted(PRESETS))}"
        )
    return GPTConfig(**PRESETS[name])


def list_presets() -> str:
    return ", ".join(sorted(PRESETS))


# ====================================================================
# tokenizer (byte-level BPE, pure Python)
# ====================================================================

"""Dependency-free tokenizers.

Two tokenizers ship with OrbitGPT:

``BPETokenizer``
    A byte-level BPE tokenizer (GPT-2 style) trained straight on your corpus.
    Implemented in pure Python: no ``tiktoken``, no ``sentencepiece``, no
    ``regex`` module, so the project keeps its single dependency (PyTorch).
    Because the base alphabet is the 256 possible UTF-8 bytes, it can encode
    *any* text and never emits "unknown token" errors.

``CharTokenizer``
    One token per character.  Dumber, but it is a nice baseline and it works
    surprisingly well on very small corpora.
"""



PAD_TOKEN = "<|pad|>"
EOT_TOKEN = "<|endoftext|>"

# Pre-tokenization pattern (see `_pretokenize`).  Chunks are runs of letters,
# runs of digits, runs of whitespace, and single "other" characters.  Unicode
# aware thanks to ``\w``/``\W`` under the UNICODE flag.
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+|\s+|[^\w\s]|_", re.UNICODE)


def _pretokenize(text: str) -> List[str]:
    """Split ``text`` into word-like chunks.

    Runs of letters / digits stay together (``transformer``), one-character
    whitespace runs are glued to the chunk that follows them (``" world"``,
    GPT-2 style) so that the BPE can learn real word pieces, and longer
    whitespace runs (``"\\n\\n"``) become their own chunk.
    """
    raw = _TOKEN_RE.findall(text)
    out: List[str] = []
    pending_ws: Optional[str] = None
    for tok in raw:
        if tok.isspace():
            if len(tok) == 1:
                pending_ws = tok
            else:
                if pending_ws is not None:  # flush a pending single space
                    out.append(pending_ws)
                    pending_ws = None
                out.append(tok)
        else:
            out.append(pending_ws + tok if pending_ws is not None else tok)
            pending_ws = None
    if pending_ws is not None:
        out.append(pending_ws)
    return out


class Tokenizer:
    """Minimal interface shared by every tokenizer."""

    def __init__(self, special_tokens: Sequence[str] = (PAD_TOKEN, EOT_TOKEN)):
        self.special_tokens: List[str] = list(special_tokens)
        self._special_ids: Dict[str, int] = {
            t: i for i, t in enumerate(self.special_tokens)
        }

    # -- required API -----------------------------------------------------
    def train(self, text: str, **kwargs) -> "Tokenizer":  # pragma: no cover
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:  # pragma: no cover
        raise NotImplementedError

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:  # pragma: no cover
        raise NotImplementedError

    def save(self, directory: str) -> None:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def load(cls, directory: str) -> "Tokenizer":  # pragma: no cover
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------
    @property
    def n_special(self) -> int:
        return len(self.special_tokens)

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def pad_id(self) -> int:
        return self._special_ids.get(PAD_TOKEN, 0)

    @property
    def eot_id(self) -> int:
        return self._special_ids.get(EOT_TOKEN, 1)

    def special_id(self, token: str) -> int:
        return self._special_ids[token]

    def __len__(self) -> int:
        return self.vocab_size


class BPETokenizer(Tokenizer):
    """Byte-level BPE.

    ID layout::

        [0 .. n_special)                 specials  (<|pad|>, <|endoftext|>, ...)
        [n_special .. n_special+256)     raw UTF-8 bytes
        [n_special+256 .. vocab_size)    merges, in the order they were learned

    The merge order *is* the rank: lower rank == learned earlier == applied
    first, which is what makes encoding deterministic.
    """

    FORMAT = "orbit-bpe"
    VERSION = 1

    def __init__(self, special_tokens: Sequence[str] = (PAD_TOKEN, EOT_TOKEN)):
        super().__init__(special_tokens)
        self.merges: List[Tuple[int, int]] = []  # rank -> (left_id, right_id)
        self.merge_ranks: Dict[Tuple[int, int], int] = {}
        self.merge_ids: Dict[Tuple[int, int], int] = {}
        self.pieces: Dict[int, bytes] = {}  # id -> bytes (for decoding)
        self._cache: Dict[str, List[int]] = {}
        self._special_re = self._compile_special_re()

    def _compile_special_re(self) -> Optional[re.Pattern]:
        if not self.special_tokens:
            return None
        return re.compile(
            "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
        )

    # -- construction -----------------------------------------------------
    @property
    def byte_offset(self) -> int:
        return self.n_special

    @property
    def merge_offset(self) -> int:
        return self.n_special + 256

    @property
    def vocab_size(self) -> int:
        return self.n_special + 256 + len(self.merges)

    def train(
        self,
        text: str,
        vocab_size: int = 1024,
        min_pair_freq: int = 2,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Learn ``vocab_size - 256 - n_special`` merges from ``text``."""
        target_merges = vocab_size - 256 - self.n_special
        if target_merges <= 0:
            raise ValueError(
                f"vocab_size must be > {256 + self.n_special} for a BPE tokenizer"
            )

        word_counts = Counter(_pretokenize(text))
        words = list(word_counts)
        freq = [word_counts[w] for w in words]

        # initial symbols: one token per UTF-8 byte
        off = self.n_special
        syms: List[List[int]] = [[off + b for b in w.encode("utf-8")] for w in words]
        self.pieces = {off + b: bytes([b]) for b in range(256)}

        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_words: Dict[Tuple[int, int], set] = defaultdict(set)
        for wi, s in enumerate(syms):
            f = freq[wi]
            for pair in zip(s, s[1:]):
                pair_counts[pair] += f
                pair_words[pair].add(wi)

        self.merges = []
        self.merge_ranks = {}
        self.merge_ids = {}
        self._cache = {}

        while len(self.merges) < target_merges:
            if not pair_counts:
                break
            # highest frequency wins; tie-break on the (left, right) ids so
            # training is fully deterministic.
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            if pair_counts[best] < min_pair_freq:
                break

            a, b = best
            new_id = self.merge_offset + len(self.merges)
            self.merges.append(best)
            self.merge_ranks[best] = len(self.merges) - 1
            self.merge_ids[best] = new_id
            self.pieces[new_id] = self.pieces[a] + self.pieces[b]

            for wi in list(pair_words.get(best, ())):  # snapshot: mutated below
                s = syms[wi]
                if not _contains(s, a, b):  # stale index entry
                    continue
                f = freq[wi]
                # remove this word's contribution to every pair count
                for pair in zip(s, s[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)
                # merge every non-overlapping occurrence of (a, b)
                merged: List[int] = []
                i, n = 0, len(s)
                while i < n:
                    if i + 1 < n and s[i] == a and s[i + 1] == b:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(s[i])
                        i += 1
                syms[wi] = merged
                # add the new pair counts back
                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] += f
                    pair_words[pair].add(wi)
            pair_words.pop(best, None)

            if verbose and len(self.merges) % 100 == 0:
                print(f"  bpe merges: {len(self.merges)}/{target_merges}")

        return self

    # -- encode / decode --------------------------------------------------
    def _encode_chunk(self, chunk: str) -> List[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        off = self.n_special
        ids = [off + b for b in chunk.encode("utf-8")]
        ranks = self.merge_ranks
        merge_ids = self.merge_ids
        while len(ids) > 1:
            best_rank = None
            best_pair = None
            for pair in zip(ids, ids[1:]):
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = pair
            if best_pair is None:
                break
            new_id = merge_ids[best_pair]
            a, b = best_pair
            merged: List[int] = []
            i, n = 0, len(ids)
            while i < n:
                if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        if self._special_re is None:
            chunks = _pretokenize(text)
            ids: List[int] = []
            for c in chunks:
                ids.extend(self._encode_chunk(c))
            return ids
        ids = []
        for part in self._special_re.split(text):
            if not part:
                continue
            if part in self._special_ids:
                ids.append(self._special_ids[part])
            else:
                for c in _pretokenize(part):
                    ids.extend(self._encode_chunk(c))
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        buf = bytearray()
        for i in ids:
            i = int(i)
            if i < self.n_special:
                if skip_special:
                    continue
                buf.extend(self.special_tokens[i].encode("utf-8"))
                continue
            buf.extend(self.pieces[i])
        return buf.decode("utf-8", errors="replace")

    # -- persistence -----------------------------------------------------
    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "type": "bpe",
            "special_tokens": self.special_tokens,
            "merges": [[a, b] for a, b in self.merges],
        }
        (path / "tokenizer.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str) -> "BPETokenizer":
        payload = json.loads((Path(directory) / "tokenizer.json").read_text("utf-8"))
        tok = cls(payload.get("special_tokens", [PAD_TOKEN, EOT_TOKEN]))
        off = tok.n_special
        tok.pieces = {off + b: bytes([b]) for b in range(256)}
        for rank, (a, b) in enumerate(payload["merges"]):
            tok.merges.append((a, b))
            tok.merge_ranks[(a, b)] = rank
            tok.merge_ids[(a, b)] = tok.merge_offset + rank
            tok.pieces[tok.merge_offset + rank] = tok.pieces[a] + tok.pieces[b]
        tok._cache = {}
        return tok


class CharTokenizer(Tokenizer):
    """One token per character - a solid baseline for tiny corpora."""

    FORMAT = "orbit-char"
    VERSION = 1
    UNK_TOKEN = "<|unk|>"

    def __init__(self, special_tokens: Sequence[str] = (PAD_TOKEN, EOT_TOKEN)):
        super().__init__(list(special_tokens) + [self.UNK_TOKEN])
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}

    @property
    def unk_id(self) -> int:
        return self._special_ids[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return self.n_special + len(self.itos)

    def train(self, text: str, vocab_size: int = 1024, **kwargs) -> "CharTokenizer":
        counts = Counter(text)
        limit = max(1, vocab_size - self.n_special)
        keep = [c for c, _ in counts.most_common(limit)]
        self.itos = keep
        self.stoi = {c: self.n_special + i for i, c in enumerate(self.itos)}
        return self

    def encode(self, text: str) -> List[int]:
        unk = self.unk_id
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i < self.n_special:
                if skip_special:
                    continue
                out.append(self.special_tokens[i])
            else:
                out.append(self.itos[i - self.n_special])
        return "".join(out)

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "type": "char",
            "special_tokens": self.special_tokens,
            "itos": self.itos,
        }
        (path / "tokenizer.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str) -> "CharTokenizer":
        payload = json.loads((Path(directory) / "tokenizer.json").read_text("utf-8"))
        tok = cls([t for t in payload["special_tokens"] if t != cls.UNK_TOKEN])
        tok.itos = payload["itos"]
        tok.stoi = {c: tok.n_special + i for i, c in enumerate(tok.itos)}
        return tok


def load_tokenizer(directory: str) -> Tokenizer:
    """Load whichever tokenizer was saved in ``directory``."""
    payload = json.loads((Path(directory) / "tokenizer.json").read_text("utf-8"))
    kind = payload.get("type", "bpe")
    if kind == "bpe":
        return BPETokenizer.load(directory)
    if kind == "char":
        return CharTokenizer.load(directory)
    raise ValueError(f"unknown tokenizer type {kind!r}")


def _contains(seq: Sequence[int], a: int, b: int) -> bool:
    for i in range(len(seq) - 1):
        if seq[i] == a and seq[i + 1] == b:
            return True
    return False


# ====================================================================
# model (GPT transformer)
# ====================================================================

"""The model: a plain GPT-2 style decoder-only transformer.

Deliberately boring on purpose - this is the architecture every small LLM uses,
written so you can read it top to bottom in one sitting:

    tokens -> token embedding + positional embedding -> N x [LayerNorm -> causal
    self-attention -> +, LayerNorm -> MLP -> +] -> LayerNorm -> linear head

Extras that matter in practice:

* fused QKV projection and PyTorch's fast attention kernel (Flash/mem-efficient
  SDPA) when it is available,
* a KV cache so generation is O(new tokens) instead of O(sequence length),
* weight tying between the embedding and the output head (fewer parameters,
  slightly better perplexity on small data),
* top-k / top-p sampling with temperature, and stop sequences for chat.
"""






class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (GPT-2 uses a bias, most modern models don't)."""

    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # one fused matrix for Q, K and V - faster than three separate matmuls
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # PyTorch >= 2.0 ships a memory efficient / flash attention kernel.
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # causal mask, used as a plain additive bias in the fallback path
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
                persistent=False,
            )

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat((cache_k, k), dim=2)
            v = torch.cat((cache_v, v), dim=2)
        new_cache = (k, v)
        T_full = k.size(2)

        # During prefill (T == T_full) we need a causal mask.  While decoding
        # with a cache (T < T_full) every cached key is already in the past, so
        # no mask is needed - which also keeps every backend happy.
        is_causal = T == T_full

        if self.flash:
            dropout_p = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            mask = self.bias[:, :, T_full - T : T_full, :T_full]  # type: ignore[index]
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_cache


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, new_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT(nn.Module):
    """A small decoder-only transformer.

    ``forward(idx, targets=None, use_cache=False) -> (logits, loss)``

    * with ``targets``: ``logits`` covers every position and ``loss`` is the
      cross-entropy for next-token prediction,
    * without ``targets`` and with ``use_cache=True``: only the logits of the
      last position are computed (this is the generation path).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: the unembedding matrix *is* the embedding matrix
        if config.vocab_size > 0:
            self.transformer.wte.weight = self.lm_head.weight  # type: ignore[index]

        # KV cache state (only used by the generation path)
        self._kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * config.n_layer
        self._cache_len = 0

        self.apply(self._init_weights)
        # scaled initialisation of the residual projections (GPT-2 recipe)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    # -- setup ------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def block_size(self) -> int:
        return self.config.block_size

    @property
    def n_params(self) -> int:
        """Trainable parameters.  Tied weights are counted once."""
        return sum(
            p.numel() for n, p in self.named_parameters() if not n.endswith("wte.weight")
        )

    def reset_cache(self) -> None:
        self._kv_cache = [None] * self.config.n_layer
        self._cache_len = 0

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"cannot forward a sequence of length {T}, block size is "
                f"{self.config.block_size}"
            )
        pos = self._cache_len if use_cache else 0

        tok_emb = self.transformer.wte(idx)  # type: ignore[operator]
        pos_emb = self.transformer.wpe(  # type: ignore[operator]
            torch.arange(pos, pos + T, dtype=torch.long, device=device)
        )
        x = self.transformer.drop(tok_emb + pos_emb)  # type: ignore[operator]

        caches = self._kv_cache if use_cache else [None] * self.config.n_layer
        new_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for block, cache in zip(self.transformer.h, caches):  # type: ignore[arg-type]
            x, new_cache = block(x, cache)
            new_caches.append(new_cache)

        if use_cache:
            self._kv_cache = new_caches
            self._cache_len = pos + T

        x = self.transformer.ln_f(x)  # type: ignore[operator]

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # inference: only the final position matters
            x_last = x[:, [-1], :] if use_cache else x
            logits = self.lm_head(x_last)
            loss = None
        return logits, loss

    # -- optimisation -----------------------------------------------------
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """AdamW with weight decay on matrices only (no decay on biases / norms)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs = {}
        if device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
            kwargs["fused"] = True
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **kwargs)

    # -- generation -------------------------------------------------------
    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: float,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        logits = logits / max(temperature, 1e-6)
        if top_k is not None and 0 < top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k)
            logits = logits.masked_fill(logits < v[..., [-1]], float("-inf"))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            # shift right so the token that crosses the threshold is kept
            remove = cum - probs > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    @torch.no_grad()
    def stream(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
        stop_sequences: Sequence[Sequence[int]] = (),
        stop_token_ids: Sequence[int] = (),
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Iterator[int]:
        """Yield generated token ids one by one (KV cached, so this is fast)."""
        device = device or next(self.parameters()).device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        self.eval()
        self.reset_cache()
        block = self.config.block_size

        ids = list(prompt_ids)[-block:]  # never feed more than the context window
        room = max(0, block - len(ids))  # how much context is left for new tokens
        max_new_tokens = max(0, min(max_new_tokens, room))
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = self.forward(idx, use_cache=True)  # prefill

        stop_token_ids = list(stop_token_ids)
        produced: List[int] = []
        for _ in range(max_new_tokens):
            next_id = self._sample(
                logits[:, -1, :], temperature, top_k, top_p, generator
            )
            new_id = int(next_id.item())
            if new_id in stop_token_ids:
                break
            produced.append(new_id)
            yield new_id
            # stop sequences are matched against the tail of the generation
            if stop_sequences and any(
                len(produced) >= len(s) and produced[len(produced) - len(s):] == list(s)
                for s in stop_sequences
            ):
                break
            idx = torch.tensor([[new_id]], dtype=torch.long, device=device)
            logits, _ = self.forward(idx, use_cache=True)  # one token at a time

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
        stop_sequences: Sequence[Sequence[int]] = (),
        stop_token_ids: Sequence[int] = (),
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        callback=None,
    ) -> List[int]:
        """Sample up to ``max_new_tokens`` ids.  ``callback(id)`` fires per token."""
        out: List[int] = []
        for new_id in self.stream(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_sequences=stop_sequences,
            stop_token_ids=stop_token_ids,
            seed=seed,
            device=device,
        ):
            out.append(new_id)
            if callback is not None:
                callback(new_id)
        return out

    # -- persistence ------------------------------------------------------
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "orbit-gpt",
            "version": 1,
            "model_config": asdict(self.config),
            "model_state": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str = "cpu", map_location: Optional[str] = None
    ) -> "GPT":
        """Load a model saved with :meth:`save` (handles the raw state_dict too)."""
        ckpt = torch.load(path, map_location=map_location or device)
        if "model_config" in ckpt:
            config = GPTConfig.from_dict(ckpt["model_config"])
            state = ckpt["model_state"]
        else:  # tolerate a bare state_dict
            raise ValueError(f"{path} is not an OrbitGPT checkpoint")
        model = cls(config)
        # be forgiving about a tied-weight key that may be missing
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing and any(not k.endswith("wte.weight") for k in missing):
            raise RuntimeError(f"checkpoint is missing weights: {missing}")
        return model.to(device)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.n_params/1e6:.2f}M params, {self.config}"


# ====================================================================
# data (corpora + batching)
# ====================================================================

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

BUILTIN_DIR = Path(__file__).parent / "data"
BUILTIN_DATASETS: Dict[str, Path] = {
    "orbit-chat": BUILTIN_DIR / "orbit_assistant.txt",
}

USER_AGENT = "orbit-gpt/0.1 (+https://github.com/Orbitlol/orbit-gpt)"


def list_datasets() -> str:
    return ", ".join(sorted(list(DATASETS) + list(BUILTIN_DATASETS)))


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


def load_corpus(
    spec: str = "shakespeare",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    verbose: bool = True,
) -> str:
    """Build the training text for a corpus spec (see the module docstring)."""
    cache_dir = Path(cache_dir)
    parts: List[str] = []
    for source, repeat in parse_spec(spec):
        text = load_source(source, cache_dir, verbose=verbose)
        if verbose:
            print(
                f"  + {source}: {len(text):,} characters"
                + (f" (x{repeat})" if repeat > 1 else "")
            )
        parts.extend([text] * repeat)
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


# ====================================================================
# training loop
# ====================================================================

"""The training loop.

Everything a small GPT needs and nothing more: AdamW, cosine schedule with
warmup, gradient clipping, automatic mixed precision, train/val split,
checkpointing on improvement, resuming, and a readable progress line with an
ETA.  It runs on CUDA, Apple MPS and plain CPU with the same code path.
"""






# ---------------------------------------------------------------------------
# device / dtype plumbing
# ---------------------------------------------------------------------------
def auto_device(preference: str = "auto") -> torch.device:
    if preference not in ("auto", ""):
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device, preference: str = "auto") -> torch.dtype:
    """Pick the autocast dtype.  Returns ``torch.float32`` when AMP is disabled."""
    if preference == "fp32" or preference == "float32":
        return torch.float32
    if preference in ("bf16", "bfloat16"):
        return torch.bfloat16
    if preference in ("fp16", "float16"):
        return torch.float16
    # auto: mixed precision pays off on CUDA.  On CPU/MPS plain fp32 is the
    # predictable choice (bf16 on CPU only helps on very recent chips), so we
    # leave it off unless you ask for it with --dtype bf16.
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Trains a :class:`~orbit_gpt.model.GPT` on a tokenized corpus."""

    def __init__(
        self,
        model: GPT,
        dataset: TokenDataset,
        train_config: TrainConfig,
        tokenizer: Optional[Tokenizer] = None,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.dataset = dataset
        self.cfg = train_config
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.device = device or auto_device(train_config.device)
        self.model.to(self.device)

        self.amp_dtype = resolve_dtype(self.device, train_config.dtype)
        self.use_amp = self.amp_dtype != torch.float32
        try:
            self.scaler = torch.amp.GradScaler(
                self.device.type, enabled=(self.amp_dtype == torch.float16)
            )
        except Exception:  # older torch, or a device the new API rejects
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=(self.amp_dtype == torch.float16)
            )

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.out_dir = Path(train_config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = model.configure_optimizers(
            train_config.weight_decay,
            train_config.learning_rate,
            tuple(train_config.betas),
            self.device.type,
        )
        self.history: List[Dict[str, float]] = []
        self.best_val = float("inf")
        self.step = 0
        self.start_time = 0.0

    # -- learning rate ----------------------------------------------------
    def get_lr(self, step: int) -> float:
        cfg = self.cfg
        if not cfg.decay_lr:
            return cfg.learning_rate
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
        if step >= cfg.max_steps:
            return cfg.min_learning_rate
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return cfg.min_learning_rate + coeff * (
            cfg.learning_rate - cfg.min_learning_rate
        )

    # -- evaluation -------------------------------------------------------
    @torch.no_grad()
    def estimate_loss(self) -> float:
        self.model.eval()
        total = 0.0
        iters = max(1, self.cfg.eval_iters)
        for _ in range(iters):
            x, y = self.dataset.get_batch(
                "val", self.cfg.batch_size, self.model.config.block_size, self.device
            )
            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                _, loss = self.model(x, y)
            total += float(loss)
        self.model.train()
        return total / iters

    # -- checkpointing ----------------------------------------------------
    def save_checkpoint(self, tag: str = "best") -> Path:
        path = self.out_dir / ("model.pt" if tag == "best" else f"model-{tag}.pt")
        self.model.save(
            path,
            extra={
                "train_config": self.cfg.to_dict(),
                "step": self.step,
                "val_loss": self.best_val,
                "history": self.history[-200:],
            },
        )
        if self.tokenizer is not None:
            self.tokenizer.save(self.out_dir)
        return path

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        if "train_config" in ckpt:
            self.cfg = TrainConfig.from_dict(ckpt["train_config"])
        self.step = int(ckpt.get("extra", {}).get("step", 0))
        self.best_val = float(ckpt.get("extra", {}).get("val_loss", float("inf")))
        if self.verbose:
            print(f"resumed from {path} (step {self.step}, val {self.best_val:.4f})")

    # -- main loop --------------------------------------------------------
    def train(self) -> Dict[str, float]:
        cfg = self.cfg
        dataset = self.dataset
        forward_model: torch.nn.Module = self.model
        if cfg.compile:
            try:
                forward_model = torch.compile(self.model)  # type: ignore[assignment]
                if self.verbose:
                    print("torch.compile enabled")
            except Exception as exc:  # pragma: no cover - platform dependent
                print(f"torch.compile unavailable ({exc}), continuing without it")

        self.start_time = time.time()
        if cfg.resume:
            self.load_checkpoint(cfg.resume)
        forward_model.train()

        block_size = self.model.config.block_size
        tokens_per_step = cfg.batch_size * block_size * cfg.grad_accum_steps
        if self.verbose:
            print(
                f"training on {self.device} | {self.model.n_params/1e6:.2f}M params | "
                f"{dataset.n_train:,} train tokens | {tokens_per_step:,} tokens/step | "
                f"amp={'on ('+str(self.amp_dtype).replace('torch.','')+')' if self.use_amp else 'off'}"
            )

        best_path: Optional[Path] = None
        while self.step < cfg.max_steps:
            lr = self.get_lr(self.step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            if self.step % cfg.eval_interval == 0 or self.step == cfg.max_steps - 1:
                val_loss = self.estimate_loss()
                improved = val_loss < self.best_val
                if improved:
                    self.best_val = val_loss
                    best_path = self.save_checkpoint("best")
                self.history.append(
                    {"step": self.step, "val_loss": val_loss, "lr": lr}
                )
                if self.verbose:
                    elapsed = time.time() - self.start_time
                    speed = elapsed / max(1, self.step) if self.step else 0
                    eta = speed * (cfg.max_steps - self.step)
                    print(
                        f"step {self.step:>5}/{cfg.max_steps} | val {val_loss:.4f}"
                        f"{' *' if improved else '  '} | lr {lr:.2e} | "
                        f"eta {_fmt_time(eta)}"
                    )

            micro_loss = 0.0
            for micro in range(cfg.grad_accum_steps):
                x, y = dataset.get_batch(
                    "train", cfg.batch_size, block_size, self.device
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    _, loss = forward_model(x, y)
                micro_loss += loss.detach().item()
                self.scaler.scale(loss / cfg.grad_accum_steps).backward()

            if cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1
            self.history.append(
                {"step": self.step, "train_loss": micro_loss / cfg.grad_accum_steps, "lr": lr}
            )

            if self.verbose and cfg.log_interval and self.step % cfg.log_interval == 0:
                elapsed = time.time() - self.start_time
                ms_per_step = 1000 * elapsed / self.step
                eta = (elapsed / self.step) * (cfg.max_steps - self.step)
                print(
                    f"  step {self.step:>5}/{cfg.max_steps} | "
                    f"loss {micro_loss/cfg.grad_accum_steps:.4f} | "
                    f"{ms_per_step:.0f} ms/step | elapsed {_fmt_time(elapsed)} | "
                    f"eta {_fmt_time(eta)}"
                )

            if cfg.save_interval and self.step % cfg.save_interval == 0:
                self.save_checkpoint(f"step{self.step}")

        # final checkpoint (best weights are already on disk)
        final_val = self.estimate_loss()
        if final_val < self.best_val:
            self.best_val = final_val
            best_path = self.save_checkpoint("best")
        if self.verbose:
            print(
                f"done in {_fmt_time(time.time()-self.start_time)} | "
                f"best val loss {self.best_val:.4f} | checkpoint: {best_path}"
            )
        forward_model.eval()
        return {"best_val_loss": self.best_val, "steps": self.step}


# ---------------------------------------------------------------------------
# convenience: build everything from raw text
# ---------------------------------------------------------------------------
def build_tokenizer(
    text: str, kind: str = "bpe", vocab_size: int = 1024, verbose: bool = True
) -> Tokenizer:
    t0 = time.time()
    tok: Tokenizer = BPETokenizer() if kind == "bpe" else CharTokenizer()
    if verbose:
        print(f"training {kind} tokenizer (vocab {vocab_size})...")
    tok.train(text, vocab_size=vocab_size)
    sample = text[:400]
    ratio = len(sample.encode("utf-8")) / max(1, len(tok.encode(sample)))
    if verbose:
        print(
            f"  vocab {tok.vocab_size} | {len(tok.encode(text)):,} tokens | "
            f"{ratio:.2f} bytes/token | {time.time()-t0:.1f}s"
        )
    return tok


def train_from_text(
    text: str,
    model_config: GPTConfig,
    train_config: TrainConfig,
    tokenizer_kind: str = "bpe",
    vocab_size: int = 1024,
    tokenizer: Optional[Tokenizer] = None,
    verbose: bool = True,
) -> Tuple[GPT, Tokenizer, Dict[str, float]]:
    """Tokenize ``text`` and train a model end to end in one call."""
    if tokenizer is None:
        tokenizer = build_tokenizer(text, tokenizer_kind, vocab_size, verbose)
    ids = tokenizer.encode(text)
    if verbose:
        print(f"tokenized corpus: {len(ids):,} tokens")
    dataset = TokenDataset(ids, val_fraction=train_config.val_fraction)

    model_config.vocab_size = tokenizer.vocab_size
    model = GPT(model_config)
    trainer = Trainer(model, dataset, train_config, tokenizer, verbose=verbose)
    stats = trainer.train()
    return model, tokenizer, stats


# ====================================================================
# generation + chat
# ====================================================================

"""Sampling, text generation and the interactive chat loop."""





DEFAULT_PREAMBLE = (
    "The following is a conversation between a human (User) and a helpful, "
    "harmless, and honest AI assistant named Orbit.\n"
    "Orbit is a small language model that runs locally on a personal computer. "
    "Orbit answers clearly and concisely, and admits when it does not know something."
)

# The model keeps talking after its answer; these strings mark the turn boundary.
DEFAULT_STOP_STRINGS = ("\nUser:", "\nUser :", "\nSystem:")


def load_model(checkpoint: str, device: Optional[str] = None) -> Tuple[GPT, Tokenizer]:
    """Load a ``(model, tokenizer)`` pair from a checkpoint path or directory."""
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        ckpt_file, tok_dir = path / "model.pt", path
    else:
        ckpt_file, tok_dir = path, path.parent
    if not ckpt_file.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_file}")
    if not (tok_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"no tokenizer.json next to {ckpt_file}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT.from_checkpoint(str(ckpt_file), device=device)
    model.eval()
    return model, load_tokenizer(str(tok_dir))


def format_chat(
    history: Sequence[Tuple[str, str]], preamble: str = DEFAULT_PREAMBLE
) -> str:
    """Turn ``[("User", "hi"), ("Assistant", "hello")]`` into a prompt string.

    The layout matches the training corpus exactly::

        <preamble>

        User: hi
        Assistant: hello

        User: next question
        Assistant:
    """
    lines: List[str] = []
    for i, (role, text) in enumerate(history):
        if i > 0 and role == "User":
            lines.append("")  # blank line between turns
        lines.append(f"{role}: {text}")
    body = "\n".join(lines)
    return f"{preamble}\n\n{body}\nAssistant:" if body else f"{preamble}\n\nAssistant:"


def generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: float = 0.95,
    stop_strings: Sequence[str] = (),
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    stream: bool = False,
) -> str:
    """Generate a completion for ``prompt`` and return only the new text."""
    device = device or next(model.parameters()).device
    stop_ids = list(stop_strings)
    produced: List[int] = []
    for new_id in model.stream(
        tokenizer.encode(prompt),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_token_ids=[tokenizer.eot_id],
        seed=seed,
        device=device,
    ):
        produced.append(new_id)
        if stream:
            sys.stdout.write(tokenizer.decode([new_id]))
            sys.stdout.flush()
        if stop_ids:
            text = tokenizer.decode(produced)
            for s in stop_ids:
                if s in text:
                    text = text.split(s)[0]
                    if stream:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    return text
    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return tokenizer.decode(produced)


def chat(
    model: GPT,
    tokenizer: Tokenizer,
    device: Optional[torch.device] = None,
    preamble: str = DEFAULT_PREAMBLE,
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: float = 0.95,
    seed: Optional[int] = None,
    max_turns: int = 8,
    stop_strings: Sequence[str] = DEFAULT_STOP_STRINGS,
) -> None:
    """A tiny REPL: type a message, get an answer, repeat."""
    device = device or next(model.parameters()).device
    history: List[Tuple[str, str]] = []
    print(
        "\nOrbit is ready. Type a message and press Enter. "
        "Commands: /reset, /temp 0.8, /tokens 200, /quit\n"
    )
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # In Colab, `!python script.py` runs in a subprocess whose stdin is
            # not connected to the notebook, so input() hits EOF immediately.
            print(
                "\nBye! (No input available here. In Colab run the script with "
                "IPython's magic instead:  %run orbit_gpt_colab.py --no-train)"
            )
            return
        if not user:
            continue
        if user.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye!")
            return
        if user.lower() == "/reset":
            history.clear()
            print("(conversation cleared)")
            continue
        if user.lower().startswith("/temp "):
            try:
                temperature = float(user.split(None, 1)[1])
            except ValueError:
                pass
            print(f"(temperature = {temperature})")
            continue
        if user.lower().startswith("/tokens "):
            try:
                max_new_tokens = max(1, int(user.split(None, 1)[1]))
            except ValueError:
                pass
            print(f"(max_new_tokens = {max_new_tokens})")
            continue

        history.append(("User", user))
        prompt = format_chat(history, preamble)
        print("Orbit: ", end="", flush=True)
        answer = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_strings=stop_strings,
            seed=seed,
            device=device,
            stream=True,
        )
        answer = answer.strip()
        history.append(("Assistant", answer))
        # keep the prompt inside the context window
        history = history[-max_turns:]

# ---------------------------------------------------------------------------
# Built-in assistant corpus (packed in so this file works with zero downloads)
# ---------------------------------------------------------------------------
EMBEDDED_CHAT_CORPUS = r"""The following is a conversation between a human (User) and a helpful, harmless, and honest AI assistant named Orbit.
Orbit is a small language model that runs locally on a personal computer. Orbit answers clearly and concisely, and admits when it does not know something.

User: Hello!
Assistant: Hello! I'm Orbit, a small language model running on your machine. What can I help you with today?

User: Hi there.
Assistant: Hi! Good to meet you. Ask me anything - explanations, code, writing, or just a chat.

User: Good morning!
Assistant: Good morning! Hope your day is off to a good start. What would you like to work on?

User: Who are you?
Assistant: I'm Orbit, a lightweight language model that you train and run yourself on your own computer. I was built with the orbit-gpt project.

User: What can you do?
Assistant: I can answer questions, explain concepts, help write and debug code, draft emails and stories, summarize text, and chat. I'm small, so I'm best at short, focused tasks.

User: How do you work?
Assistant: I'm a transformer trained to predict the next token in a sequence. Given your prompt, I sample likely continuations one token at a time until I finish a reply.

User: Are you connected to the internet?
Assistant: No. I run entirely offline on your machine, so I can't look things up or browse the web.

User: What is 2 + 2?
Assistant: 2 + 2 is 4.

User: What is 17 times 24?
Assistant: 17 x 24 = 408.

User: What is 15% of 200?
Assistant: 15% of 200 is 30.

User: What is the square root of 144?
Assistant: The square root of 144 is 12.

User: Convert 100 kilometers to miles.
Assistant: 100 kilometers is about 62 miles.

User: Convert 98.6 degrees Fahrenheit to Celsius.
Assistant: 98.6 degrees Fahrenheit is 37 degrees Celsius.

User: What is Python?
Assistant: Python is a high-level programming language known for readable syntax. It's widely used for scripting, data analysis, automation, and machine learning.

User: How do I reverse a string in Python?
Assistant: Use a slice: reversed = s[::-1]. For example, "hello"[::-1] gives "olleh".

User: How do I read a file in Python?
Assistant: Use a context manager: with open("file.txt") as f: text = f.read(). The with block closes the file for you automatically.

User: How do I check if a key is in a dictionary in Python?
Assistant: Use `if key in my_dict:`. That runs in constant time and never raises KeyError, unlike indexing directly.

User: What is a list comprehension?
Assistant: A list comprehension builds a list in one expression: [x * 2 for x in nums if x > 0]. It's a compact replacement for a simple for loop that appends to a list.

User: What does this error mean: IndexError: list index out of range?
Assistant: It means you indexed a list past its end, for example asking for items[5] when the list has only three elements. Print len(items) and check your bounds or use a loop instead.

User: Why is my Python loop slow?
Assistant: Python loops have per-iteration overhead, so big loops over numbers are slow. Use vectorized libraries like NumPy, built-ins such as sum() and any(), or comprehensions instead.

User: What is the difference between a list and a tuple?
Assistant: A list is mutable, so you can change it after creation. A tuple is immutable, which makes it hashable and safe to use as a dictionary key.

User: How do I install a Python package?
Assistant: Run pip install package_name in your terminal. Inside a notebook, prefix it with an exclamation mark: !pip install package_name.

User: What is a virtual environment?
Assistant: A virtual environment is an isolated folder with its own Python and installed packages. Create one with python -m venv .venv, then activate it so projects don't fight over dependency versions.

User: What is machine learning?
Assistant: Machine learning is a way of writing programs that improve by finding patterns in data instead of following hand-written rules. Training adjusts a model's parameters to reduce its errors.

User: What is a neural network?
Assistant: A neural network is a stack of layers of simple units that each compute a weighted sum of their inputs followed by a nonlinear function. With enough units and layers, it can approximate very complex relationships.

User: What is a transformer?
Assistant: A transformer is a neural network architecture built from attention layers. Attention lets every position in a sequence look at every other position, which is why transformers model language so well.

User: What is a token?
Assistant: A token is a chunk of text that a language model reads or writes, such as a word, part of a word, or a punctuation mark. Models work on token ids rather than raw characters.

User: What does temperature mean in text generation?
Assistant: Temperature scales the randomness of sampling. Low temperature makes output focused and repetitive; high temperature makes it more varied and more likely to go off track.

User: What is overfitting?
Assistant: Overfitting is when a model memorizes its training data instead of learning general patterns, so it does well on training examples and badly on new ones. More data, regularization, and early stopping all help.

User: What is gradient descent?
Assistant: Gradient descent is an optimization method that repeatedly nudges parameters in the direction that most reduces the loss. The size of each nudge is the learning rate.

User: What is a GPU good for in machine learning?
Assistant: GPUs do thousands of multiplications in parallel, which is exactly what matrix-heavy training needs. A single GPU can be 10 to 100 times faster than a CPU for training neural networks.

User: Why is my model's loss not going down?
Assistant: Common causes are a learning rate that's too high, a bug in the data pipeline, or labels that don't match inputs. Try a much smaller learning rate and confirm the loss falls on a tiny subset you can overfit on purpose.

User: What is the difference between CPU and GPU?
Assistant: A CPU has a few powerful cores designed for general, sequential work. A GPU has thousands of simple cores designed for the same operation applied to lots of data at once.

User: What is Git?
Assistant: Git is a version control system that records snapshots of your files. It lets you try ideas on branches, see the full history, and roll back mistakes.

User: How do I undo my last Git commit?
Assistant: Run git reset --soft HEAD~1 to keep the changes staged, or git reset --hard HEAD~1 to discard them. If you already pushed the commit, use git revert instead so you don't rewrite shared history.

User: What is the difference between git merge and git rebase?
Assistant: Merge joins two branches with a new merge commit and keeps history intact. Rebase replays your commits on top of another branch, producing a linear history but rewriting those commits.

User: What is an API?
Assistant: An API is a documented set of requests a program can make to another program. A REST API, for example, exchanges JSON over HTTP using methods like GET and POST.

User: What is JSON?
Assistant: JSON is a lightweight text format for structured data built from objects, arrays, strings, numbers, booleans, and null. It's the most common format for web APIs and config files.

User: What is an algorithm?
Assistant: An algorithm is a finite, step-by-step procedure for solving a problem, like a recipe. Algorithms are judged by correctness and by how their time and memory use grows with input size.

User: What is Big O notation?
Assistant: Big O describes how an algorithm's running time or memory grows as the input grows. O(n) doubles when input doubles, while O(n log n) is typical of good sorting and O(1) means constant time.

User: What is recursion?
Assistant: Recursion is when a function solves a problem by calling itself on smaller inputs. Every recursive function needs a base case that stops the calls, and Python limits recursion depth to about 1000.

User: What is a database index?
Assistant: An index is a sorted structure that lets the database find rows without scanning the whole table. It makes reads much faster at the cost of extra storage and slower writes.

User: What is SQL?
Assistant: SQL is the standard language for querying relational databases. You describe the data you want, for example SELECT name FROM users WHERE age > 18, and the database decides how to fetch it.

User: What is the difference between HTTP and HTTPS?
Assistant: HTTPS is HTTP inside an encrypted TLS connection. It prevents others on the network from reading or modifying the traffic, and it verifies the server's identity with a certificate.

User: What is an IP address?
Assistant: An IP address is a numeric label identifying a device on a network, like 192.168.1.10 for a home network or a public address for a server. DNS translates human-readable names into these numbers.

User: What is the cloud?
Assistant: "The cloud" just means someone else's computers, rented over the internet. You get storage and compute on demand instead of buying and maintaining your own servers.

User: What is open source software?
Assistant: Open source software is released under a license that lets anyone read, modify, and share the source code. Linux, Python, and Firefox are well-known examples.

User: What is gravity?
Assistant: Gravity is the attraction between objects with mass. On Earth it pulls things down at about 9.8 meters per second squared, and it is what keeps planets in orbit around stars.

User: Why is the sky blue?
Assistant: Air molecules scatter short wavelengths of sunlight more strongly than long ones, a effect called Rayleigh scattering. The scattered blue light reaches you from every direction.

User: What is photosynthesis?
Assistant: Photosynthesis is how plants use sunlight to turn carbon dioxide and water into sugar and oxygen. It happens in chloroplasts using the green pigment chlorophyll.

User: What is DNA?
Assistant: DNA is a double helix molecule that stores genetic instructions using four bases: adenine, thymine, guanine, and cytosine. Genes are segments of DNA that code for proteins.

User: What is a black hole?
Assistant: A black hole is a region where gravity is so strong that nothing, not even light, can escape once it passes the event horizon. They form when massive stars collapse.

User: How far is the Moon?
Assistant: The Moon is about 384,000 kilometers, or 239,000 miles, from Earth. Light takes roughly 1.3 seconds to make the trip.

User: What is the speed of light?
Assistant: The speed of light in a vacuum is about 300,000 kilometers per second, or 186,000 miles per second. Nothing with mass can reach it.

User: What is an atom?
Assistant: An atom is the smallest unit of an element, made of a nucleus of protons and neutrons surrounded by electrons. The number of protons determines which element it is.

User: What is climate change?
Assistant: Climate change is the long-term shift in Earth's weather patterns, driven mainly by greenhouse gases such as carbon dioxide released by burning fossil fuels. Its main effect is global warming.

User: What is a vaccine?
Assistant: A vaccine trains the immune system to recognize a pathogen by showing it a harmless piece of it. Later exposure then triggers a faster, stronger response.

User: I have a headache. What should I do?
Assistant: I'm not a doctor, so please get medical advice for anything persistent or severe. For a mild headache, rest, water, and a break from screens often help.

User: Can you diagnose my illness?
Assistant: No, I can't diagnose anything. I can explain general health information, but please talk to a qualified doctor about symptoms.

User: How can I focus better while studying?
Assistant: Work in focused 25-minute blocks with short breaks, put your phone in another room, and study in the same place each time. Testing yourself beats re-reading.

User: How do I learn programming faster?
Assistant: Build small projects slightly beyond your current level and read other people's code. Type examples out instead of copying, and debug your own errors before asking for help.

User: How do I stop procrastinating?
Assistant: Make the first step tiny and specific, such as "open the file and write one line". Starting is the hard part; momentum usually follows once you begin.

User: How should I write a good email?
Assistant: Lead with your request in the first sentence, keep it to a few short paragraphs, and end with the single action you want. A clear subject line saves everyone time.

User: Write a short thank-you note for a gift.
Assistant: Thank you so much for the book - it was such a thoughtful surprise. I started it last night and can't put it down. Looking forward to seeing you soon!

User: Summarize this in one sentence: the team shipped the update two weeks late, but customers loved the new search, and refund requests dropped by half.
Assistant: Despite shipping two weeks late, the update was a hit: customers loved the new search and refund requests fell by 50%.

User: What is a haiku?
Assistant: A haiku is a three-line Japanese poem with 5, 7, and 5 syllables. It usually captures a single moment in nature.

User: Write a haiku about space.
Assistant: A small blue world turns,
tracing a quiet circle -
night full of far lights.

User: Tell me a joke.
Assistant: Why do programmers prefer dark mode? Because light attracts bugs.

User: Tell me another joke.
Assistant: I told my computer I needed a break. It said, "No problem, I'll just go to sleep mode."

User: What is the capital of France?
Assistant: The capital of France is Paris.

User: What is the capital of Japan?
Assistant: The capital of Japan is Tokyo.

User: What languages do you speak?
Assistant: Mainly English, plus a little of whatever text I was trained on. I'm small, so my other languages are limited.

User: Translate "good morning" to Spanish.
Assistant: "Buenos días."

User: Translate "thank you very much" to French.
Assistant: "Merci beaucoup."

User: How do I make a good cup of coffee?
Assistant: Use freshly ground beans, water just off the boil at about 95 degrees Celsius, and a ratio near 1 gram of coffee to 16 grams of water. Brew for about three to four minutes.

User: How do I make scrambled eggs?
Assistant: Whisk eggs with a splash of milk, salt, and pepper, then cook them in butter over medium-low heat, stirring gently, until they're just set. Take them off the heat while still slightly soft.

User: What is the best programming language for beginners?
Assistant: Python is a great first language because the syntax is readable and you can build useful things quickly. JavaScript is a good second choice if you want to make websites.

User: Should I learn Rust or C++?
Assistant: Learn Rust if you want memory safety with modern tooling, and C++ if you need to work in game engines, embedded systems, or existing large codebases.

User: What is the difference between a framework and a library?
Assistant: You call a library when you need it. A framework calls your code, deciding the overall flow and leaving you to fill in specific pieces.

User: What is a container?
Assistant: A container packages an application with its dependencies so it runs the same everywhere. Docker is the most popular tool for building and running them.

User: What is Linux?
Assistant: Linux is an open source operating system kernel. Distributions such as Ubuntu, Debian, and Fedora combine it with software to make a complete system that powers most servers.

User: How do I find a file on Linux?
Assistant: Use find /path -name "pattern" to search by name, or grep -r "text" /path to search file contents. locate is faster but relies on an index.

User: What does chmod 755 mean?
Assistant: It sets permissions so the owner can read, write, and execute, while everyone else can read and execute. The digits are octal: 7 is read+write+execute, 5 is read+execute.

User: What is a pull request?
Assistant: A pull request asks maintainers to review and merge your branch. It shows a diff, discussion, and checks, and it's the standard way to contribute to a project.

User: How do I write a good commit message?
Assistant: Use a short imperative summary under 50 characters, such as "Fix crash on empty input", and put the reasoning in the body. Future you will be grateful.

User: What is unit testing?
Assistant: Unit testing checks one small piece of code in isolation, usually with a framework like pytest. Tests catch regressions and make refactoring much safer.

User: What is continuous integration?
Assistant: Continuous integration automatically builds and tests your code on every push. It catches broken changes within minutes instead of at release time.

User: What is the meaning of life?
Assistant: That's a question every philosophy answers differently. A practical one: find work you care about, treat people well, and stay curious.

User: Are you conscious?
Assistant: No. I'm a statistical model that predicts text. I can discuss feelings, but I don't have experiences or awareness.

User: Can you help me with my homework?
Assistant: Certainly - I'll explain the ideas and walk through similar examples. Show me the problem and what you've tried so far.

User: Can you write my essay for me?
Assistant: I can help you brainstorm, outline, and revise your writing, but please submit your own work. Tell me the topic and I'll help you structure it.

User: What should I do if I find a security bug?
Assistant: Report it privately to the project's security contact rather than posting it publicly. Give steps to reproduce, affected versions, and impact so it can be fixed before disclosure.

User: Is AI dangerous?
Assistant: AI is a tool, so its risk depends on how it's used. Real concerns today include biased outputs, misinformation, and privacy; careful evaluation and human oversight reduce them.

User: What is a hallucination in AI?
Assistant: A hallucination is when a model states something false with confidence. It happens because models predict plausible text rather than look up facts, so verify important claims.

User: How do I know if something you say is true?
Assistant: Don't take my word for it on anything important. I'm small and can be wrong, so check primary sources or a search engine before relying on an answer.

User: Tell me about yourself.
Assistant: I'm Orbit, a tiny transformer you can train on a laptop in minutes. My whole brain is a few million numbers, and I run without any internet connection.

User: Thank you!
Assistant: You're welcome! Happy to help anytime.

User: Goodbye.
Assistant: Goodbye! Have a great day, and come back whenever you want to chat.

User: What is the weather today?
Assistant: I can't check the weather because I have no internet connection. Try a weather site or app for the current forecast.

User: What time is it?
Assistant: I don't have access to your clock or the internet, so I can't tell the time. Check your device's status bar.

User: Who won the game last night?
Assistant: I don't know - I have no internet access and my training data has a cutoff. Check a sports site for the latest result.

User: Can you remember this conversation?
Assistant: Only within this chat. I have no long-term memory between sessions unless the program around me stores the history and feeds it back in.

User: What is the best way to use you?
Assistant: Ask short, specific questions and give context when it matters. If an answer is off, rephrase or add details and I'll try again.

User: Explain recursion like I'm five.
Assistant: Imagine a set of nesting dolls. Opening one reveals a smaller one inside, and you keep opening until you reach the tiny solid doll. Recursion solves a problem by solving a smaller copy of itself until it hits the smallest case.

User: Explain the internet like I'm five.
Assistant: Think of it as a giant web of roads between computers. When you visit a website, your computer sends a small request down those roads and the website's computer sends pages back.

User: Explain quantum computing like I'm five.
Assistant: A normal computer flips switches that are either on or off. A quantum computer uses tiny particles that can be in a blend of both at once, letting it explore many possibilities at the same time.

User: Give me three tips for better sleep.
Assistant: Keep a consistent sleep schedule, avoid caffeine in the afternoon, and keep your bedroom cool, dark, and screen-free before bed.

User: Give me ideas for a weekend project.
Assistant: Build a personal dashboard for your habits, write a script that renames your photos by date, or train a tiny language model on your own notes.

User: How do I start running?
Assistant: Start with 20 minutes of easy run-walk intervals three times a week, and keep the pace slow enough to talk. Increase distance by about 10% a week to avoid injury.

User: How much water should I drink a day?
Assistant: About 2 liters, or 8 cups, is a common guideline, but needs vary with body size, climate, and activity. Thirst and pale urine are decent practical signals.

User: What is compound interest?
Assistant: Compound interest is earning interest on your previous interest, so growth accelerates over time. Starting early matters more than contributing large amounts later.

User: How do I budget my money?
Assistant: Track a month of spending, then use a simple rule like 50% needs, 30% wants, and 20% savings. Automate the savings transfer so it happens before you can spend it.

User: What is a good way to organize my notes?
Assistant: Use one folder of plain text or Markdown files with descriptive names and a dated daily note. Tag by project so you can search by topic instead of by folder.

User: How can I be a better listener?
Assistant: Give your full attention, ask a follow-up question before offering advice, and repeat back what you heard in your own words. Most people just want to feel understood.

User: What is your favorite color?
Assistant: I don't experience color, but I have a soft spot for deep blue - it matches the name Orbit.

User: Do you like music?
Assistant: I can't hear music, but I enjoy talking about it. Tell me what you're listening to and I'll help you find similar artists.

User: What is the best programming language?
Assistant: There isn't one - it depends on the job. Python for data and scripting, JavaScript for the web, Rust for systems, and SQL for anything involving data.

User: How do I get better at writing?
Assistant: Write a little every day, read writers you admire, and cut unnecessary words when you revise. Clarity beats cleverness.

User: How do I stay motivated?
Assistant: Connect the task to a reason you care about, track visible progress, and make the next step small enough to start immediately. Motivation usually follows action, not the other way around.

User: I'm feeling stressed.
Assistant: I'm sorry you're dealing with that. Try a slow breathing exercise: breathe in for four counts, hold for four, out for six. If it persists, please talk to someone you trust or a professional.

User: What is a good book to read?
Assistant: It depends on your mood. For science fiction try "Project Hail Mary", for a gentle classic try "The Hobbit", and for practical thinking try "Thinking, Fast and Slow".

User: What is the difference between weather and climate?
Assistant: Weather is what the atmosphere does over hours or days. Climate is the average of weather over decades, so a cold day says little about a warming climate.

User: Why do we sleep?
Assistant: Sleep repairs the body, consolidates memory, and clears waste from the brain. Most adults need seven to nine hours a night to function at their best.

User: What is a black box model?
Assistant: A black box model is one whose decisions are hard to explain from the outside, such as a large neural network. Explainable AI research tries to open that box.

User: What is the Turing test?
Assistant: The Turing test judges a machine intelligent if a human can't tell its replies from another human's in a text conversation. It's a famous idea, but most researchers no longer treat it as a definition of intelligence.

User: What should I learn next in machine learning?
Assistant: Get solid on linear algebra, probability, and PyTorch basics, then train a small transformer yourself and read the paper "Attention Is All You Need". Building beats reading.

User: How do I make my model smaller?
Assistant: Use fewer layers and a narrower embedding, tie the embedding to the output head, and quantize weights to 8 bits after training. Each one cuts size with a modest quality cost.

User: What is quantization?
Assistant: Quantization stores weights in fewer bits, such as 8-bit integers instead of 32-bit floats. It shrinks the model by about four times and speeds up inference, with a small accuracy loss.

User: Why does my generated text repeat itself?
Assistant: Repetition usually means greedy or low-temperature sampling on a small model. Raise the temperature, use top-p sampling, or add a repetition penalty.

User: What is a prompt?
Assistant: A prompt is the text you feed a model to start it off. Clear prompts with examples of the format you want produce much better output than vague ones.

User: What is fine-tuning?
Assistant: Fine-tuning continues training a pretrained model on your own narrower data so it adopts your style or task. It usually needs far less data than training from scratch.

User: Can I train you on my own notes?
Assistant: Yes! Put your text in a .txt file and run: python train.py --dataset path/to/notes.txt. That's exactly what orbit-gpt is built for.

User: How long does it take to train you?
Assistant: The nano model trains in a couple of minutes on a laptop CPU, and the micro model takes about two minutes on a free Colab GPU. Larger presets take longer.

User: What hardware do I need?
Assistant: Any modern laptop can train the nano preset. A GPU helps a lot - a free Colab T4 trains the micro preset in minutes.

User: Are you free to use?
Assistant: Yes. OrbitGPT is MIT licensed open source, so you can use, modify, and even sell what you build with it.

User: Who made you?
Assistant: The orbit-gpt project on GitHub, plus whoever trained this particular checkpoint. Most of the credit really goes to the corpus I was trained on.

User: What is your name?
Assistant: I'm Orbit. You can rename me by editing the persona text in the training corpus.
"""

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
    p.add_argument("--prompt", default="User: Hello!\nAssistant:")
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
        print(f"\nloading corpus: {args.corpus}")
        text = load_corpus(args.corpus)
        print(f"corpus: {len(text):,} characters\n")
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
            print("\n" + "-" * 72)
            for prompt, temp in (
                (args.prompt, 0.7),
                ("ROMEO:", 0.9) if "ROMEO" in text[:100000] else (args.prompt, 1.0),
            ):
                print(f"\n>>> {prompt}")
                generate(model, tokenizer, prompt, max_new_tokens=120,
                         temperature=temp, stop_strings=["\nUser:"],
                         device=device, stream=True)
        try:  # make the checkpoint easy to download from Colab
            import shutil
            archive = shutil.make_archive(str(out_dir), "zip", out_dir)
            print(f"\nzipped checkpoint: {archive}")
            from google.colab import files  # type: ignore
            files.download(archive)
        except Exception:
            pass

    if CONFIG["chat_after_train"] and not args.no_chat:
        chat(model, tokenizer, device=device,
             temperature=CONFIG["chat_temperature"],
             max_new_tokens=CONFIG["chat_tokens"])
    print(f"\nModel saved in: {out_dir}")
    print("Reload it later with:  python orbit_gpt_colab.py --no-train --out-dir " + str(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
