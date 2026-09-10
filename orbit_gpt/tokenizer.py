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

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
