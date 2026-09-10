"""Model / training configuration objects and the built-in size presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


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
