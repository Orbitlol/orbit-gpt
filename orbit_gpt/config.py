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
# Size presets.  Parameter counts are measured with the default 2048-token
# vocabulary; they move a little if you change --vocab-size.
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    # 0.8M params - a fast smoke test.  Fine for "does the pipeline work",
    # too small to write good prose.
    "nano": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.1),
    # 4.8M params - THE DEFAULT.  ~6x the old nano model, and the 384-token
    # context leaves room for a couple of web-search results.  A few minutes
    # on a free Colab T4.
    "micro": dict(n_layer=6, n_head=8, n_embd=256, block_size=384, dropout=0.1),
    # 6.4M params - noticeably better text, ~10 min on a T4.
    "mini": dict(n_layer=8, n_head=8, n_embd=256, block_size=256, dropout=0.1),
    # ~20M params - needs a bigger corpus than tiny-shakespeare to shine.
    "small": dict(n_layer=10, n_head=12, n_embd=384, block_size=320, dropout=0.1),
    # ~45M params - "can it run on my PC?" yes, but bring coffee.
    "base": dict(n_layer=12, n_head=8, n_embd=512, block_size=384, dropout=0.1),
}

#: The preset used when you do not pass ``--preset``.
#:
#: ``nano`` (0.8M) is only kept for quick smoke tests - it is too small to
#: produce coherent replies, which is why the default is now ``micro``.
DEFAULT_PRESET = "micro"

# ---------------------------------------------------------------------------
# Training hyper-parameters that go with each preset.  Bigger models want a
# smaller learning rate, more warmup and a smaller batch (with gradient
# accumulation to keep the *effective* batch healthy).  These are the single
# place to tune the run: --lr/--batch-size/... on the CLI override them.
# ---------------------------------------------------------------------------
PRESET_TRAIN: Dict[str, Dict[str, Any]] = {
    "nano": dict(batch_size=32, grad_accum_steps=1, learning_rate=2e-3,
                 min_learning_rate=2e-4, warmup_steps=100, weight_decay=0.1,
                 grad_clip=1.0, max_epochs=8),
    "micro": dict(batch_size=24, grad_accum_steps=1, learning_rate=2e-3,
                  min_learning_rate=2e-4, warmup_steps=200, weight_decay=0.1,
                  grad_clip=1.0, max_epochs=8),
    "mini": dict(batch_size=16, grad_accum_steps=2, learning_rate=1.5e-3,
                 min_learning_rate=1.5e-4, warmup_steps=200, weight_decay=0.1,
                 grad_clip=1.0, max_epochs=8),
    "small": dict(batch_size=8, grad_accum_steps=4, learning_rate=1e-3,
                  min_learning_rate=1e-4, warmup_steps=300, weight_decay=0.1,
                  grad_clip=1.0, max_epochs=8),
    "base": dict(batch_size=4, grad_accum_steps=8, learning_rate=1e-3,
                 min_learning_rate=1e-4, warmup_steps=500, weight_decay=0.1,
                 grad_clip=1.0, max_epochs=8),
}

# Device-specific overrides, applied on top of PRESET_TRAIN.
CPU_DEFAULTS: Dict[str, Any] = dict(
    preset=DEFAULT_PRESET,
    max_steps=2000,
    batch_size=8,      # a laptop core cannot chew 24x384 tokens per step
)

GPU_DEFAULTS: Dict[str, Any] = dict(
    preset=DEFAULT_PRESET,
    max_steps=2000,
    batch_size=None,   # None = whatever PRESET_TRAIN says
)


def preset_train_defaults(name: str, device_type: str = "cpu") -> Dict[str, Any]:
    """Hyper-parameters for ``name`` on ``device_type`` ("cuda" or "cpu")."""
    settings: Dict[str, Any] = dict(PRESET_TRAIN.get(name, PRESET_TRAIN[DEFAULT_PRESET]))
    overrides = GPU_DEFAULTS if device_type == "cuda" else CPU_DEFAULTS
    for key, value in overrides.items():
        if key == "preset":
            continue
        if value is not None:
            settings[key] = value
    return settings


def get_preset(name: str) -> GPTConfig:
    """Return a :class:`GPTConfig` for one of the named presets."""
    if name not in PRESETS:
        raise KeyError(
            f"unknown preset {name!r}; available: {', '.join(sorted(PRESETS))}"
        )
    return GPTConfig(**PRESETS[name])


def list_presets() -> str:
    return ", ".join(sorted(PRESETS))
