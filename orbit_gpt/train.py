"""The training loop.

Everything a small GPT needs and nothing more: AdamW, cosine schedule with
warmup, gradient clipping, automatic mixed precision, train/val split,
checkpointing on improvement, resuming, and a readable progress line with an
ETA.  It runs on CUDA, Apple MPS and plain CPU with the same code path.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from orbit_gpt.config import GPTConfig, TrainConfig
from orbit_gpt.data import TokenDataset
from orbit_gpt.model import GPT
from orbit_gpt.tokenizer import BPETokenizer, CharTokenizer, Tokenizer


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
