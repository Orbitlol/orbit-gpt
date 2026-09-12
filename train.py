#!/usr/bin/env python3
"""Train an OrbitGPT model from the command line.

    python train.py                                  # sensible defaults
    python train.py --preset micro --dataset shakespeare,orbit-chat:10
    python train.py --dataset ./my_notes.txt --preset mini --max-steps 4000
    python train.py --out-dir out/notes --resume out/notes/model.pt

The defaults adapt to your hardware: a laptop CPU gets the ``nano`` preset, a
GPU gets ``micro``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from orbit_gpt.checkpoints import (
    default_checkpoint_dir,
    human_size,
    resolve_resume,
)
from orbit_gpt.config import (
    DEFAULT_PRESET,
    PRESETS,
    TrainConfig,
    get_preset,
    preset_train_defaults,
)
from orbit_gpt.data import DEFAULT_CACHE_DIR, TokenDataset, load_corpus, list_datasets
from orbit_gpt.config import GPTConfig
from orbit_gpt.generate import generate
from orbit_gpt.model import GPT
from orbit_gpt.train import (
    Trainer,
    auto_device,
    build_tokenizer,
    save_tokens,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a small GPT (OrbitGPT) from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = p.add_argument_group("data")
    d.add_argument(
        "--dataset",
        default="conversation",
        help="corpus spec: name[:repeat],... a local file, a folder, or a URL "
        f"(known: {list_datasets()})",
    )
    d.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    d.add_argument("--tokenizer", choices=("bpe", "char"), default="bpe")
    d.add_argument("--vocab-size", type=int, default=None,
                   help="None = the preset default (1024 for nano, 2048 above)")
    d.add_argument("--val-fraction", type=float, default=0.1)
    d.add_argument("--no-shuffle", action="store_true",
                   help="keep repeated copies of a corpus in the same order")

    m = p.add_argument_group("model")
    m.add_argument(
        "--preset",
        default=None,
        choices=sorted(PRESETS),
        help=f"size preset (default: {DEFAULT_PRESET})",
    )
    m.add_argument("--n-layer", type=int, default=None)
    m.add_argument("--n-head", type=int, default=None)
    m.add_argument("--n-embd", type=int, default=None)
    m.add_argument("--block-size", type=int, default=None)
    m.add_argument("--dropout", type=float, default=None)

    t = p.add_argument_group("training")
    t.add_argument("--batch-size", type=int, default=None)
    t.add_argument("--grad-accum", type=int, default=None)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--max-epochs", type=float, default=None,
                   help="never train for more than this many passes over the "
                        "corpus (keeps a small corpus from being memorised)")
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--min-lr", type=float, default=None)
    t.add_argument("--warmup-steps", type=int, default=None)
    t.add_argument("--weight-decay", type=float, default=None)
    t.add_argument("--grad-clip", type=float, default=None)
    t.add_argument("--seed", type=int, default=1337)

    e = p.add_argument_group("logging / io")
    e.add_argument(
        "--out-dir",
        default=str(default_checkpoint_dir()),
        help="checkpoint directory (Google Drive is never used)",
    )
    e.add_argument("--eval-interval", type=int, default=250)
    e.add_argument("--eval-iters", type=int, default=20)
    e.add_argument("--log-interval", type=int, default=50)
    e.add_argument("--save-interval", type=int, default=250,
                   help="also write model-latest.pt every N steps (0 = only "
                        "on improvement and at the end)")
    e.add_argument("--init-from", default="",
                   help="fine-tune (SFT) from this checkpoint directory: keeps "
                        "its weights and tokenizer, starts a fresh schedule")
    e.add_argument("--no-cache", action="store_true",
                   help="do not reuse the cached tokenizer/tokens")
    e.add_argument("--resume", default="",
                   help="path to a checkpoint, or 'auto' for the newest one "
                        "in --out-dir")
    e.add_argument("--device", default="auto")
    e.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    e.add_argument("--compile", action="store_true", help="torch.compile (Linux/GPU)")
    e.add_argument("--no-sample", action="store_true", help="skip the sample at the end")
    e.add_argument("--prompt", default="User: Hello!\nAssistant:")
    e.add_argument("--sample-tokens", type=int, default=160)
    e.add_argument("--temperature", type=float, default=0.8)
    e.add_argument("--chat", action="store_true", help="start chatting when done")
    e.add_argument(
        "--push-checkpoint",
        action="store_true",
        help="commit+push the checkpoint to git if it is small enough "
             "(off by default; Colab has no git credentials)",
    )
    e.add_argument("--push-size-limit", type=float, default=50.0,
                   help="refuse to push checkpoints larger than this many MB")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    device = auto_device(args.device)
    preset_name = args.preset or DEFAULT_PRESET
    # hyper-parameters that suit this preset on this device; CLI flags win
    td = preset_train_defaults(preset_name, device.type)
    model_config = get_preset(preset_name)
    for key, value in (
        ("n_layer", args.n_layer),
        ("n_head", args.n_head),
        ("n_embd", args.n_embd),
        ("block_size", args.block_size),
        ("dropout", args.dropout),
    ):
        if value is not None:
            setattr(model_config, key, value)
    batch_size = args.batch_size or td["batch_size"]
    max_steps = args.max_steps or td.get("max_steps", 2000)
    max_epochs = td["max_epochs"] if args.max_epochs is None else args.max_epochs
    learning_rate = td["learning_rate"] if args.lr is None else args.lr
    min_lr = td["min_learning_rate"] if args.min_lr is None else args.min_lr
    warmup_steps = td["warmup_steps"] if args.warmup_steps is None else args.warmup_steps
    weight_decay = td["weight_decay"] if args.weight_decay is None else args.weight_decay
    grad_clip = td["grad_clip"] if args.grad_clip is None else args.grad_clip
    grad_accum = td["grad_accum_steps"] if args.grad_accum is None else args.grad_accum

    vocab_size = args.vocab_size or td["vocab_size"]
    print(f"OrbitGPT | preset={preset_name} device={device}")
    print(
        f"  lr={learning_rate:g} warmup={warmup_steps} batch={batch_size}"
        f"x{grad_accum} wd={weight_decay} clip={grad_clip} seed={args.seed}"
    )
    print(f"loading corpus: {args.dataset}")
    text = load_corpus(
        args.dataset, cache_dir=Path(args.cache_dir), shuffle=not args.no_shuffle
    )
    print(f"corpus: {len(text):,} characters\n")

    init_dir = Path(args.init_from) if args.init_from else None
    cache = None if args.no_cache else Path(args.cache_dir)
    if init_dir is not None:
        # stage 2 (SFT): reuse stage 1's tokenizer so the vocab matches
        from orbit_gpt.tokenizer import load_tokenizer

        tokenizer = load_tokenizer(str(init_dir))
        ids = tokenizer.encode(text)
        print(f"tokenizer from {init_dir} (vocab {tokenizer.vocab_size})")
        print(f"tokenized: {len(ids):,} tokens")
    else:
        tokenizer, cached_ids = build_tokenizer(
            text, args.tokenizer, vocab_size, cache_dir=cache
        )
        if cached_ids is None:
            cached_ids = tokenizer.encode(text)
            save_tokens(cached_ids, text, args.tokenizer, vocab_size, cache)
        ids = cached_ids
        print(f"tokenized: {len(ids):,} tokens")
    dataset = TokenDataset(ids, val_fraction=args.val_fraction)

    model_config.vocab_size = tokenizer.vocab_size
    if init_dir is not None:
        # A fine-tune has to reuse the pre-trained architecture, otherwise no
        # weight matches and we would just be training from scratch.  An
        # explicit --n-layer/--n-head/--n-embd still wins.
        ckpt = torch.load(init_dir / "model.pt", map_location="cpu")
        saved = GPTConfig.from_dict(
            ckpt.get("model_config") or ckpt.get("config") or {}
        )
        if saved.vocab_size == tokenizer.vocab_size and saved.n_layer:
            model_config = saved
            for key, value in (
                ("n_layer", args.n_layer), ("n_head", args.n_head),
                ("n_embd", args.n_embd), ("block_size", args.block_size),
                ("dropout", args.dropout),
            ):
                if value is not None:
                    setattr(model_config, key, value)
            model_config.vocab_size = tokenizer.vocab_size
            print(f"fine-tuning the architecture from {init_dir} "
                  f"({model_config.n_layer}L/{model_config.n_head}H/"
                  f"{model_config.n_embd}d)")

    if max_epochs:
        tokens_per_step = batch_size * model_config.block_size * grad_accum
        epoch_cap = max(1, math.ceil(dataset.n_train * max_epochs / tokens_per_step))
        if epoch_cap < max_steps:
            print(f"--max-epochs {max_epochs} -> capping {max_steps} steps to {epoch_cap}")
            max_steps = epoch_cap

    resume_path = resolve_resume(args.out_dir, args.resume)
    if args.resume and resume_path is None:
        print(f"! --resume {args.resume}: no such checkpoint, starting from scratch")
    if resume_path is not None:
        # the checkpoint knows the architecture; the preset only fills gaps
        saved = GPTConfig.from_dict(
            torch.load(resume_path, map_location="cpu").get("config", {})
        )
        if saved.vocab_size == tokenizer.vocab_size:
            model_config = saved
            print(f"resuming from {resume_path} -> using its architecture "
                  f"({saved.n_layer}L/{saved.n_head}H/{saved.n_embd}d)")
        else:
            print(
                f"! {resume_path} has vocab {saved.vocab_size} but the corpus "
                f"needs {tokenizer.vocab_size}; starting from scratch"
            )
            resume_path = None
    model = GPT(model_config)
    train_cfg = TrainConfig(
        batch_size=batch_size,
        grad_accum_steps=grad_accum,
        max_steps=max_steps,
        learning_rate=learning_rate,
        min_learning_rate=min_lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        val_fraction=args.val_fraction,
        device=str(device),
        dtype=args.dtype,
        compile=args.compile,
        seed=args.seed,
        out_dir=args.out_dir,
        resume=str(resume_path) if resume_path else "",
        init_from=str(init_dir / "model.pt") if init_dir is not None else "",
    )
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "train_config.json").write_text(
        json.dumps(
            {
                "model_config": model_config.to_dict(),
                "train_config": train_cfg.to_dict(),
                "dataset": args.dataset,
                "tokenizer": args.tokenizer,
            },
            indent=2,
        )
    )

    print(f"checkpoints: {Path(args.out_dir).resolve()}")
    trainer = Trainer(model, dataset, train_cfg, tokenizer, device=device)
    t0 = time.time()
    stats = trainer.train()
    print(f"total time: {time.time()-t0:.0f}s | {stats}")
    print(f"resume later with:  python train.py --out-dir {args.out_dir} "
          f"--resume auto --max-steps {max_steps + 1000}")

    if not args.no_sample:
        model.eval()
        print("\n--- sample ---")
        out = generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.sample_tokens,
            temperature=args.temperature,
            device=device,
        )
        print(args.prompt + out)

    if args.chat:
        from orbit_gpt.generate import chat

        chat(model, tokenizer, device=device)

    best = Path(args.out_dir) / "model.pt"
    print(f"\nbest checkpoint:  {best}  ({human_size(best) if best.exists() else '-'})")
    print(f"latest (resume):  {Path(args.out_dir)/'model-latest.pt'}")
    print(f"chat with it:     python generate.py --checkpoint {args.out_dir} --chat")
    if args.push_checkpoint:
        _push_checkpoint(args.out_dir, args.push_size_limit)
    return 0


def _push_checkpoint(out_dir: str, size_limit_mb: float) -> None:
    """Optionally commit the checkpoint.  Off by default, and size-limited."""
    import subprocess

    path = Path(out_dir) / "model.pt"
    if not path.exists():
        return
    size_mb = path.stat().st_size / 1e6
    if size_mb > size_limit_mb:
        print(f"! checkpoint is {size_mb:.1f} MB > {size_limit_mb:g} MB limit - "
              f"not committing it (keep it in {out_dir} and resume with "
              f"--resume auto)")
        return
    try:
        subprocess.run(["git", "add", "-f", str(path)], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"checkpoint: {path} ({human_size(path)})"],
            check=True,
        )
        subprocess.run(["git", "push"], check=True)
        print(f"pushed {path} ({human_size(path)})")
    except Exception as exc:
        print(f"! could not push the checkpoint ({exc}) - it is still on disk")


if __name__ == "__main__":
    sys.exit(main())
