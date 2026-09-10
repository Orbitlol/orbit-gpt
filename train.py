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

from orbit_gpt.config import (
    CPU_DEFAULTS,
    GPU_DEFAULTS,
    PRESETS,
    TrainConfig,
    get_preset,
)
from orbit_gpt.data import DEFAULT_CACHE_DIR, TokenDataset, load_corpus, list_datasets
from orbit_gpt.generate import generate
from orbit_gpt.model import GPT
from orbit_gpt.train import Trainer, auto_device, build_tokenizer, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a small GPT (OrbitGPT) from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = p.add_argument_group("data")
    d.add_argument(
        "--dataset",
        default="shakespeare,orbit-chat:10",
        help="corpus spec: name[:repeat],... a local file, a folder, or a URL "
        f"(known: {list_datasets()})",
    )
    d.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    d.add_argument("--tokenizer", choices=("bpe", "char"), default="bpe")
    d.add_argument("--vocab-size", type=int, default=1024)
    d.add_argument("--val-fraction", type=float, default=0.1)

    m = p.add_argument_group("model")
    m.add_argument(
        "--preset",
        default=None,
        choices=sorted(PRESETS),
        help=f"size preset (default: {GPU_DEFAULTS['preset']} on GPU, "
        f"{CPU_DEFAULTS['preset']} on CPU)",
    )
    m.add_argument("--n-layer", type=int, default=None)
    m.add_argument("--n-head", type=int, default=None)
    m.add_argument("--n-embd", type=int, default=None)
    m.add_argument("--block-size", type=int, default=None)
    m.add_argument("--dropout", type=float, default=None)

    t = p.add_argument_group("training")
    t.add_argument("--batch-size", type=int, default=None)
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--epochs", type=float, default=None,
                   help="if set, --max-steps is derived from the corpus size")
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--min-lr", type=float, default=2e-4)
    t.add_argument("--warmup-steps", type=int, default=100)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=1337)

    e = p.add_argument_group("logging / io")
    e.add_argument("--out-dir", default="out/orbit")
    e.add_argument("--eval-interval", type=int, default=250)
    e.add_argument("--eval-iters", type=int, default=20)
    e.add_argument("--log-interval", type=int, default=50)
    e.add_argument("--save-interval", type=int, default=0)
    e.add_argument("--resume", default="")
    e.add_argument("--device", default="auto")
    e.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    e.add_argument("--compile", action="store_true", help="torch.compile (Linux/GPU)")
    e.add_argument("--no-sample", action="store_true", help="skip the sample at the end")
    e.add_argument("--prompt", default="User: Hello!\nAssistant:")
    e.add_argument("--sample-tokens", type=int, default=160)
    e.add_argument("--temperature", type=float, default=0.8)
    e.add_argument("--chat", action="store_true", help="start chatting when done")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(args.seed)

    device = auto_device(args.device)
    defaults = GPU_DEFAULTS if device.type == "cuda" else CPU_DEFAULTS
    preset_name = args.preset or defaults["preset"]
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
    batch_size = args.batch_size or defaults["batch_size"]
    max_steps = args.max_steps or defaults["max_steps"]

    print(f"OrbitGPT | preset={preset_name} device={device}")
    print(f"loading corpus: {args.dataset}")
    text = load_corpus(args.dataset, cache_dir=Path(args.cache_dir))
    print(f"corpus: {len(text):,} characters\n")

    tokenizer = build_tokenizer(text, args.tokenizer, args.vocab_size)
    ids = tokenizer.encode(text)
    print(f"tokenized: {len(ids):,} tokens")
    dataset = TokenDataset(ids, val_fraction=args.val_fraction)

    if args.epochs:
        tokens_per_step = batch_size * model_config.block_size * args.grad_accum
        max_steps = max(1, math.ceil(dataset.n_train * args.epochs / tokens_per_step))
        print(f"--epochs {args.epochs} -> {max_steps} steps")

    model_config.vocab_size = tokenizer.vocab_size
    model = GPT(model_config)
    train_cfg = TrainConfig(
        batch_size=batch_size,
        grad_accum_steps=args.grad_accum,
        max_steps=max_steps,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
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
        resume=args.resume,
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

    trainer = Trainer(model, dataset, train_cfg, tokenizer, device=device)
    t0 = time.time()
    stats = trainer.train()
    print(f"total time: {time.time()-t0:.0f}s | {stats}")

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

    print(f"\ncheckpoint: {Path(args.out_dir)/'model.pt'}")
    print(f"chat with it:  python generate.py --checkpoint {args.out_dir} --chat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
