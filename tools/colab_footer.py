# ---------------------------------------------------------------------------
# Built-in assistant corpus (packed in so this file works with zero downloads)
# ---------------------------------------------------------------------------
EMBEDDED_CHAT_CORPUS = r"""@@CHAT_CORPUS@@"""

_original_load_source = load_source


def load_source(source, cache_dir=DEFAULT_CACHE_DIR, verbose=True):  # noqa: F811
    if source == "orbit-chat":
        return EMBEDDED_CHAT_CORPUS
    if source == "conversation":      # generated locally, no download needed
        return build_conversation_corpus()
    if source == "prose":             # stage 1 of SFT
        return build_prose_corpus()
    return _original_load_source(source, cache_dir, verbose)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def install_dependencies(packages=("ddgs",)):
    """Install the optional pip packages we can live without."""
    import subprocess

    for package in packages:
        try:
            __import__(package)
            continue
        except Exception:
            pass
        try:
            print(f"installing {package} ...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package],
                check=True, timeout=300,
            )
        except Exception as exc:
            print(f"  ! could not install {package} ({exc}) - continuing without it")


def pick_preset(preset, device):
    if preset not in (None, "", "auto"):
        return preset
    return "micro" if device.type == "cuda" else DEFAULT_PRESET


def token_cache_dir():
    import os

    return Path(os.path.expanduser("~/.cache/orbit-gpt"))


def describe_checkpoints(directory):
    """One line per checkpoint file, so it is obvious what is on disk."""
    directory = Path(directory)
    if not directory.is_dir():
        return "  (none yet)"
    lines = []
    for path in sorted(directory.glob("*.pt")):
        lines.append(f"  {path.name:24s} {path.stat().st_size/1e6:6.1f} MB")
    return "\n".join(lines) if lines else "  (none yet)"


def run_stage(
    label,
    text,
    tokenizer,
    model_config,
    init_path,
    steps,
    learning_rate,
    warmup,
    batch_size,
    max_epochs,
    out_dir,
    device,
    seed,
):
    """Train one stage of the pipeline and return (model, tokenizer).

    Stage 1 (pre-training) learns sentence structure from plain prose.
    Stage 2 (SFT) starts from those weights and learns the chat format.
    """
    print("")
    print("-" * 72)
    print(f"{label}: {steps} steps, lr {learning_rate:g}, batch {batch_size}")
    print("-" * 72)

    if tokenizer is None:
        tokenizer, cached = build_tokenizer(
            text, "bpe", CONFIG["vocab_size"], cache_dir=token_cache_dir()
        )
        ids = cached if cached is not None else tokenizer.encode(text)
    else:
        ids = tokenizer.encode(text)   # stage 2 reuses stage 1's vocabulary
    dataset = TokenDataset(ids, val_fraction=0.1)
    print(f"tokens: {len(ids):,}")

    model_config.vocab_size = tokenizer.vocab_size
    model = GPT(model_config)

    if max_epochs:
        per_step = batch_size * model_config.block_size
        cap = max(1, math.ceil(dataset.n_train * max_epochs / per_step))
        if cap < steps:
            print(f"capping {steps} steps at {cap} ({max_epochs} epochs)")
            steps = cap

    cfg = TrainConfig(
        batch_size=batch_size,
        max_steps=steps,
        learning_rate=learning_rate,
        min_learning_rate=learning_rate / 10.0,
        warmup_steps=warmup,
        weight_decay=CONFIG["weight_decay"],
        grad_clip=CONFIG["grad_clip"],
        seed=seed,
        out_dir=str(out_dir),
        device=str(device),
        save_interval=CONFIG["save_interval"],
        init_from=init_path or "",
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "train_config.json").write_text(json.dumps({
        "model_config": model_config.to_dict(),
        "train_config": cfg.to_dict(),
        "stage": label,
    }, indent=2))
    trainer = Trainer(model, dataset, cfg, tokenizer, device=device)
    trainer.train()
    return model, tokenizer, steps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train and chat with a tiny GPT.")
    p.add_argument("--preset", default=CONFIG["preset"])
    p.add_argument("--vocab-size", type=int, default=CONFIG["vocab_size"])
    p.add_argument("--block-size", type=int, default=CONFIG["block_size"])
    p.add_argument("--n-layer", type=int, default=CONFIG["n_layer"])
    p.add_argument("--n-head", type=int, default=CONFIG["n_head"])
    p.add_argument("--n-embd", type=int, default=CONFIG["n_embd"])
    p.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--max-steps", type=int, default=CONFIG["max_steps"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--out-dir", default=CONFIG["out_dir"],
                   help='"" = auto: <repo>/checkpoints/orbit, or '
                        "/content/checkpoints/orbit in Colab")
    p.add_argument("--device", default="auto")
    p.add_argument("--retrain", action="store_true",
                   help="train from scratch even if a saved model exists")
    p.add_argument("--resume", nargs="?", const="auto", default="",
                   help="continue training the saved model ('auto' = newest "
                        "checkpoint in --out-dir)")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="skip stage 1 and fine-tune on the chat corpus only")
    p.add_argument("--no-search", action="store_true", help="disable web search")
    p.add_argument("--no-chat", action="store_true")
    p.add_argument("--no-sample", action="store_true")
    args = p.parse_args(argv)

    print("=" * 72)
    print("  OrbitGPT - train a small language model on your own machine")
    print("=" * 72)

    # 1. dependencies ------------------------------------------------------
    use_web_search = CONFIG["use_web_search"] and not args.no_search
    if use_web_search:
        install_dependencies(("ddgs",))

    # 2. device ------------------------------------------------------------
    device = auto_device(args.device)
    name = pick_preset(args.preset, device)
    print("device: " + str(device)
          + (" (" + torch.cuda.get_device_name(0) + ")" if device.type == "cuda" else ""))
    print("preset: " + name)
    # CONFIG["vocab_size"] = None means "whatever this preset wants"
    if not CONFIG["vocab_size"]:
        CONFIG["vocab_size"] = preset_train_defaults(name, device.type)["vocab_size"]

    # 3. where checkpoints live (never Google Drive) -----------------------
    out_dir = Path(args.out_dir) if args.out_dir else default_checkpoint_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("checkpoints: " + str(out_dir))
    existing = latest_checkpoint(out_dir)

    # 4. reuse, resume, or train -------------------------------------------
    retrain = args.retrain or CONFIG["retrain"]
    if existing is not None and not retrain and not args.resume:
        print("")
        print("Found a trained model: " + str(existing))
        print("Loading it - no waiting. "
              "(--retrain = train a new one, --resume = keep training)")
        model, tokenizer = load_model(str(out_dir), device=str(device))
        print("loaded %.2fM parameters" % (model.n_params / 1e6))
    else:
        if args.resume and existing is not None:
            print("")
            print("Continuing training from " + str(existing))
            model = GPT.from_checkpoint(str(existing), device=str(device))
            tokenizer = load_tokenizer(str(out_dir))
            model_config = model.config
            init, tokenizer, do_pretrain = str(existing), tokenizer, False
        else:
            if existing is not None:
                print("")
                print("--retrain: ignoring the saved model and starting fresh")
            set_seed(args.seed)
            model_config = get_preset(name)
            for key, value in (
                ("n_layer", args.n_layer), ("n_head", args.n_head),
                ("n_embd", args.n_embd), ("block_size", args.block_size),
                ("dropout", CONFIG["dropout"]),
            ):
                if value is not None:
                    setattr(model_config, key, value)
            init, tokenizer = "", None
            do_pretrain = CONFIG["pretrain_corpus"] and not args.skip_pretrain

        td = preset_train_defaults(name, device.type)
        batch_size = args.batch_size or td["batch_size"]
        total_steps = args.max_steps or td.get("max_steps", 2000)
        pretrain_steps = CONFIG["pretrain_steps"]
        sft_steps = CONFIG["sft_steps"]
        if pretrain_steps is None:
            pretrain_steps = int(total_steps * CONFIG["pretrain_fraction"])
        if sft_steps is None:
            sft_steps = max(1, total_steps - (pretrain_steps if do_pretrain else 0))
        sft_lr = CONFIG["sft_lr"] or td["learning_rate"] / 3.0

        if do_pretrain:
            # ---- stage 1: language modelling on plain prose --------------
            print("")
            print("loading corpus: " + CONFIG["pretrain_corpus"])
            try:
                prose = load_corpus(CONFIG["pretrain_corpus"])
            except Exception as exc:
                print("  ! could not load %r: %s" % (CONFIG["pretrain_corpus"], exc))
                prose = None
            if prose:
                pretrain_dir = out_dir / "pretrain"
                _, tokenizer, used = run_stage(
                    "Stage 1/2 - pre-training (prose)",
                    prose, tokenizer, model_config, init,
                    pretrain_steps, td["learning_rate"], td["warmup_steps"],
                    batch_size, CONFIG["pretrain_epochs"], pretrain_dir, device,
                    args.seed,
                )
                init = str(pretrain_dir / "model.pt")
                # an epoch cap can cut stage 1 short; hand the rest to stage 2
                if used < pretrain_steps:
                    sft_steps = max(1, total_steps - used)

        # ---- stage 2: SFT on the chat corpus ------------------------------
        print("")
        print("loading corpus: " + CONFIG["sft_corpus"])
        try:
            text = load_corpus(CONFIG["sft_corpus"])
        except Exception as exc:
            print("  ! could not load %r: %s" % (CONFIG["sft_corpus"], exc))
            print("  ! falling back to the embedded assistant corpus")
            text = load_corpus("orbit-chat:20")
        model, tokenizer, _ = run_stage(
            "Stage 2/2 - SFT (chat format)",
            text, tokenizer, model_config, init,
            sft_steps, sft_lr, max(20, td["warmup_steps"] // 2),
            batch_size, CONFIG["sft_epochs"], out_dir, device, args.seed,
        )

        if CONFIG["sample_after_train"] and not args.no_sample:
            print("")
            print("-" * 72)
            for prompt, temp in (
                ("User: What is 47 + 86?\nAssistant:", 0.4),
                ("User: Can you explain what recursion is?\nAssistant:", 0.6),
            ):
                print("")
                print(">>> " + prompt)
                generate(model, tokenizer, prompt, max_new_tokens=100,
                         temperature=temp, top_k=40, top_p=0.95,
                         stop_strings=["\nUser:"], device=device, stream=True,
                         repetition_penalty=CONFIG["chat_repetition_penalty"])

    # 5. report where the weights are --------------------------------------
    print("")
    print("Checkpoints in " + str(out_dir))
    print(describe_checkpoints(out_dir))

    # 6. inference (with optional web search) ------------------------------
    if CONFIG["chat_after_train"] and not args.no_chat:
        chat(model, tokenizer, device=device,
             temperature=CONFIG["chat_temperature"],
             max_new_tokens=CONFIG["chat_tokens"],
             repetition_penalty=CONFIG["chat_repetition_penalty"],
             use_web_search=use_web_search,
             web_results=CONFIG["web_results"])

    print("")
    print("Resume training:  %run orbit_gpt_colab.py --resume --max-steps 4000")
    print("Train again:      %run orbit_gpt_colab.py --retrain")
    print("Just chat:        %run orbit_gpt_colab.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
