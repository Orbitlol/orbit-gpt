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
    return _original_load_source(source, cache_dir, verbose)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def install_dependencies(packages=("ddgs",)):
    """Install the optional pip packages we can live without.

    PyTorch is expected to be present (Colab ships it); everything else is
    best-effort - if it fails the run continues with that feature switched off.
    """
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


def describe_checkpoints(directory):
    """One line per checkpoint file, so it is obvious what is on disk."""
    directory = Path(directory)
    if not directory.is_dir():
        return "  (none yet)"
    lines = []
    for path in sorted(directory.glob("*.pt")):
        lines.append(f"  {path.name:24s} {path.stat().st_size/1e6:6.1f} MB")
    return "\n".join(lines) if lines else "  (none yet)"


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
    p.add_argument("--max-epochs", type=float, default=CONFIG["max_epochs"])
    p.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    p.add_argument("--dropout", type=float, default=CONFIG["dropout"])
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
    p.add_argument("--no-search", action="store_true", help="disable web search")
    p.add_argument("--no-chat", action="store_true")
    p.add_argument("--no-sample", action="store_true")
    p.add_argument("--prompt", default="User: Hello!\nAssistant:")
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
        else:
            if existing is not None:
                print("")
                print("--retrain: ignoring the saved model and starting fresh")
            set_seed(args.seed)
            model_config = get_preset(name)
            for key, value in (
                ("n_layer", args.n_layer), ("n_head", args.n_head),
                ("n_embd", args.n_embd), ("block_size", args.block_size),
                ("dropout", args.dropout),
            ):
                if value is not None:
                    setattr(model_config, key, value)
            tokenizer = None

        train_defaults = preset_train_defaults(name, device.type)
        batch_size = args.batch_size or train_defaults["batch_size"]
        max_steps = args.max_steps or train_defaults.get("max_steps", 2000)
        learning_rate = args.lr or train_defaults["learning_rate"]
        max_epochs = (
            train_defaults["max_epochs"] if args.max_epochs is None else args.max_epochs
        )
        print("")
        print("loading corpus: " + args.corpus)
        try:
            text = load_corpus(args.corpus)
        except Exception as exc:
            # offline / blocked CDN: keep going with the embedded corpus
            print("  ! could not load %r: %s" % (args.corpus, exc))
            print("  ! falling back to the generated conversation corpus")
            text = load_corpus("conversation")
        print("corpus: %s characters" % format(len(text), ","))
        print("")

        if tokenizer is None:
            tokenizer = build_tokenizer(text, "bpe", args.vocab_size)
        ids = tokenizer.encode(text)
        dataset = TokenDataset(ids, val_fraction=0.1)
        print("tokenized: %s tokens" % format(len(ids), ","))

        model_config.vocab_size = tokenizer.vocab_size
        if not (args.resume and existing is not None):
            model = GPT(model_config)

        # do not grind over the same text until it is memorised
        if max_epochs:
            per_step = batch_size * model_config.block_size
            epoch_cap = max(1, math.ceil(dataset.n_train * max_epochs / per_step))
            if epoch_cap < max_steps:
                print("capping %d steps at %d (%s epochs over %s tokens)"
                      % (max_steps, epoch_cap, max_epochs,
                         format(dataset.n_train, ",")))
                max_steps = epoch_cap

        train_cfg = TrainConfig(
            batch_size=batch_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            min_learning_rate=train_defaults["min_learning_rate"],
            warmup_steps=train_defaults["warmup_steps"],
            weight_decay=train_defaults["weight_decay"],
            grad_clip=train_defaults["grad_clip"],
            grad_accum_steps=train_defaults["grad_accum_steps"],
            seed=args.seed,
            out_dir=str(out_dir),
            device=str(device),
            save_interval=CONFIG["save_interval"],
            resume=str(existing) if (args.resume and existing is not None) else "",
        )
        (out_dir / "train_config.json").write_text(json.dumps({
            "model_config": model_config.to_dict(),
            "train_config": train_cfg.to_dict(),
            "corpus": args.corpus,
            "preset": name,
        }, indent=2))
        trainer = Trainer(model, dataset, train_cfg, tokenizer, device=device)
        trainer.train()

        if CONFIG["sample_after_train"] and not args.no_sample:
            print("")
            print("-" * 72)
            for prompt, temp in (
                ("User: What is 47 + 86?\nAssistant:", 0.4),
                (args.prompt, 0.7),
            ):
                print("")
                print(">>> " + prompt)
                generate(model, tokenizer, prompt, max_new_tokens=100, temperature=temp,
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
