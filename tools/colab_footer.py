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
# Train once, reuse forever
# ---------------------------------------------------------------------------
def mount_drive():
    """Mount Google Drive so the checkpoint survives a Colab session restart."""
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        path = Path("/content/drive/MyDrive/orbit-gpt")
        path.mkdir(parents=True, exist_ok=True)
        print("google drive mounted: " + str(path))
        return path
    except Exception as exc:
        print("(google drive unavailable: %s)" % exc)
        print("  -> the model will only live for this session")
        return None


def find_checkpoint(*dirs):
    """First directory holding both a model and its tokenizer, else None."""
    for d in dirs:
        d = Path(d)
        if (d / "model.pt").exists() and (d / "tokenizer.json").exists():
            return d
    return None


def pick_preset(preset, device):
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
    p.add_argument("--max-epochs", type=float, default=CONFIG["max_epochs"])
    p.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    p.add_argument("--dropout", type=float, default=CONFIG["dropout"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--out-dir", default=CONFIG["out_dir"])
    p.add_argument("--device", default="auto")
    p.add_argument("--retrain", action="store_true",
                   help="train from scratch even if a saved model exists")
    p.add_argument("--resume", action="store_true",
                   help="continue training the saved model instead of starting over")
    p.add_argument("--no-drive", action="store_true", help="do not touch Google Drive")
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
    print("device: " + str(device)
          + (" (" + torch.cuda.get_device_name(0) + ")" if device.type == "cuda" else ""))
    print("preset: " + name)

    # ---- where does the model live? -------------------------------------
    local_dir = Path(args.out_dir)
    drive_dir = mount_drive() if (CONFIG["save_to_drive"] and not args.no_drive) else None
    save_dir = (drive_dir / local_dir.name) if drive_dir else local_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = find_checkpoint(save_dir, local_dir)
    retrain = args.retrain or CONFIG["retrain"]

    if existing is not None and not retrain and not args.resume:
        print("")
        print("Found a trained model in " + str(existing))
        print("Loading it - no waiting. "
              "(--retrain = train a new one, --resume = keep training)")
        model, tokenizer = load_model(str(existing), device=str(device))
        print("loaded %.2fM parameters" % (model.n_params / 1e6))
    else:
        if args.resume and existing is not None:
            print("")
            print("Continuing training from " + str(existing))
            model = GPT.from_checkpoint(str(existing / "model.pt"), device=str(device))
            tokenizer = load_tokenizer(str(existing))
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

        batch_size = args.batch_size or defaults["batch_size"]
        max_steps = args.max_steps or defaults["max_steps"]

        print("")
        print("loading corpus: " + args.corpus)
        try:
            text = load_corpus(args.corpus)
        except Exception as exc:
            # offline / blocked CDN: keep going with the embedded corpus
            print("  ! could not load %r: %s" % (args.corpus, exc))
            print("  ! falling back to the built-in assistant corpus")
            text = load_corpus("orbit-chat:20")
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
        if args.max_epochs:
            per_step = batch_size * model_config.block_size
            epoch_cap = max(1, math.ceil(dataset.n_train * args.max_epochs / per_step))
            if epoch_cap < max_steps:
                print("capping %d steps at %d (%s epochs over %s tokens)"
                      % (max_steps, epoch_cap, args.max_epochs,
                         format(dataset.n_train, ",")))
                max_steps = epoch_cap

        train_cfg = TrainConfig(
            batch_size=batch_size,
            max_steps=max_steps,
            learning_rate=args.lr,
            warmup_steps=min(100, max(1, max_steps // 10)),
            seed=args.seed,
            out_dir=str(save_dir),
            device=str(device),
            resume=str(existing / "model.pt") if (args.resume and existing is not None) else "",
        )
        (save_dir / "train_config.json").write_text(json.dumps({
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
                         stop_strings=["\nUser:"], device=device, stream=True)
        try:  # make the checkpoint easy to download from Colab
            import shutil
            archive = shutil.make_archive(str(local_dir), "zip", save_dir)
            print("")
            print("zipped checkpoint: " + archive)
            from google.colab import files  # type: ignore
            files.download(archive)
        except Exception:
            pass

    if CONFIG["chat_after_train"] and not args.no_chat:
        chat(model, tokenizer, device=device,
             temperature=CONFIG["chat_temperature"],
             max_new_tokens=CONFIG["chat_tokens"],
             repetition_penalty=CONFIG["chat_repetition_penalty"])

    print("")
    print("Model saved in: " + str(save_dir))
    print("Next time just run this cell again - it loads the saved model instead "
          "of training.")
    print("  --retrain   train from scratch")
    print("  --resume    keep training the saved model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
