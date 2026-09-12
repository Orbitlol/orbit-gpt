"""Checkpoint location, naming and resuming (run via tests/test_smoke.py)."""

from __future__ import annotations

import tempfile
from pathlib import Path


def test_default_checkpoint_dir_is_local_and_gitignored():
    """Never Google Drive: repo checkout, /content, or ./checkpoints."""
    from orbit_gpt.checkpoints import default_checkpoint_dir, repo_root

    directory = default_checkpoint_dir()
    assert "drive" not in str(directory).lower()
    assert directory.name == "orbit"
    root = repo_root()
    if root is not None:  # running inside the checkout
        assert str(directory).startswith(str(root))

        # a gitignore comment on the same line would silently break the rule,
        # so check the pattern is on a line of its own
        patterns = {
            line.strip()
            for line in (root / ".gitignore").read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        assert "checkpoints/" in patterns
        assert "*.pt" in patterns


def test_checkpoint_names_are_sorted_and_predictable():
    from orbit_gpt.checkpoints import (
        BEST_NAME,
        LATEST_NAME,
        step_name,
    )

    assert BEST_NAME == "model.pt"
    assert LATEST_NAME == "model-latest.pt"
    assert step_name(500) < step_name(1500)      # lexicographic sort works


def test_latest_checkpoint_prefers_the_freshest_save():
    from orbit_gpt.checkpoints import latest_checkpoint

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert latest_checkpoint(d) is None
        (d / "model-step000250.pt").write_bytes(b"x")
        assert latest_checkpoint(d).name == "model-step000250.pt"
        (d / "model.pt").write_bytes(b"x")
        assert latest_checkpoint(d).name == "model.pt"        # best wins over old
        (d / "model-latest.pt").write_bytes(b"x")
        assert latest_checkpoint(d).name == "model-latest.pt"  # freshest wins
        (d / "model-step002000.pt").write_bytes(b"x")
        assert latest_checkpoint(d).name == "model-latest.pt"


def test_resolve_resume_understands_auto_and_paths():
    from orbit_gpt.checkpoints import resolve_resume

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert resolve_resume(d, "") is None
        assert resolve_resume(d, "auto") is None
        (d / "model-latest.pt").write_bytes(b"x")
        for flag in ("", "auto", "latest", "1", "True"):
            assert resolve_resume(d, flag) == d / "model-latest.pt"
        assert resolve_resume(d, str(d)) == d / "model-latest.pt"
        assert resolve_resume(d, str(d / "nope.pt")) is None
        assert resolve_resume(d, str(d / "model-latest.pt")) == d / "model-latest.pt"


def test_keep_last_n_deletes_old_snapshots():
    from orbit_gpt.checkpoints import keep_last_n, step_name

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for step in (250, 500, 750, 1000):
            (d / step_name(step)).write_bytes(b"x")
        keep_last_n(d, n=2)
        names = sorted(p.name for p in d.glob("model-step*.pt"))
        assert names == ["model-step000750.pt", "model-step001000.pt"]


def test_trainer_writes_best_and_latest(tmpdir=None):
    """A short run must leave both a best and a resumable checkpoint."""
    import tempfile

    import torch

    from orbit_gpt.config import TrainConfig, get_preset
    from orbit_gpt.data import TokenDataset
    from orbit_gpt.model import GPT
    from orbit_gpt.tokenizer import BPETokenizer
    from orbit_gpt.train import Trainer

    with tempfile.TemporaryDirectory() as d:
        corpus = ("User: hello\nAssistant: hi there\n\n" * 200)
        tok = BPETokenizer().train(corpus, vocab_size=300)
        dataset = TokenDataset(tok.encode(corpus), val_fraction=0.1)
        cfg = get_preset("nano")
        cfg.vocab_size = tok.vocab_size
        model = GPT(cfg)
        trainer = Trainer(
            model,
            dataset,
            TrainConfig(
                batch_size=8, max_steps=6, eval_interval=3, log_interval=3,
                save_interval=3, out_dir=d,
            ),
            tok,
            device=torch.device("cpu"),
            verbose=False,
        )
        trainer.train()

        out = Path(d)
        assert (out / "model.pt").exists(), "best checkpoint missing"
        assert (out / "model-latest.pt").exists(), "resumable checkpoint missing"
        assert (out / "tokenizer.json").exists()
        snapshots = sorted(out.glob("model-step*.pt"))
        assert snapshots, "no periodic snapshot was written"

        # and the checkpoint knows where it stopped
        import orbit_gpt.checkpoints as ckpt

        assert ckpt.latest_checkpoint(out) is not None
        state = torch.load(out / "model-latest.pt", map_location="cpu")
        assert state["extra"]["step"] == 6


def test_training_resumes_from_the_latest_checkpoint():
    """--resume auto continues at the step the previous run stopped at."""
    import tempfile

    import torch

    from orbit_gpt.checkpoints import resolve_resume
    from orbit_gpt.config import TrainConfig, get_preset
    from orbit_gpt.data import TokenDataset
    from orbit_gpt.model import GPT
    from orbit_gpt.tokenizer import BPETokenizer
    from orbit_gpt.train import Trainer

    corpus = ("User: hello\nAssistant: hi there\n\n" * 200)
    tok = BPETokenizer().train(corpus, vocab_size=300)
    dataset = TokenDataset(tok.encode(corpus), val_fraction=0.1)
    cfg = get_preset("nano")
    cfg.vocab_size = tok.vocab_size

    with tempfile.TemporaryDirectory() as d:
        first = Trainer(
            GPT(cfg), dataset,
            TrainConfig(batch_size=8, max_steps=4, eval_interval=1000,
                        log_interval=1000, out_dir=d),
            tok, device=torch.device("cpu"), verbose=False,
        )
        first.train()
        first.save_latest()
        stopped_at = first.step

        resume = resolve_resume(d, "auto")
        assert resume is not None
        assert torch.load(resume, map_location="cpu")["extra"]["step"] == stopped_at

        # a fresh trainer picks the run back up instead of starting over
        second = Trainer(
            GPT(cfg), dataset,
            TrainConfig(batch_size=8, max_steps=stopped_at + 3, eval_interval=1000,
                        log_interval=1000, out_dir=d,
                        resume=str(resume)),
            tok, device=torch.device("cpu"), verbose=False,
        )
        second.train()
        assert second.step == stopped_at + 3


def test_human_size_reports_units():
    from orbit_gpt.checkpoints import human_size

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "f"
        path.write_bytes(b"x" * 2048)
        assert human_size(path).endswith("KB")
