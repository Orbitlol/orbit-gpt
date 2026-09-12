"""Stage-1 prose corpus, SFT weight transfer and the ONNX export.

Everything here is cheap (or skipped when the export stack is missing), so it
runs inside tests/test_smoke.py on every push.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# prose corpus (stage 1 of the SFT pipeline)
# ---------------------------------------------------------------------------
def test_prose_corpus_is_registered_and_deterministic():
    from orbit_gpt.data import load_source

    text = load_source("prose")
    assert len(text) > 100_000, "the prose corpus is far too small to teach grammar"
    assert load_source("prose") == text, "the corpus must be deterministic"


def test_prose_sentences_are_grammatical_shapes():
    """No double stops, no lowercase paragraph starts, no repeated sentence."""
    from orbit_gpt.corpora.prose import build_prose_corpus

    text = build_prose_corpus(seed=7)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    assert len(paragraphs) > 500

    assert not re.search(r"(?:^|\n\n)[a-z]", text), "paragraphs must start uppercase"
    # "key=..." inside a code snippet is fine, a real double stop is not
    assert not re.search(r"[a-z]\.\.[a-z]", text), "no double full stops"
    for paragraph in paragraphs[:400]:
        assert paragraph[0].isupper()
        assert paragraph.rstrip()[-1] in ".!?", paragraph
        sentences = [s.strip() for s in paragraph.split(". ") if s.strip()]
        assert len(sentences) == len(set(sentences)), f"repeated: {paragraph}"


def test_prose_reuses_the_chat_knowledge():
    """The facts it writes about are the same ones the chat corpus answers."""
    from orbit_gpt.corpora.conversation import FACTS
    from orbit_gpt.corpora.prose import build_prose_corpus

    text = build_prose_corpus(seed=3)
    for topic, _answers in list(FACTS.items())[:25]:
        assert topic.lower() in text.lower(), f"{topic} missing from the prose corpus"


# ---------------------------------------------------------------------------
# SFT: --init-from keeps the weights, drops the schedule
# ---------------------------------------------------------------------------
def test_init_weights_from_copies_matching_tensors():
    import torch

    from orbit_gpt.config import GPTConfig
    from orbit_gpt.model import GPT
    from orbit_gpt.train import Trainer

    class _Dummy:  # Trainer.__init__ is heavy; we only want the loader
        pass

    device = torch.device("cpu")
    source = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2,
                           n_embd=32))
    with tempfile.TemporaryDirectory() as tmp:
        source.save(Path(tmp) / "model.pt")

        model = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2,
                              n_embd=32))
        trainer = _Dummy()
        trainer.model = model
        trainer.device = device
        trainer.verbose = False
        Trainer.init_weights_from(trainer, str(Path(tmp) / "model.pt"))

        for a, b in zip(source.parameters(), model.parameters()):
            assert torch.equal(a.detach(), b.detach()), "weights did not transfer"


def test_init_weights_from_survives_a_shape_mismatch():
    """A narrower model keeps what fits instead of crashing (e.g. new vocab)."""
    import torch

    from orbit_gpt.config import GPTConfig
    from orbit_gpt.model import GPT
    from orbit_gpt.train import Trainer

    class _Dummy:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        wide = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2,
                             n_embd=32))
        wide.save(Path(tmp) / "model.pt")

        small = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2,
                              n_embd=16))
        before = [p.detach().clone() for p in small.parameters()]
        trainer = _Dummy()
        trainer.model = small
        trainer.device = torch.device("cpu")
        trainer.verbose = False
        Trainer.init_weights_from(trainer, str(Path(tmp) / "model.pt"))

        # nothing should have changed: every tensor has a different shape
        for a, b in zip(before, small.parameters()):
            assert torch.equal(a, b.detach())


def test_train_config_has_a_separate_init_from():
    from orbit_gpt.config import TrainConfig

    assert TrainConfig().init_from == ""
    assert hasattr(TrainConfig(), "resume")      # and resume is still there


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def _export_stack() -> bool:
    try:
        import onnx
        import onnxruntime
        import torch.onnx
    except Exception:
        return False
    return all(m is not None for m in (onnx, onnxruntime, torch.onnx))


def test_onnx_export_matches_pytorch():
    if not _export_stack():
        return  # the export stack is optional; nothing to check

    import torch

    from orbit_gpt.config import GPTConfig
    from orbit_gpt.export import export_onnx
    from orbit_gpt.model import GPT
    from orbit_gpt.tokenizer import BPETokenizer

    torch.manual_seed(0)
    config = GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2,
                       n_embd=32)
    model = GPT(config)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model.save(tmp / "model.pt")
        text = ("User: hello there\nAssistant: hello, how can I help? "
                "Recursion is when a function calls itself. " * 40)
        BPETokenizer().train(text, vocab_size=300).save(str(tmp))

        export_onnx(tmp, tmp / "orbit.onnx", quantize=False, verbose=False)
        assert (tmp / "orbit.onnx").is_file()
        assert (tmp / "tokenizer.json").is_file()

        from orbit_gpt.export import check_onnx

        worst = check_onnx(tmp / "orbit.onnx", tmp, steps=4, tol=1e-3)
        assert worst < 1e-3
