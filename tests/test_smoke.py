"""Fast, dependency-light tests for the whole pipeline.

Run them either way::

    python tests/test_smoke.py      # no pytest needed
    pytest tests/ -q
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbit_gpt.config import GPTConfig, TrainConfig, get_preset  # noqa: E402
from orbit_gpt.data import TokenDataset, parse_spec  # noqa: E402
from orbit_gpt.generate import format_chat, load_model  # noqa: E402
from orbit_gpt.model import GPT  # noqa: E402
from orbit_gpt.tokenizer import (  # noqa: E402
    BPETokenizer,
    CharTokenizer,
    _pretokenize,
    load_tokenizer,
)
from orbit_gpt.train import Trainer  # noqa: E402

TINY_CORPUS = (
    "The following is a conversation between a human (User) and an AI assistant "
    "named Orbit.\n\n"
    "User: What is 2 + 2?\nAssistant: 2 + 2 is 4.\n\n"
    "User: What is the capital of France?\nAssistant: The capital of France is Paris.\n\n"
) * 40


# --------------------------------------------------------------------------- tokenizer
def test_pretokenize_glues_whitespace():
    assert _pretokenize("Hello world") == ["Hello", " world"]
    assert _pretokenize("Hi!\n\nThere") == ["Hi", "!", "\n\n", "There"]
    assert _pretokenize("a1b") == ["a", "1", "b"]


def test_bpe_roundtrip():
    tok = BPETokenizer().train(TINY_CORPUS, vocab_size=300, min_pair_freq=2)
    ids = tok.encode(TINY_CORPUS)
    assert tok.decode(ids) == TINY_CORPUS
    assert max(ids) < tok.vocab_size
    # BPE should compress plain text by at least 1.5x on this corpus
    assert len(TINY_CORPUS.encode()) / len(ids) > 1.5
    assert tok.vocab_size == 300


def test_bpe_handles_unicode_and_unseen_text():
    tok = BPETokenizer().train(TINY_CORPUS, vocab_size=300)
    weird = "héllo wörld 😀 — 42°C ünïcode"
    assert tok.decode(tok.encode(weird)) == weird


def test_bpe_save_load(tmpdir):
    tok = BPETokenizer().train(TINY_CORPUS, vocab_size=300)
    ids = tok.encode(TINY_CORPUS)
    tok.save(tmpdir)
    loaded = load_tokenizer(tmpdir)
    assert isinstance(loaded, BPETokenizer)
    assert loaded.encode(TINY_CORPUS) == ids
    assert loaded.decode(ids) == TINY_CORPUS


def test_char_tokenizer():
    tok = CharTokenizer().train(TINY_CORPUS, vocab_size=300)
    ids = tok.encode(TINY_CORPUS)
    assert tok.decode(ids) == TINY_CORPUS
    assert max(ids) < tok.vocab_size
    tok.save(tmpdir := tempfile.mkdtemp())
    assert load_tokenizer(tmpdir).decode(ids) == TINY_CORPUS


def test_special_tokens():
    tok = BPETokenizer().train(TINY_CORPUS, vocab_size=300)
    assert tok.pad_id == 0 and tok.eot_id == 1
    ids = tok.encode("hello <|endoftext|> world")
    assert tok.eot_id in ids
    assert "<|endoftext|>" not in tok.decode(ids)  # skipped by default
    assert "<|endoftext|>" in tok.decode(ids, skip_special=False)


# --------------------------------------------------------------------------- data
def test_dataset_batches():
    tok = BPETokenizer().train(TINY_CORPUS, vocab_size=300)
    ds = TokenDataset(tok.encode(TINY_CORPUS * 3), val_fraction=0.1)
    x, y = ds.get_batch("train", 4, 16, torch.device("cpu"))
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])  # targets are inputs shifted by one


def test_parse_spec():
    assert parse_spec("shakespeare") == [("shakespeare", 1)]
    assert parse_spec("shakespeare,orbit-chat:12") == [("shakespeare", 1), ("orbit-chat", 12)]
    assert parse_spec("./notes.txt") == [("./notes.txt", 1)]
    assert parse_spec("https://example.com/a.txt") == [("https://example.com/a.txt", 1)]


# --------------------------------------------------------------------------- model
def _tiny_model(vocab_size=64, block_size=32, dropout=0.0) -> GPT:
    cfg = GPTConfig(
        vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=4,
        n_embd=64, dropout=dropout,
    )
    return GPT(cfg).eval()


def test_forward_shapes_and_loss():
    m = _tiny_model()
    idx = torch.randint(0, 64, (2, 16))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 16, 64)
    assert loss is not None and loss.item() > 0
    # an untrained model is close to uniform (a little under ln(vocab) because
    # the residual stream carries the input embedding, i.e. a "copy" bias)
    assert 2.8 < loss.item() < math.log(64) + 0.2


def test_kv_cache_matches_full_recompute():
    """The KV cache is an optimisation - it must not change the output."""
    m = _tiny_model()
    prompt = [5, 9, 22, 41, 3, 7]
    m.reset_cache()
    out, _ = m(torch.tensor([prompt]), use_cache=True)
    cached, cur = [], out[:, -1, :].argmax(-1)
    for _ in range(8):
        cached.append(int(cur))
        out, _ = m(cur.view(1, 1), use_cache=True)
        cur = out[:, -1, :].argmax(-1)

    seq, plain = list(prompt), []
    for _ in range(8):
        out, _ = m(torch.tensor([seq]))
        nxt = int(out[:, -1, :].argmax(-1))
        plain.append(nxt)
        seq.append(nxt)
    assert cached == plain


def test_generation_respects_block_size():
    m = _tiny_model(block_size=32)
    out = m.generate([1, 2, 3], max_new_tokens=500, temperature=1.0, top_k=5)
    assert len(out) == 32 - 3  # context window is never exceeded
    long_prompt = list(range(40))
    out = m.generate(long_prompt, max_new_tokens=10, temperature=1.0, top_k=5)
    assert len(out) == 0  # prompt already fills the window


def test_generation_is_reproducible_with_seed():
    m = _tiny_model()
    a = m.generate([1, 2, 3], max_new_tokens=12, temperature=1.0, seed=7)
    b = m.generate([1, 2, 3], max_new_tokens=12, temperature=1.0, seed=7)
    assert a == b


def test_checkpoint_roundtrip(tmpdir=None):
    tmpdir = tmpdir or tempfile.mkdtemp()
    m = _tiny_model()
    m.save(str(Path(tmpdir) / "model.pt"))
    m2 = GPT.from_checkpoint(str(Path(tmpdir) / "model.pt"))
    x = torch.randint(0, 64, (1, 8))
    assert torch.allclose(m(x)[0], m2(x)[0], atol=1e-6)


def test_presets_are_valid():
    for name in ("nano", "micro", "mini", "small", "base"):
        cfg = get_preset(name)
        cfg.vocab_size = 1024
        cfg.validate()
        assert cfg.n_embd % cfg.n_head == 0
    assert get_preset("micro").n_layer == 6


# --------------------------------------------------------------------------- training
def test_trainer_reduces_loss():
    tok = BPETokenizer().train(TINY_CORPUS, vocab_size=300)
    ds = TokenDataset(tok.encode(TINY_CORPUS * 4), val_fraction=0.1)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=4, n_embd=64)
    model = GPT(cfg)
    out_dir = tempfile.mkdtemp()
    train_cfg = TrainConfig(
        batch_size=8, max_steps=60, eval_interval=30, log_interval=0,
        warmup_steps=5, out_dir=out_dir, learning_rate=5e-3,
    )
    trainer = Trainer(model, ds, train_cfg, tok, device=torch.device("cpu"), verbose=False)
    first = trainer.estimate_loss()
    trainer.train()
    assert trainer.best_val < first
    assert (Path(out_dir) / "model.pt").exists()
    assert (Path(out_dir) / "tokenizer.json").exists()
    # the saved model + tokenizer must reload together
    loaded_model, loaded_tok = load_model(out_dir, device="cpu")
    assert loaded_tok.vocab_size == tok.vocab_size
    assert loaded_model.n_params == model.n_params


def test_learning_rate_schedule():
    cfg = TrainConfig(learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=10, max_steps=100)
    trainer = Trainer(_tiny_model(), TokenDataset(list(range(100))), cfg, verbose=False)
    assert trainer.get_lr(0) < trainer.get_lr(10)
    assert abs(trainer.get_lr(10) - 1e-3) < 1e-9
    assert abs(trainer.get_lr(100) - 1e-4) < 1e-9
    assert trainer.get_lr(50) < 1e-3 and trainer.get_lr(50) > 1e-4


# --------------------------------------------------------------------------- chat
def test_chat_prompt_format():
    history = [("User", "hi"), ("Assistant", "hello there")]
    prompt = format_chat(history)
    assert prompt.endswith("\nAssistant:")
    assert "User: hi\nAssistant: hello there" in prompt
    assert "Orbit" in prompt and prompt.startswith("The following is a conversation")


# --------------------------------------------------------------------------- runner
def main() -> int:
    import inspect
    import traceback

    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    failures = 0
    for name, fn in tests:
        kwargs = {}
        if "tmpdir" in inspect.signature(fn).parameters:
            kwargs["tmpdir"] = tempfile.mkdtemp()
        try:
            fn(**kwargs)
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
