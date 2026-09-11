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
from orbit_gpt.generate import (  # noqa: E402
    DEFAULT_PREAMBLE,
    _earliest_stop,
    _hold_back,
    _trim_history,
    format_chat,
    load_model,
)
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
def test_stop_string_helpers():
    assert _earliest_stop("hi\nUser: yo", ["\nUser:"]) == 2
    assert _earliest_stop("nothing here", ["\nUser:"]) == -1
    # a partial stop string at the end must be held back while streaming
    assert _hold_back("hello\nUs", ["\nUser:"]) >= 3
    # ...but we still keep a margin so a multi-byte character is never split
    assert _hold_back("hello", ["\nUser:"]) == 4


def test_trim_history_keeps_prompt_inside_context():
    tok = BPETokenizer().train(TINY_CORPUS, vocab_size=300)
    history = [
        (role, f"message number {i} with a bit more text so it costs tokens")
        for i in range(10)
        for role in ("User", "Assistant")
    ]
    history.append(("User", "and finally the question we are answering now"))
    trimmed = _trim_history(history, tok, 128, DEFAULT_PREAMBLE)
    assert len(trimmed) < len(history)
    # the most recent question always survives, and we shrink until the prompt
    # fits the budget (or until only that question is left)
    assert trimmed[-1] == history[-1] and trimmed[-1][0] == "User"
    prompt_tokens = len(tok.encode(format_chat(trimmed, DEFAULT_PREAMBLE)))
    assert prompt_tokens <= int(128 * 0.6) or len(trimmed) == 1


def test_corpus_has_no_preamble():
    """The prompt must look exactly like the training text - pure exchanges.

    A persona header that only appears at the top of the corpus teaches the
    model "document start", which makes it ignore the actual question.
    """
    corpus = (
        Path(__file__).resolve().parents[1] / "orbit_gpt" / "corpora" / "orbit_assistant.txt"
    ).read_text(encoding="utf-8")
    assert corpus.lstrip().startswith("User:")
    assert DEFAULT_PREAMBLE == ""


def test_chat_repl_never_runs_out_of_context():
    """A long conversation must be trimmed instead of overflowing the window.

    With a 32-token context an un-trimmed history runs out of room after two
    turns and the REPL silently stops answering.
    """
    import builtins
    import contextlib
    import io

    from orbit_gpt.generate import chat

    tok = BPETokenizer().train(TINY_CORPUS, vocab_size=300)
    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=48, n_layer=2, n_head=4, n_embd=64,
        dropout=0.0,
    )
    model = GPT(cfg).eval()
    # questions made of words the tokenizer has seen, so the prompt is short
    script = iter(["What is 2 + 2?", "What is the capital of France?", "What is 2 + 2?", "quit"])
    real_input, buf = builtins.input, io.StringIO()
    builtins.input = lambda prompt="": next(script)
    try:
        with contextlib.redirect_stdout(buf):
            # stop_strings=() isolates the failure we care about: an untrained
            # model may emit a turn marker at any time, but running out of
            # context is a bug in the history trimming.
            chat(model, tok, max_new_tokens=8, stop_strings=())
    finally:
        builtins.input = real_input
    out = buf.getvalue()
    assert "no room left" not in out
    assert out.count("Orbit:") == 3


def test_chat_prompt_format():
    # a chat prompt always ends with the question we want answered
    history = [("User", "hi"), ("Assistant", "hello there"), ("User", "bye")]
    prompt = format_chat(history)
    assert prompt == "User: hi\nAssistant: hello there\n\nUser: bye\nAssistant:"
    assert format_chat([], "You are Orbit") == "You are Orbit\n\nAssistant:"
    assert format_chat([]) == "Assistant:"


# --------------------------------------------------------------- generated corpus
def test_generated_conversation_corpus_is_deterministic_and_varied():
    from orbit_gpt.corpora.conversation import build_conversation_corpus

    a = build_conversation_corpus()
    b = build_conversation_corpus()
    assert a == b, "the generated corpus must be reproducible"
    assert len(a) > 500_000, len(a)
    assert a.count("User:") > 20_000
    # the one line every canned corpus used to fall back on must be gone
    assert "How can I help you today?" not in a


def test_generated_corpus_answers_are_correct():
    """Every arithmetic answer in the corpus must actually be right.

    The answers are computed, not written by hand - this is the guard that
    keeps the model from being trained on wrong maths.
    """
    from orbit_gpt.corpora.conversation import build_conversation_corpus

    text = build_conversation_corpus()
    checked = 0
    for line in text.splitlines():
        if " = " not in line:
            continue
        head, tail = line.split("Assistant:", 1)[-1].rsplit(" = ", 1)
        got = tail.rstrip(".").strip()
        if not got.lstrip("-").isdigit():
            continue
        got = int(got)
        # "81 - 39 = 42"
        for op, combine in (("+", int.__add__), ("\u00d7", int.__mul__),
                            ("\u2212", int.__sub__), ("-", int.__sub__)):
            if op in head:
                left, right = head.split(op, 1)
                if left.strip().isdigit() and right.strip().isdigit():
                    assert got == combine(int(left), int(right)), line
                    checked += 1
                break
        else:
            # "10% of 200 = 20" and "6 squared = 36"
            if "% of " in head:
                percent, base = head.split("% of ", 1)
                if percent.strip().isdigit() and base.strip().isdigit():
                    assert got == round(int(base) * int(percent) / 100), line
                    checked += 1
            elif head.strip().endswith("squared"):
                base = head.strip()[: -len("squared")].strip()
                if base.isdigit():
                    assert got == int(base) ** 2, line
                    checked += 1
    assert checked > 500, checked

def test_generated_corpus_leaves_holdout_values_for_the_tests():
    """The corpus must NOT contain the numbers/phrasings the eval relies on."""
    from orbit_gpt.corpora.conversation import (
        TEST_ASKS,
        TEST_HOW_ASKS,
        build_conversation_corpus,
    )

    text = build_conversation_corpus()
    for phrasing in TEST_ASKS + TEST_HOW_ASKS:
        assert phrasing.format(t="anything") not in text
    # 3-digit addition is never trained on, so it is a fair generalisation test
    assert "137 + 268" not in text
    assert "500 - 137" not in text


def test_conversation_corpus_loads_through_the_data_module():
    from orbit_gpt.data import load_corpus

    text = load_corpus("conversation")
    assert text.count("User:") > 20_000
    assert text == load_corpus("conversation")


def test_repetition_penalty_stops_the_same_token_looping():
    """A tiny model otherwise repeats one word for the whole reply."""
    from orbit_gpt import GPT
    from orbit_gpt.config import GPTConfig

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=80, block_size=32, n_layer=2, n_head=2, n_embd=32))
    assert model.n_params > 0
    logits = torch.randn(1, 80)
    logits[0, 7] = 4.0  # one token dominates -> without a penalty it loops

    def repeats(penalty, n=60):
        out = []
        for i in range(n):  # same draws for every penalty: only the penalty moves
            generator = torch.Generator().manual_seed(i)
            out.append(int(GPT._sample(logits.clone(), 1.0, None, 1.0, generator,
                                       seen=out, repetition_penalty=penalty)))
        return out.count(7)

    assert repeats(1.0) > repeats(1.5) > repeats(2.5)

def test_colab_build_can_reuse_a_saved_checkpoint(tmpdir=None):
    """Train once, chat many times: a saved model must be detected and loaded."""
    import tempfile

    mod = _load_colab_module()
    with tempfile.TemporaryDirectory() as d:
        assert mod.find_checkpoint(d) is None
        (Path(d) / "model.pt").write_bytes(b"x")       # a model with no tokenizer
        assert mod.find_checkpoint(d) is None
        (Path(d) / "tokenizer.json").write_text("{}")  # now it is complete
        assert mod.find_checkpoint(d) == Path(d)
    assert mod.CONFIG["save_to_drive"] is True
    assert mod.CONFIG["retrain"] is False


def test_colab_footer_is_the_one_shipped_in_the_built_file():
    """tools/colab_footer.py is the source of the single-file build."""
    footer = (Path(__file__).resolve().parent.parent / "tools" / "colab_footer.py").read_text(encoding="utf-8")
    built = _colab_path().read_text(encoding="utf-8")
    for needle in ("def mount_drive(", "def find_checkpoint(", "retrain", "save_to_drive"):
        assert needle in footer
        assert needle in built


# --------------------------------------------------------------------------- runner
def _colab_path():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent / "colab" / "orbit_gpt_colab.py"


def _load_colab_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("orbit_colab_built", _colab_path())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    import inspect
    import traceback

    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    # every other tests/test_*.py module is part of the same suite
    import tests.test_skills as skills_module  # noqa: F401  (imported for its tests)

    for extra in (skills_module,):
        tests += [
            (f"{extra.__name__.split('.')[-1]}.{name}", fn)
            for name, fn in sorted(vars(extra).items())
            if name.startswith("test_") and inspect.isfunction(fn)
        ]
    tests.sort()
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
