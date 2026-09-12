#!/usr/bin/env python3
"""Run your trained OrbitGPT locally in a desktop chat window.

This is the file you keep.  It needs **no PyTorch** - the model is an ONNX
graph and the only extra package is ``onnxruntime``::

    pip install onnxruntime
    python orbit_app.py --model exports/orbit-int8.onnx

It opens a small window (Tkinter is part of Python, so there is nothing else
to install).  ``--no-gui`` gives you the same engine in the terminal instead.

Export the model first, from inside the repo::

    python -m orbit_gpt.export checkpoints/orbit --out exports/orbit.onnx

``exports/`` then holds ``orbit.onnx`` (fp32), ``orbit-int8.onnx`` (4x
smaller, the app prefers it), ``tokenizer.json`` and ``config.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

# the repo lives next to this file; add it so ``orbit_gpt.tokenizer`` imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DEFAULT_STOP_STRINGS = ("\nUser:", "\nUser :", "\nSystem:", "\nUser", "\nSystem")

try:  # exact answers (arithmetic, unit conversions) - pure Python, no torch
    from orbit_gpt.skills import answer_arithmetic as _answer_arithmetic
except Exception:  # pragma: no cover - skills are optional
    _answer_arithmetic = None


def exact_answer(user: str) -> Optional[str]:
    """Answer sums and conversions with Python instead of guessing."""
    if _answer_arithmetic is None:
        return None
    try:
        return _answer_arithmetic(user)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# sampling (a numpy mirror of GPT._sample so the app matches training)
# ---------------------------------------------------------------------------
def sample(
    logits: np.ndarray,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
    rng: np.random.Generator,
    seen: Optional[Sequence[int]] = None,
    repetition_penalty: float = 1.0,
) -> int:
    logits = logits.astype(np.float64)
    if seen and repetition_penalty != 1.0:
        for tid in set(int(t) for t in seen):
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty
            else:
                logits[tid] *= repetition_penalty
    logits /= max(temperature, 1e-6)

    if top_k is not None and 0 < top_k < logits.shape[-1]:
        cut = np.sort(logits)[-top_k]
        logits = np.where(logits < cut, -np.inf, logits)
    if 0.0 < top_p < 1.0:
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        probs = _softmax(sorted_logits)
        cum = np.cumsum(probs)
        remove = (cum - probs) > top_p
        sorted_logits[remove] = -np.inf
        logits = np.full_like(logits, -np.inf)
        logits[order] = sorted_logits

    probs = _softmax(logits)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def _softmax(x: np.ndarray) -> np.ndarray:
    top = x.max()
    if not np.isfinite(top):          # everything was filtered out
        return np.ones_like(x) / x.size
    x = x - top
    e = np.exp(x)
    return e / e.sum()


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------
class OnnxGPT:
    """Runs the exported decoder step with onnxruntime.

    The graph takes one token plus the KV cache and hands back logits plus the
    updated cache, so generation costs one small forward pass per token.
    """

    def __init__(self, model_path: str | Path, tokenizer_path: Optional[Path] = None):
        import onnxruntime as ort

        model_path = Path(model_path)
        folder = model_path.parent
        if tokenizer_path is None:
            tokenizer_path = folder / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise SystemExit(
                f"no tokenizer at {tokenizer_path} - export the model first:\n"
                "  python -m orbit_gpt.export checkpoints/orbit "
                "--out exports/orbit.onnx"
            )

        from orbit_gpt.tokenizer import load_tokenizer

        self.tokenizer = load_tokenizer(str(tokenizer_path.parent))
        self.model_path = str(model_path)
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        cfg_path = folder / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        arch = cfg.get("model_config", cfg)
        self.block_size = int(arch.get("block_size", 256))
        self.n_layer = int(arch.get("n_layer", 6))
        self.n_head = int(arch.get("n_head", 8))
        self.n_embd = int(arch.get("n_embd", 256))
        self.head_dim = self.n_embd // self.n_head

        # stop on the end-of-text / pad tokens the model was trained with
        self._stop_ids = {int(getattr(self.tokenizer, "eot_id", 1)),
                          int(getattr(self.tokenizer, "pad_id", 0))}

        self._in_names = [i.name for i in self.session.get_inputs()]
        self._out_names = [o.name for o in self.session.get_outputs()]
        self._cache: dict = {}
        self._pos = 0

    # -- cache handling ----------------------------------------------------
    def reset(self) -> None:
        self._cache = {
            name: np.zeros((1, self.n_head, 0, self.head_dim), dtype=np.float32)
            for name in self._in_names
            if name.startswith("past_")
        }
        self._pos = 0

    def _step(self, token_id: int) -> np.ndarray:
        feeds = dict(self._cache)
        feeds["token_id"] = np.array([[token_id]], dtype=np.int64)
        feeds["pos"] = np.array([self._pos], dtype=np.int64)
        outputs = self.session.run(None, feeds)
        for name, value in zip(self._out_names, outputs):
            if name.startswith("present_"):
                self._cache["past_" + name.split("_", 1)[1]] = value
        self._pos += 1
        return outputs[0][0]  # logits for the newest position

    # -- generation --------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15,
        stop_strings: Sequence[str] = DEFAULT_STOP_STRINGS,
        seed: Optional[int] = None,
        cancel: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Yield the answer piece by piece (decode-safe, stream-friendly)."""
        rng = np.random.default_rng(seed)
        ids = self.tokenizer.encode(prompt)[-self.block_size:]
        room = max(0, self.block_size - len(ids))
        max_new_tokens = max(0, min(max_new_tokens, room))
        if max_new_tokens == 0:
            return

        self.reset()
        logits = None
        for token in ids:                     # prefill, one token per step
            logits = self._step(token)        # last one predicts the reply

        produced: List[int] = []
        printed = 0
        for _ in range(max_new_tokens):
            if cancel is not None and cancel.is_set():
                break
            next_id = sample(logits, temperature, top_k, top_p, rng,
                             seen=produced, repetition_penalty=repetition_penalty)
            if next_id in self._stop_ids:
                break
            produced.append(next_id)
            text = self.tokenizer.decode(produced)

            cut = _earliest_stop(text, stop_strings)
            if cut >= 0:                      # it started the next turn
                if cut > printed:
                    yield text[printed:cut]
                return
            # hold back anything that could still *become* a stop string
            upto = len(text) - _hold_back(text, stop_strings)
            if upto > printed:
                yield text[printed:upto]
                printed = upto
            logits = self._step(next_id)

        if produced:                          # flush, minus a half-written turn
            text = self.tokenizer.decode(produced)
            cut = _earliest_stop(text, stop_strings)
            if cut >= 0:
                text = text[:cut]
            text = _trim_partial(text, stop_strings)
            if len(text) > printed:
                yield text[printed:]


def _hold_back(text: str, stops: Sequence[str]) -> int:
    """Characters to keep unprinted: a partial stop string or a split UTF-8."""
    hold = 4
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), 0, -1):
            if text.endswith(s[:k]):
                hold = max(hold, k)
                break
    return min(hold, len(text))


def _trim_partial(text: str, stops: Sequence[str]) -> str:
    """Drop a half-written turn marker left over when the reply ends."""
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), 0, -1):
            if text.endswith(s[:k]):
                return text[: len(text) - k]
    return text


def _earliest_stop(text: str, stops: Sequence[str]) -> int:
    best = -1
    for s in stops:
        i = text.find(s)
        if i >= 0 and (best < 0 or i < best):
            best = i
    return best


# ---------------------------------------------------------------------------
# prompt building (mirrors orbit_gpt.generate.format_chat - kept local so the
# app never imports torch)
# ---------------------------------------------------------------------------
def format_chat(history: Sequence[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for i, (role, text) in enumerate(history):
        if i > 0 and role == "User":
            lines.append("")
        lines.append(f"{role}: {text}")
    body = "\n".join(lines)
    return f"{body}\nAssistant:" if body else "Assistant:"


def trim_history(history, tokenizer, block_size, max_tokens=160) -> list:
    """Drop the oldest turns until the prompt fits the context window."""
    keep = list(history)
    while keep:
        prompt = format_chat(keep)
        if len(tokenizer.encode(prompt)) + max_tokens <= block_size:
            break
        keep = keep[2:] if len(keep) > 1 else keep[1:]
    return keep


def best_model(folder: Path = Path("exports")) -> Optional[Path]:
    """Prefer the small int8 export, fall back to fp32."""
    if not folder.is_dir():
        return None
    onnx = sorted(folder.glob("*.onnx"))
    if not onnx:
        return None
    for path in onnx:
        if "int8" in path.stem:
            return path
    return onnx[0]


# ---------------------------------------------------------------------------
# terminal mode
# ---------------------------------------------------------------------------
def repl(engine: OnnxGPT, temperature: float, max_tokens: int) -> int:
    print("OrbitGPT - type a message, or 'quit'.")
    history: List[Tuple[str, str]] = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if user.lower() in ("quit", "exit", "q"):
            return 0
        if not user:
            continue
        history.append(("User", user))
        history[:] = trim_history(history, engine.tokenizer,
                                  engine.block_size, max_tokens)
        exact = exact_answer(user)
        if exact:
            print(f"Orbit: {exact}")
            history.append(("Assistant", exact))
            continue
        print("Orbit: ", end="", flush=True)
        answer = "".join(engine.generate(
            format_chat(history), max_new_tokens=max_tokens,
            temperature=temperature))
        print(answer.strip() or "(no answer)")
        history.append(("Assistant", answer.strip()))


# ---------------------------------------------------------------------------
# desktop mode
# ---------------------------------------------------------------------------
def gui(engine: OnnxGPT, temperature: float, max_tokens: int) -> int:
    import queue
    import tkinter as tk
    from tkinter import ttk, scrolledtext

    root = tk.Tk()
    root.title(f"OrbitGPT - {Path(engine.model_path).name}")
    root.geometry("760x560")

    log = scrolledtext.ScrolledText(root, wrap="word", font=("Segoe UI", 11),
                                    state="disabled")
    log.pack(fill="both", expand=True, padx=8, pady=(8, 4))

    bar = ttk.Frame(root)
    bar.pack(fill="x", padx=8, pady=(0, 8))
    entry = ttk.Entry(bar, font=("Segoe UI", 11))
    entry.pack(side="left", fill="x", expand=True)
    send = ttk.Button(bar, text="Send", width=8)
    send.pack(side="left", padx=(6, 0))
    stop_btn = ttk.Button(bar, text="Stop", width=8, state="disabled")
    stop_btn.pack(side="left", padx=(6, 0))

    history: List[Tuple[str, str]] = []
    updates: "queue.Queue[tuple]" = queue.Queue()
    cancel = threading.Event()
    busy = {"on": False}

    def say(role: str, text: str) -> None:
        log.configure(state="normal")
        log.insert("end", f"{role}: {text}\n")
        log.configure(state="disabled")
        log.see("end")

    def pump() -> None:
        try:
            while True:
                kind, text = updates.get_nowait()
                if kind == "token":
                    log.configure(state="normal")
                    log.insert("end", text)
                    log.configure(state="disabled")
                    log.see("end")
                elif kind == "done":
                    log.configure(state="normal")
                    log.insert("end", "\n")
                    log.configure(state="disabled")
                    log.see("end")
                    busy["on"] = False
                    send.configure(state="normal")
                    stop_btn.configure(state="disabled")
        except queue.Empty:
            pass
        root.after(60, pump)

    def worker(prompt: str) -> None:
        answer = ""
        started = time.time()
        try:
            for piece in engine.generate(
                prompt, max_new_tokens=max_tokens,
                temperature=temperature, cancel=cancel,
            ):
                if cancel.is_set():
                    break
                if not answer:
                    updates.put(("token", piece.lstrip()))
                else:
                    updates.put(("token", piece))
                answer += piece
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            updates.put(("token", f"\n[error: {exc}]\n"))
        answer = answer.strip()
        history.append(("Assistant", answer))
        if time.time() - started > 0.2:
            updates.put(("token", "\n(%.1fs)\n" % (time.time() - started)))
        updates.put(("done", ""))

    def on_send() -> None:
        user = entry.get().strip()
        if not user or busy["on"]:
            return
        entry.delete(0, "end")
        say("You", user)
        history.append(("User", user))
        history[:] = trim_history(history, engine.tokenizer,
                                  engine.block_size, max_tokens)

        exact = exact_answer(user)
        if exact:                      # maths is exact, skip the model
            say("Orbit", exact)
            history.append(("Assistant", exact))
            return

        busy["on"] = True
        cancel.clear()
        send.configure(state="disabled")
        stop_btn.configure(state="normal")
        say("Orbit", "")
        log.configure(state="normal")
        log.insert("end", "\n")       # keep the streaming text on its own line
        log.configure(state="disabled")
        threading.Thread(target=worker, args=(format_chat(history),),
                         daemon=True).start()

    def on_stop() -> None:
        cancel.set()

    send.configure(command=on_send)
    stop_btn.configure(command=on_stop)
    entry.bind("<Return>", lambda _e: on_send())
    entry.focus_set()
    root.after(60, pump)
    say("Orbit", "Ask me anything.  (Enter sends, Stop ends a reply.)")
    root.mainloop()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run OrbitGPT locally (ONNX).")
    p.add_argument("--model", default=None,
                   help="path to the .onnx export (default: exports/*int8*.onnx)")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--no-gui", action="store_true", help="terminal REPL")
    args = p.parse_args(argv)

    model = Path(args.model) if args.model else best_model()
    if model is None or not model.is_file():
        raise SystemExit(
            "no exported model found.  Export one with:\n"
            "  python -m orbit_gpt.export checkpoints/orbit "
            "--out exports/orbit.onnx"
        )
    tokenizer = Path(args.tokenizer) if args.tokenizer else None
    engine = OnnxGPT(model, tokenizer)
    print(f"loaded {model} ({model.stat().st_size/1e6:.1f} MB)")

    if args.no_gui:
        return repl(engine, args.temperature, args.max_tokens)
    try:
        return gui(engine, args.temperature, args.max_tokens)
    except ImportError as exc:
        print(f"no Tkinter on this Python ({exc}) - using the terminal instead")
        return repl(engine, args.temperature, args.max_tokens)


if __name__ == "__main__":
    raise SystemExit(main())
