"""Sampling, text generation and the interactive chat loop."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

from orbit_gpt.model import GPT
from orbit_gpt.tokenizer import Tokenizer, load_tokenizer

DEFAULT_PREAMBLE = (
    "The following is a conversation between a human (User) and a helpful, "
    "harmless, and honest AI assistant named Orbit.\n"
    "Orbit is a small language model that runs locally on a personal computer. "
    "Orbit answers clearly and concisely, and admits when it does not know something."
)

# The model keeps talking after its answer; these strings mark the turn boundary.
DEFAULT_STOP_STRINGS = ("\nUser:", "\nUser :", "\nSystem:")


def load_model(checkpoint: str, device: Optional[str] = None) -> Tuple[GPT, Tokenizer]:
    """Load a ``(model, tokenizer)`` pair from a checkpoint path or directory."""
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        ckpt_file, tok_dir = path / "model.pt", path
    else:
        ckpt_file, tok_dir = path, path.parent
    if not ckpt_file.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_file}")
    if not (tok_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"no tokenizer.json next to {ckpt_file}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT.from_checkpoint(str(ckpt_file), device=device)
    model.eval()
    return model, load_tokenizer(str(tok_dir))


def format_chat(
    history: Sequence[Tuple[str, str]], preamble: str = DEFAULT_PREAMBLE
) -> str:
    """Turn ``[("User", "hi"), ("Assistant", "hello")]`` into a prompt string.

    The layout matches the training corpus exactly::

        <preamble>

        User: hi
        Assistant: hello

        User: next question
        Assistant:
    """
    lines: List[str] = []
    for i, (role, text) in enumerate(history):
        if i > 0 and role == "User":
            lines.append("")  # blank line between turns
        lines.append(f"{role}: {text}")
    body = "\n".join(lines)
    return f"{preamble}\n\n{body}\nAssistant:" if body else f"{preamble}\n\nAssistant:"


def generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: float = 0.95,
    stop_strings: Sequence[str] = (),
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    stream: bool = False,
) -> str:
    """Generate a completion for ``prompt`` and return only the new text."""
    device = device or next(model.parameters()).device
    stop_ids = list(stop_strings)
    produced: List[int] = []
    for new_id in model.stream(
        tokenizer.encode(prompt),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_token_ids=[tokenizer.eot_id],
        seed=seed,
        device=device,
    ):
        produced.append(new_id)
        if stream:
            sys.stdout.write(tokenizer.decode([new_id]))
            sys.stdout.flush()
        if stop_ids:
            text = tokenizer.decode(produced)
            for s in stop_ids:
                if s in text:
                    text = text.split(s)[0]
                    if stream:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    return text
    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return tokenizer.decode(produced)


def chat(
    model: GPT,
    tokenizer: Tokenizer,
    device: Optional[torch.device] = None,
    preamble: str = DEFAULT_PREAMBLE,
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: float = 0.95,
    seed: Optional[int] = None,
    max_turns: int = 8,
    stop_strings: Sequence[str] = DEFAULT_STOP_STRINGS,
) -> None:
    """A tiny REPL: type a message, get an answer, repeat."""
    device = device or next(model.parameters()).device
    history: List[Tuple[str, str]] = []
    print(
        "\nOrbit is ready. Type a message and press Enter. "
        "Commands: /reset, /temp 0.8, /tokens 200, /quit\n"
    )
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # In Colab, `!python script.py` runs in a subprocess whose stdin is
            # not connected to the notebook, so input() hits EOF immediately.
            print(
                "\nBye! (No input available here. In Colab run the script with "
                "IPython's magic instead:  %run orbit_gpt_colab.py --no-train)"
            )
            return
        if not user:
            continue
        if user.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye!")
            return
        if user.lower() == "/reset":
            history.clear()
            print("(conversation cleared)")
            continue
        if user.lower().startswith("/temp "):
            try:
                temperature = float(user.split(None, 1)[1])
            except ValueError:
                pass
            print(f"(temperature = {temperature})")
            continue
        if user.lower().startswith("/tokens "):
            try:
                max_new_tokens = max(1, int(user.split(None, 1)[1]))
            except ValueError:
                pass
            print(f"(max_new_tokens = {max_new_tokens})")
            continue

        history.append(("User", user))
        prompt = format_chat(history, preamble)
        print("Orbit: ", end="", flush=True)
        answer = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_strings=stop_strings,
            seed=seed,
            device=device,
            stream=True,
        )
        answer = answer.strip()
        history.append(("Assistant", answer))
        # keep the prompt inside the context window
        history = history[-max_turns:]
