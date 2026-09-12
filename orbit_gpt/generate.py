"""Sampling, text generation and the interactive chat loop."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

from orbit_gpt import search as websearch
from orbit_gpt.model import GPT
from orbit_gpt.skills import answer_arithmetic
from orbit_gpt.tokenizer import Tokenizer, load_tokenizer

# A persona header prepended to every chat prompt.  It is empty by default on
# purpose: a preamble that only ever appears at the *top* of the training text
# teaches a small model "this is the start of the document", and it then answers
# every question with the first exchange it memorised (measured: 10/10 correct
# answers without a preamble vs ~1/10 with one).  If you want a persona, put the
# same text at the top of *every* training document and pass it here.
DEFAULT_PREAMBLE = ""

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

        User: hi
        Assistant: hello

        User: next question
        Assistant:

    With a non-empty ``preamble`` it is prepended, separated by a blank line.
    """
    lines: List[str] = []
    for i, (role, text) in enumerate(history):
        if i > 0 and role == "User":
            lines.append("")  # blank line between turns
        lines.append(f"{role}: {text}")
    body = "\n".join(lines)
    if not body:
        return f"{preamble}\n\nAssistant:" if preamble else "Assistant:"
    return f"{preamble}\n\n{body}\nAssistant:" if preamble else f"{body}\nAssistant:"


def _earliest_stop(text: str, stops: Sequence[str]) -> int:
    """Index of the first stop string in ``text``, or -1."""
    best = -1
    for s in stops:
        i = text.find(s)
        if i != -1 and (best == -1 or i < best):
            best = i
    return best


def _hold_back(text: str, stops: Sequence[str]) -> int:
    """How many characters at the end of ``text`` to keep unprinted.

    Anything that could still *become* a stop string (or a half-decoded UTF-8
    character) is held back so streaming never shows text we are about to cut.
    """
    hold = 4  # longest UTF-8 sequence: never print a partial character
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), 0, -1):
            if text.endswith(s[:k]):
                hold = max(hold, k)
                break
    return min(hold, len(text))


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
    repetition_penalty: float = 1.15,
) -> str:
    """Generate a completion for ``prompt`` and return only the new text."""
    device = device or next(model.parameters()).device
    stops = list(stop_strings)
    produced: List[int] = []
    printed = 0

    for new_id in model.stream(
        tokenizer.encode(prompt),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_token_ids=[tokenizer.eot_id],
        seed=seed,
        device=device,
        repetition_penalty=repetition_penalty,
    ):
        produced.append(new_id)
        text = tokenizer.decode(produced)
        cut = _earliest_stop(text, stops) if stops else -1
        if cut >= 0:
            if stream:
                sys.stdout.write(text[printed:cut] + "\n")
                sys.stdout.flush()
            return text[:cut]
        if stream:
            upto = len(text) - _hold_back(text, stops)
            if upto > printed:
                sys.stdout.write(text[printed:upto])
                sys.stdout.flush()
                printed = upto

    text = tokenizer.decode(produced)
    if stream:
        sys.stdout.write(text[printed:] + "\n")
        sys.stdout.flush()
    return text


def _trim_history(
    history: List[Tuple[str, str]],
    tokenizer: Tokenizer,
    block_size: int,
    preamble: str,
    keep_fraction: float = 0.6,
) -> List[Tuple[str, str]]:
    """Drop the oldest exchanges so the prompt leaves room for a reply.

    The context window has to hold the prompt *and* the answer, so we keep the
    prompt to roughly half of it and forget the beginning of the conversation
    when it no longer fits.
    """
    budget = max(16, int(block_size * keep_fraction))
    # history ends with the question we are about to answer, so we only ever
    # drop whole exchanges from the front and never that last message.
    while len(history) >= 3:
        if len(tokenizer.encode(format_chat(history, preamble))) <= budget:
            break
        history = history[2:]  # forget the oldest User+Assistant exchange
    return history


def _web_prompt(
    question: str,
    results,
    model: GPT,
    tokenizer: Tokenizer,
    max_new_tokens: int,
):
    """Build a search-augmented prompt that always leaves room for the answer.

    The web block is capped in characters and then checked with the real
    tokenizer; if it still does not fit, it is shrunk, and if even the bare
    question does not fit we return ``None`` and let the normal (trimmed)
    prompt path deal with it.
    """
    block_size = model.config.block_size
    reserve = min(max_new_tokens + 24, max(32, block_size // 2))
    budget = max(150, int((block_size - reserve) * 3.0))
    prompt = websearch.augment_prompt(question, results, max_chars=budget)
    while len(tokenizer.encode(prompt)) > block_size - reserve and budget > 150:
        budget = int(budget * 0.6)
        prompt = websearch.augment_prompt(question, results, max_chars=budget)
    if len(tokenizer.encode(prompt)) > block_size - reserve:
        return None
    return prompt


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
    repetition_penalty: float = 1.15,
    use_skills: bool = True,
    use_web_search: Optional[bool] = None,
    web_results: int = 5,
) -> None:
    """A tiny REPL: type a message, get an answer, repeat.

    ``use_web_search=None`` follows :data:`orbit_gpt.search.USE_WEB_SEARCH`;
    it can be flipped at run time with ``/search on`` / ``/search off``.
    """
    device = device or next(model.parameters()).device
    if use_web_search is None:
        use_web_search = websearch.USE_WEB_SEARCH
    history: List[Tuple[str, str]] = []
    search_state = (
        "on" if use_web_search and websearch.available()
        else "on, but no backend (pip install ddgs)" if use_web_search
        else "off"
    )
    print(
        "\nOrbit is ready. Type a message and press Enter. "
        "Commands: /reset, /temp 0.8, /tokens 200, /rep 1.2, /search off, /quit\n"
        f"Web search: {search_state}\n"
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
        if user.lower().startswith("/rep "):
            try:
                repetition_penalty = float(user.split(None, 1)[1])
            except ValueError:
                pass
            print(f"(repetition penalty = {repetition_penalty})")
            continue
        if user.lower().startswith("/tokens "):
            try:
                max_new_tokens = max(1, int(user.split(None, 1)[1]))
            except ValueError:
                pass
            print(f"(max_new_tokens = {max_new_tokens})")
            continue
        if user.lower().startswith("/search"):
            parts = user.split(None, 1)
            arg = parts[1].strip().lower() if len(parts) > 1 else "status"
            if arg in ("on", "true", "1", "yes", "enable"):
                use_web_search = True
            elif arg in ("off", "false", "0", "no", "disable"):
                use_web_search = False
            print(
                f"(web search = {'on' if use_web_search else 'off'}"
                + ("" if not use_web_search or websearch.available()
                   else " - no backend installed, run: pip install ddgs")
                + ")"
            )
            continue

        history.append(("User", user))
        if use_skills:
            # a 1M-parameter model cannot do multi-digit maths, so answer
            # calculations with Python instead of letting it invent a number
            exact = answer_arithmetic(user)
            if exact is not None:
                print(f"Orbit: {exact}")
                history.append(("Assistant", exact))
                history = history[-max_turns:]
                continue
        # ---- optional web search ----------------------------------------
        # Only for questions that want fresh information; failures are
        # reported and the model falls back to answering on its own.
        prompt = None
        sources = None
        if use_web_search and websearch.needs_search(user):
            print("[Web search enabled]")
            query = websearch.build_query(user)
            print(f"Searching for: {query}")
            results = websearch.search_web(query, max_results=web_results)
            if results:
                print(f"Found {len(results)} relevant results.")
                sources = results
                prompt = _web_prompt(
                    user, results, model, tokenizer, max_new_tokens
                )
            else:
                print(
                    f"(web search unavailable: {websearch.last_error()} - "
                    "answering without it)"
                )
            if prompt:
                print("Generating answer...")

        if prompt is None:
            # keep the prompt (plus room for the answer) inside the context window
            history = _trim_history(history, tokenizer, model.config.block_size, preamble)
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
            repetition_penalty=repetition_penalty,
        )
        answer = answer.strip()
        if answer and sources:
            # the model is small: always show where the claim came from, so a
            # paraphrase that drifted can be checked against the snippets
            print("\nSources:")
            for i, r in enumerate(sources, start=1):
                print(f"  [{i}] {r.title}" + (f" - {r.url}" if r.url else ""))
        if not answer:
            print(
                "(no room left in the context window - try /reset, or train with a "
                "larger --block-size)"
            )
            history.pop()  # forget the question we could not answer
        else:
            history.append(("Assistant", answer))
        history = history[-max_turns:]
