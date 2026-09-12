#!/usr/bin/env python3
"""Generate text with a trained OrbitGPT checkpoint.

    python generate.py --checkpoint out/orbit --prompt "Once upon a time"
    python generate.py --checkpoint out/orbit --chat              # interactive
    python generate.py -c out/orbit -p "ROMEO:" -n 4 -t 1.0       # 4 samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from orbit_gpt.generate import chat, generate, load_model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Sample from a trained OrbitGPT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", "-c", default="out/orbit")
    p.add_argument("--prompt", "-p", default="User: Hello!\nAssistant:")
    p.add_argument("--file", help="read the prompt from a text file instead")
    p.add_argument("--num-samples", "-n", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--temperature", "-t", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--chat", action="store_true", help="start an interactive chat")
    p.add_argument("--search", dest="search", action="store_true", default=None,
                   help="enable web search in the chat (needs the ddgs package)")
    p.add_argument("--no-search", dest="search", action="store_false",
                   help="disable web search in the chat")
    p.add_argument("--stop", nargs="*", default=["\nUser:"],
                   help="stop generation when one of these strings appears")
    args = p.parse_args(argv)

    model, tokenizer = load_model(args.checkpoint, device=args.device)
    device = next(model.parameters()).device

    if args.chat:
        chat(
            model,
            tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            use_web_search=args.search,
        )
        return 0

    prompt = Path(args.file).read_text() if args.file else args.prompt
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n=== sample {i+1}/{args.num_samples} ===")
        text = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_strings=args.stop,
            seed=None if args.seed is None else args.seed + i,
            device=device,
            stream=not args.no_stream,
        )
        if args.no_stream:
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
