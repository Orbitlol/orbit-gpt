"""Export a trained OrbitGPT checkpoint to ONNX for a small, standalone app.

The training model keeps its KV cache inside the module (``GPT._kv_cache``),
which an exported graph cannot express.  :class:`DecoderStep` is the same
network written as a pure function: you hand it the caches in, it hands the
updated caches back out.  One graph therefore covers both the prefill and the
decode step - feed the prompt one token at a time and keep the caches.

Files produced by :func:`export_onnx`::

    orbit.onnx          fp32 decoder step
    tokenizer.json      the BPE vocabulary it was trained with
    config.json         architecture + generation defaults

and, if onnxruntime is installed, :func:`quantize_onnx` adds::

    orbit-int8.onnx     same graph, 8-bit weights (~4x smaller)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .model import GPT


class DecoderStep(nn.Module):
    """One cached decode step, with the cache passed in and out explicitly.

    Inputs:  ``token_id`` ``[1, 1]``, ``pos`` ``[1]``,
             ``past_k_0..L-1`` / ``past_v_0..L-1`` ``[1, H, P, D]``
    Outputs: ``logits`` ``[1, V]``, ``present_k_0..L-1`` / ``present_v_0..L-1``
    """

    def __init__(self, model: GPT) -> None:
        super().__init__()
        self.model = model
        self.n_layer = model.config.n_layer

    def forward(
        self, token_id: torch.Tensor, pos: torch.Tensor, *past: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        gpt = self.model
        # positions are built with tensor ops so the sequence length stays
        # dynamic under every exporter
        positions = torch.cumsum(torch.ones_like(token_id[0]), dim=0) - 1 + pos
        x = gpt.transformer.wte(token_id) + gpt.transformer.wpe(positions).unsqueeze(0)

        presents: List[torch.Tensor] = []
        for i, block in enumerate(gpt.transformer.h):
            cache = (past[2 * i], past[2 * i + 1])
            x, new_cache = block(x, kv_cache=cache)
            presents.append(new_cache[0])
            presents.append(new_cache[1])

        x = gpt.transformer.ln_f(x)
        logits = gpt.lm_head(x[:, -1, :])  # only the newest position matters
        return (logits,) + tuple(presents)


def export_onnx(
    checkpoint_dir: str | Path,
    out_path: str | Path = "orbit.onnx",
    opset: int = 18,
    quantize: bool = True,
    verbose: bool = True,
) -> Path:
    """Export ``checkpoint_dir`` to an ONNX decoder step.

    Also copies ``tokenizer.json`` and writes a ``config.json`` next to the
    model so the app only needs one folder.  Returns the path of the fp32
    model (the int8 sibling, if any, sits beside it).
    """
    checkpoint_dir = Path(checkpoint_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load(checkpoint_dir)
    model.eval()
    # Weight tying (the embedding *is* the un-embedding) makes one initializer
    # serve two different shapes, which is what breaks the ONNX quantiser's
    # shape inference.  A clone with identical values keeps the maths the same.
    with torch.no_grad():
        model.transformer.wte.weight = nn.Parameter(
            model.lm_head.weight.detach().clone()
        )
    decoder = DecoderStep(model).eval()

    cfg = model.config
    head_dim = cfg.n_embd // cfg.n_head
    # sample shapes: batch 1, one new token, one cached position
    b, t, p = 1, 1, 1
    sample = (
        torch.zeros((b, t), dtype=torch.long),
        torch.zeros((1,), dtype=torch.long),
        *[torch.zeros((b, cfg.n_head, p, head_dim)) for _ in range(2 * cfg.n_layer)],
    )
    names = ["token_id", "pos"] + [
        f"{kind}{i}"
        for i in range(cfg.n_layer)
        for kind in ("past_k_", "past_v_")
    ]
    out_names = ["logits"] + [
        f"{kind}{i}"
        for i in range(cfg.n_layer)
        for kind in ("present_k_", "present_v_")
    ]
    # one shared cache length for every layer: at runtime they always match
    seq_dim = torch.export.Dim("seq", min=1)
    past_dim = torch.export.Dim("past", min=0)
    # the *past varargs are one nested structure, matching forward()'s `*past`
    dynamic_shapes = (
        {1: seq_dim},   # token_id [1, T]
        None,           # pos [1] - always one scalar
        tuple({2: past_dim} for _ in range(2 * cfg.n_layer)),
    )

    # The training code picks a fast attention kernel with a data-dependent
    # check (`is_causal = T == T_full`) that an exported graph cannot keep.
    # The explicit masked path is branch-free and gives identical numbers, so
    # we export that one.
    flashed = [bool(b.attn.flash) for b in decoder.model.transformer.h]
    for block in decoder.model.transformer.h:
        attn = block.attn
        if not hasattr(attn, "bias"):     # the mask the explicit path needs
            attn.register_buffer(
                "bias",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                    1, 1, cfg.block_size, cfg.block_size
                ),
                persistent=False,
            )
        attn.flash = False
    try:
        with torch.no_grad():
            torch.onnx.export(
                decoder,
                sample,
                str(out_path),
                input_names=names,
                output_names=out_names,
                dynamic_shapes=dynamic_shapes,
                opset_version=opset,
                dynamo=True,
                external_data=False,   # one portable file, not .onnx + .data
            )
    finally:
        for block, value in zip(decoder.model.transformer.h, flashed):
            block.attn.flash = value

    # the app needs the vocabulary and the architecture next to the weights
    source_dir = checkpoint_dir if checkpoint_dir.is_dir() else checkpoint_dir.parent
    tok_src, tok_dst = source_dir / "tokenizer.json", out_path.parent / "tokenizer.json"
    if tok_src.is_file() and tok_src.resolve() != tok_dst.resolve():
        shutil.copy2(tok_src, tok_dst)
    meta: Dict[str, Any] = {
        "model_config": cfg.to_dict(),
        "n_params": model.n_params,
        "export": "orbit-gpt decoder step (kv-cache in, kv-cache out)",
        "inputs": names,
        "outputs": out_names,
        "opset": opset,
    }
    (out_path.parent / "config.json").write_text(json.dumps(meta, indent=2))

    if verbose:
        print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        print(f"      {out_path.parent / 'tokenizer.json'}")
        print(f"      {out_path.parent / 'config.json'}")

    if quantize:
        try:
            quantize_onnx(out_path, verbose=verbose)
        except Exception as exc:  # pragma: no cover - depends on onnxruntime
            print(f"  ! skipped int8 quantisation ({exc})")
    return out_path


def quantize_onnx(path: str | Path, out_path: Optional[Path] = None,
                  verbose: bool = True) -> Path:
    """8-bit weight quantisation: ~3x smaller, runs on the CPU, same replies.

    onnxruntime needs its shape/graph pre-processing pass first - without it
    the quantiser trips over the tied embedding matrix.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.preprocess import quant_pre_process

    path = Path(path)
    out_path = Path(out_path) if out_path else path.with_name(
        path.stem + "-int8" + path.suffix
    )
    pre = path.with_name(path.stem + "-pre" + path.suffix)
    source = path
    try:
        quant_pre_process(str(path), str(pre))   # Gemm -> MatMul, shapes fixed
        source = pre
    except Exception as exc:  # pragma: no cover - older onnxruntime
        print(f"  ! pre-processing failed ({exc}), quantising directly")
    try:
        quantize_dynamic(
            model_input=str(source),
            model_output=str(out_path),
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=True,   # safer on CPUs without AVX512/VNNI
        )
    finally:
        if pre.is_file():
            pre.unlink()
    if verbose:
        before, after = path.stat().st_size, out_path.stat().st_size
        print(f"wrote {out_path} ({after / 1e6:.1f} MB, "
              f"{before / max(after, 1):.1f}x smaller)")
    return out_path


def _load(checkpoint_dir: Path) -> Tuple[GPT, Any]:
    """Load (model, tokenizer) from a checkpoint directory or file."""
    from .tokenizer import load_tokenizer

    path = Path(checkpoint_dir)
    if not path.is_file():
        candidate = path / "model.pt"
        if not candidate.is_file():
            raise FileNotFoundError(f"no model.pt in {checkpoint_dir}")
        path = candidate
    model = GPT.from_checkpoint(str(path), device="cpu")
    model.eval()
    return model, load_tokenizer(str(path.parent))


# ---------------------------------------------------------------------------
# sanity check
# ---------------------------------------------------------------------------
def check_onnx(path: str | Path, checkpoint_dir: str | Path,
               steps: int = 8, tol: float = 2e-3) -> float:
    """Run the ONNX graph and the PyTorch model on the same tokens.

    Returns the largest absolute difference in logits; raises if the graph is
    broken.  This is what we run before shipping an export.
    """
    import numpy as np
    import onnxruntime as ort

    path = Path(path)
    model, tokenizer = _load(checkpoint_dir)
    model.eval()

    text = "User: What is 12 + 30?\nAssistant:"
    ids = tokenizer.encode(text)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    cfg = model.config
    head_dim = cfg.n_embd // cfg.n_head
    feeds: Dict[str, Any] = {}
    for name in (i.name for i in session.get_inputs()):
        if name.startswith("past_"):
            feeds[name] = np.zeros((1, cfg.n_head, 0, head_dim), dtype=np.float32)

    worst = 0.0
    model.reset_cache()
    with torch.no_grad():
        for step, token in enumerate(ids[:steps]):
            tid = torch.tensor([[token]], dtype=torch.long)
            pos = torch.tensor([step], dtype=torch.long)
            torch_logits, _ = model.forward(tid, use_cache=True)

            feeds["token_id"] = tid.numpy()
            feeds["pos"] = pos.numpy()
            outputs = session.run(None, feeds)
            ort_logits = outputs[0]

            worst = max(worst, float(np.abs(ort_logits - torch_logits.numpy()).max()))
            for idx, name in enumerate(i.name for i in session.get_outputs()):
                if name.startswith("present_"):
                    feeds["past_" + name.split("_", 1)[1]] = outputs[idx]
    if worst > tol:
        raise AssertionError(
            f"onnx and pytorch disagree by {worst:.4f} (tolerance {tol})"
        )
    print(f"onnx ok: {steps} steps, max |onnx - torch| = {worst:.5f}")
    return worst


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Export a checkpoint to ONNX.")
    p.add_argument("checkpoint", nargs="?", default="checkpoints/orbit")
    p.add_argument("--out", default="exports/orbit.onnx")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="compare the exported graph against PyTorch")
    args = p.parse_args(argv)

    path = export_onnx(args.checkpoint, args.out, opset=args.opset,
                       quantize=not args.no_quantize)
    if args.check:
        check_onnx(path, args.checkpoint)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
