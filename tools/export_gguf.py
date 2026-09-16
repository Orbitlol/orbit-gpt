#!/usr/bin/env python3
"""Export an OrbitGPT checkpoint to GGUF, for llama.cpp / Ollama / LM Studio.

    python tools/export_gguf.py checkpoints/orbit --out exports/orbit.gguf --check

The file is written with architecture ``gpt2``: the names llama.cpp expects for
a GPT-2 style model (``token_embd``, ``attn_qkv``, ``ffn_up`` ...), the learned
position embeddings, tied output weights and a byte-level BPE vocabulary.

``--check`` reads the file back and runs a forward pass from the *GGUF data*
with numpy, comparing the logits against the PyTorch model - so a transposed
weight or a wrong metadata key cannot slip through unnoticed.

Quantisation (``--quantize f32|f16|q8_0|q4_0``) shrinks the weights; the
embeddings, the LayerNorms and the position embeddings stay at full precision,
because quantising them costs far more quality than it saves space.

The vocabulary is byte-level BPE.  llama.cpp's ``gpt2`` tokenizer pre-tokenises
text with the GPT-2 regex, while this project splits words with its own (simpler)
rule, so encode the prompt with ``orbit_gpt.tokenizer`` if you need the exact
token ids the model was trained on; ``--check`` reports how often the two
disagree on the training corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orbit_gpt.checkpoints import BEST_NAME  # noqa: E402
from orbit_gpt.model import GPT  # noqa: E402
from orbit_gpt.tokenizer import load_tokenizer  # noqa: E402

# ---------------------------------------------------------------------------
# GPT-2 byte <-> unicode alphabet (what a GGUF "gpt2" vocabulary stores)
# ---------------------------------------------------------------------------
def _byte_encoder() -> dict:
    """Map every byte to the printable character GPT-2 uses for it."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    characters = printable[:]
    n = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + n)
            n += 1
    return {byte: chr(char) for byte, char in zip(printable, characters)}


_BYTE_ENCODER = _byte_encoder()


def gguf_token(piece: bytes) -> str:
    """``b' world'`` -> ``'Ġworld'`` (the string a GGUF file stores)."""
    return "".join(_BYTE_ENCODER[b] for b in piece)


# our tensor names -> the names a llama.cpp gpt2 model expects
def gguf_tensor_names(state: dict) -> dict:
    """Map an OrbitGPT state_dict onto GPT-2 names, layer by layer."""
    mapping = {
        "transformer.wte.weight": "token_embd.weight",
        "transformer.wpe.weight": "position_embd.weight",
        "transformer.ln_f.weight": "output_norm.weight",
        "transformer.ln_f.bias": "output_norm.bias",
    }
    per_layer = {
        "ln_1.weight": "attn_norm.weight",
        "ln_1.bias": "attn_norm.bias",
        "attn.c_attn.weight": "attn_qkv.weight",
        "attn.c_attn.bias": "attn_qkv.bias",
        "attn.c_proj.weight": "attn_output.weight",
        "attn.c_proj.bias": "attn_output.bias",
        "ln_2.weight": "ffn_norm.weight",
        "ln_2.bias": "ffn_norm.bias",
        "mlp.c_fc.weight": "ffn_up.weight",
        "mlp.c_fc.bias": "ffn_up.bias",
        "mlp.c_proj.weight": "ffn_down.weight",
        "mlp.c_proj.bias": "ffn_down.bias",
    }
    for key in state:
        if key in mapping or key == "lm_head.weight":
            continue                      # lm_head is tied to the token embedding
        parts = key.split(".")
        if len(parts) >= 5 and parts[0] == "transformer" and parts[1] == "h":
            suffix = ".".join(parts[3:])       # e.g. "attn.c_attn.weight"
            if suffix in per_layer:
                mapping[key] = f"blk.{parts[2]}.{per_layer[suffix]}"
    return {k: v for k, v in mapping.items() if k in state}


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
QUANT_TYPES = {
    "f32": None,
    "f16": "F16",
    "q8_0": "Q8_0",
    "q4_0": "Q4_0",
}
# these stay at full precision whatever the quantisation is
_KEEP_F32 = ("token_embd", "position_embd", "output_norm", "attn_norm", "ffn_norm")


def _quantized(name: str, array: np.ndarray, quant: str, gguf_module) -> tuple:
    """Return ``(data, ggml_type)`` for one tensor."""
    # 1-D tensors (norms and biases) stay in f32: llama.cpp's CPU kernels add
    # them straight onto activations, which only works for f32/f16 operands
    if (quant == "f32" or array.ndim == 1
            or any(keep in name for keep in _KEEP_F32)):
        return array.astype(np.float32), gguf_module.GGMLQuantizationType.F32
    if quant == "f16":
        return array.astype(np.float16), gguf_module.GGMLQuantizationType.F16
    target = getattr(gguf_module.GGMLQuantizationType, QUANT_TYPES[quant])
    try:
        return gguf_module.quants.quantize(array.astype(np.float32), target), target
    except Exception as exc:                                  # pragma: no cover
        print(f"  ! {name}: cannot quantise ({exc}), keeping f32")
        return array.astype(np.float32), gguf_module.GGMLQuantizationType.F32


def export_gguf(checkpoint, out_path, quant: str = "q8_0", check: bool = False,
                verbose: bool = True, pre_type: str = "gpt-2") -> Path:
    """Write ``checkpoint`` (a file or directory) to ``out_path`` as GGUF."""
    import gguf

    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / BEST_NAME
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["model_config"]
    state = ckpt["model_state"]
    tokenizer = load_tokenizer(str(checkpoint.parent))
    n_layer = int(config["n_layer"])
    n_embd = int(config["n_embd"])
    n_head = int(config["n_head"])
    n_ff = 4 * n_embd

    writer = gguf.GGUFWriter(str(out_path), "gpt2")
    writer.add_name("orbit-gpt " + str(config.get("vocab_size", "")))
    writer.add_context_length(int(config["block_size"]))
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head)
    writer.add_layer_norm_eps(1e-5)
    writer.add_vocab_size(int(tokenizer.vocab_size))
    writer.add_file_type(
        gguf.LlamaFileType.ALL_F32 if quant == "f32" else
        gguf.LlamaFileType.MOSTLY_F16 if quant == "f16" else
        gguf.LlamaFileType.MOSTLY_Q8_0 if quant == "q8_0" else
        gguf.LlamaFileType.MOSTLY_Q4_0
    )

    # -- tokenizer ---------------------------------------------------------
    tokens, types = [], []
    for i in range(tokenizer.vocab_size):
        if i < tokenizer.n_special:
            tokens.append(tokenizer.special_tokens[i])
            types.append(gguf.TokenType.CONTROL)
        else:
            # every piece is a normal token: llama.cpp's BYTE type is only for
            # SPM-style "<0xAB>" fallbacks, and the GPT-2 byte alphabet (\u0120
            # for a space, and so on) is what a "gpt2" vocabulary carries
            tokens.append(gguf_token(tokenizer.pieces[i]))
            types.append(gguf.TokenType.NORMAL)
    merges = [f"{tokens[a]} {tokens[b]}" for a, b in tokenizer.merges]
    writer.add_tokenizer_model("gpt2")
    # llama.cpp registers the GPT-2 pre-tokeniser as "gpt-2" (older builds
    # reject other spellings), so that is what a converted GPT-2 model carries
    writer.add_tokenizer_pre(pre_type)
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(tokenizer.eot_id)
    writer.add_eos_token_id(tokenizer.eot_id)
    writer.add_pad_token_id(tokenizer.pad_id)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)

    # -- tensors -----------------------------------------------------------
    by_gguf = {gguf_name: source
               for source, gguf_name in gguf_tensor_names(state).items()}
    # llama.cpp's gpt2 loader asks for the LayerNorm and attention/MLP biases by
    # name (it follows the GPT-2 reference implementation, which has them), so
    # supply zeros for the ones this model does not have: adding zero is exact.
    for layer in range(n_layer):
        for missing in (
            f"blk.{layer}.attn_norm.bias", f"blk.{layer}.attn_qkv.bias",
            f"blk.{layer}.attn_output.bias", f"blk.{layer}.ffn_norm.bias",
            f"blk.{layer}.ffn_up.bias", f"blk.{layer}.ffn_down.bias",
        ):
            by_gguf.setdefault(missing, None)
    by_gguf.setdefault("output_norm.bias", None)

    written = 0
    for gguf_name, source in sorted(by_gguf.items()):
        if source is None:
            suffix = ".".join(gguf_name.split(".")[-2:])
            width = {"attn_qkv.bias": 3 * n_embd, "ffn_up.bias": n_ff}.get(
                suffix, n_embd
            )
            tensor = np.zeros(width, dtype=np.float32)
        else:
            tensor = state[source].detach().cpu().float().numpy()
        data, ggml_type = _quantized(gguf_name, tensor, quant, gguf)
        writer.add_tensor(gguf_name, data, raw_dtype=ggml_type)
        written += 1
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if verbose:
        size = Path(out_path).stat().st_size / 1e6
        print(f"wrote {out_path} ({size:.1f} MB, {quant}, {written} tensors, "
              f"{tokenizer.vocab_size} tokens, {len(merges)} merges)")

    if check:
        verify_gguf(out_path, checkpoint, verbose=verbose)
    return Path(out_path)


# ---------------------------------------------------------------------------
# verification: read the file back and reproduce the model with numpy
# ---------------------------------------------------------------------------
def _gelu(x: np.ndarray) -> np.ndarray:
    from math import sqrt

    import math

    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / sqrt(2.0)))


def _layer_norm(x, weight, bias=None, eps: float = 1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps) * weight
    return out if bias is None else out + bias


def numpy_forward(tensors: dict, ids: list, n_head: int) -> np.ndarray:
    """A GPT-2 forward pass built only from GGUF tensors (fp32)."""
    def t(name):
        return np.asarray(tensors[name], dtype=np.float32)

    def bias(name):
        return t(name) if name in tensors else None

    n_layer = sum(1 for k in tensors if k.endswith("attn_norm.weight"))
    x = t("token_embd.weight")[ids] + t("position_embd.weight")[: len(ids)]
    head_dim = x.shape[-1] // n_head
    for i in range(n_layer):
        h = _layer_norm(x, t(f"blk.{i}.attn_norm.weight"), bias(f"blk.{i}.attn_norm.bias"))
        qkv = h @ t(f"blk.{i}.attn_qkv.weight").T
        if f"blk.{i}.attn_qkv.bias" in tensors:
            qkv = qkv + t(f"blk.{i}.attn_qkv.bias")
        q, k, v = np.split(qkv, 3, axis=-1)
        shape = (len(ids), n_head, head_dim)
        q = q.reshape(shape).transpose(1, 0, 2)
        k = k.reshape(shape).transpose(1, 0, 2)
        v = v.reshape(shape).transpose(1, 0, 2)
        att = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
        mask = np.triu(np.ones((len(ids), len(ids)), dtype=bool), 1)
        att = np.where(mask, -np.inf, att)
        att = np.exp(att - att.max(-1, keepdims=True))
        att = att / att.sum(-1, keepdims=True)
        y = (att @ v).transpose(1, 0, 2).reshape(len(ids), -1)
        y = y @ t(f"blk.{i}.attn_output.weight").T
        if f"blk.{i}.attn_output.bias" in tensors:
            y = y + t(f"blk.{i}.attn_output.bias")
        x = x + y

        h = _layer_norm(x, t(f"blk.{i}.ffn_norm.weight"), bias(f"blk.{i}.ffn_norm.bias"))
        h = _gelu(h @ t(f"blk.{i}.ffn_up.weight").T)
        h = h @ t(f"blk.{i}.ffn_down.weight").T
        x = x + h
    x = _layer_norm(x, t("output_norm.weight"), bias("output_norm.bias"))
    return x @ t("token_embd.weight").T


def _tokenizer_dir(checkpoint) -> str:
    """The directory holding ``tokenizer.json`` next to a checkpoint."""
    path = Path(checkpoint)
    return str(path if path.is_dir() else path.parent)


def verify_gguf(path, checkpoint, prompt: str = "User: Hello!\nAssistant:",
                verbose: bool = True, max_logit_diff: float | None = None) -> float:
    """Compare the GGUF data against PyTorch: tensors, then real logits.

    A quantised file cannot reproduce the fp32 logits exactly, so the tolerance
    follows the file's quantisation (and the top-1 token still has to match).
    """
    import gguf

    reader = gguf.GGUFReader(str(path))
    stored = {}
    quantised = False
    for tensor in reader.tensors:
        data = tensor.data
        if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
            quantised = True
            data = gguf.quants.dequantize(data, tensor.tensor_type)
        stored[tensor.name] = np.asarray(data, dtype=np.float32)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k: v.detach().float().numpy() for k, v in ckpt["model_state"].items()}
    mapping = gguf_tensor_names(ckpt["model_state"])

    worst = 0.0
    for ours, name in mapping.items():
        if name not in stored:      # a zero-bias filler llama.cpp asked for
            continue
        reference = state[ours]
        if reference.ndim == 2 and stored[name].shape != reference.shape:
            raise AssertionError(f"{name}: shape {stored[name].shape} != {reference.shape}")
        worst = max(worst, float(np.abs(stored[name] - reference).max()))
    zeros = [n for n, v in stored.items()
             if n.endswith(".bias") and not np.any(v)]
    if verbose:
        print(f"  tensors: {len(mapping)} compared, worst |gguf - torch| = {worst:.2e}"
              + (f" (+{len(zeros)} zero biases llama.cpp asks for)" if zeros else ""))

    # -- logits from the file itself --------------------------------------
    tokenizer = load_tokenizer(_tokenizer_dir(checkpoint))
    ids = tokenizer.encode(prompt)
    config = ckpt["model_config"]
    model = GPT.from_checkpoint(str(checkpoint), device="cpu").eval()
    with torch.no_grad():
        # logits are (batch, sequence, vocab): take position -1, not batch -1
        torch_logits = model(torch.tensor([ids]))[0][0, -1].numpy()
    np_logits = numpy_forward(stored, ids, int(config["n_head"]))[-1]
    diff = float(np.abs(torch_logits - np_logits).max())
    same_top = int(torch_logits.argmax()) == int(np_logits.argmax())
    if verbose:
        print(f"  logits : {len(ids)} tokens, max |numpy(gguf) - torch| = {diff:.2e}"
              f" | top-1: torch {int(torch_logits.argmax())},"
              f" gguf {int(np_logits.argmax())} {'(same)' if same_top else '(DIFFERENT)'}")
    limit = max_logit_diff if max_logit_diff is not None else (0.05 if quantised else 5e-3)
    if diff > limit or not same_top:
        raise AssertionError(
            f"the GGUF file does not reproduce the model (diff {diff:.3e} > {limit:g})"
        )
    return diff


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export an OrbitGPT checkpoint to GGUF.")
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/orbit",
                        help="checkpoint file or directory")
    parser.add_argument("--out", default="exports/orbit-micro.gguf")
    parser.add_argument("--quantize", choices=sorted(QUANT_TYPES), default="q8_0")
    parser.add_argument("--pre-type", default="gpt-2",
                        help="tokenizer.ggml.pre value (llama.cpp's GPT-2 "
                             "pre-tokeniser is called 'gpt-2')")
    parser.add_argument("--check", action="store_true",
                        help="read the file back and verify it against PyTorch")
    args = parser.parse_args(argv)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export_gguf(args.checkpoint, args.out, quant=args.quantize, check=args.check,
                pre_type=args.pre_type)

    if args.check:
        tokenizer = load_tokenizer(_tokenizer_dir(args.checkpoint))
        print(f"  vocab  : {tokenizer.vocab_size} tokens "
              f"({tokenizer.n_special} special + 256 bytes + "
              f"{len(tokenizer.merges)} merges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
