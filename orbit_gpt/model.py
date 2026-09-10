"""The model: a plain GPT-2 style decoder-only transformer.

Deliberately boring on purpose - this is the architecture every small LLM uses,
written so you can read it top to bottom in one sitting:

    tokens -> token embedding + positional embedding -> N x [LayerNorm -> causal
    self-attention -> +, LayerNorm -> MLP -> +] -> LayerNorm -> linear head

Extras that matter in practice:

* fused QKV projection and PyTorch's fast attention kernel (Flash/mem-efficient
  SDPA) when it is available,
* a KV cache so generation is O(new tokens) instead of O(sequence length),
* weight tying between the embedding and the output head (fewer parameters,
  slightly better perplexity on small data),
* top-k / top-p sampling with temperature, and stop sequences for chat.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from orbit_gpt.config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (GPT-2 uses a bias, most modern models don't)."""

    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # one fused matrix for Q, K and V - faster than three separate matmuls
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # PyTorch >= 2.0 ships a memory efficient / flash attention kernel.
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # causal mask, used as a plain additive bias in the fallback path
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
                persistent=False,
            )

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat((cache_k, k), dim=2)
            v = torch.cat((cache_v, v), dim=2)
        new_cache = (k, v)
        T_full = k.size(2)

        # During prefill (T == T_full) we need a causal mask.  While decoding
        # with a cache (T < T_full) every cached key is already in the past, so
        # no mask is needed - which also keeps every backend happy.
        is_causal = T == T_full

        if self.flash:
            dropout_p = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            mask = self.bias[:, :, T_full - T : T_full, :T_full]  # type: ignore[index]
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_cache


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, new_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class GPT(nn.Module):
    """A small decoder-only transformer.

    ``forward(idx, targets=None, use_cache=False) -> (logits, loss)``

    * with ``targets``: ``logits`` covers every position and ``loss`` is the
      cross-entropy for next-token prediction,
    * without ``targets`` and with ``use_cache=True``: only the logits of the
      last position are computed (this is the generation path).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: the unembedding matrix *is* the embedding matrix
        if config.vocab_size > 0:
            self.transformer.wte.weight = self.lm_head.weight  # type: ignore[index]

        # KV cache state (only used by the generation path)
        self._kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * config.n_layer
        self._cache_len = 0

        self.apply(self._init_weights)
        # scaled initialisation of the residual projections (GPT-2 recipe)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    # -- setup ------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def block_size(self) -> int:
        return self.config.block_size

    @property
    def n_params(self) -> int:
        """Trainable parameters.  Tied weights are counted once."""
        return sum(
            p.numel() for n, p in self.named_parameters() if not n.endswith("wte.weight")
        )

    def reset_cache(self) -> None:
        self._kv_cache = [None] * self.config.n_layer
        self._cache_len = 0

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(
                f"cannot forward a sequence of length {T}, block size is "
                f"{self.config.block_size}"
            )
        pos = self._cache_len if use_cache else 0

        tok_emb = self.transformer.wte(idx)  # type: ignore[operator]
        pos_emb = self.transformer.wpe(  # type: ignore[operator]
            torch.arange(pos, pos + T, dtype=torch.long, device=device)
        )
        x = self.transformer.drop(tok_emb + pos_emb)  # type: ignore[operator]

        caches = self._kv_cache if use_cache else [None] * self.config.n_layer
        new_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for block, cache in zip(self.transformer.h, caches):  # type: ignore[arg-type]
            x, new_cache = block(x, cache)
            new_caches.append(new_cache)

        if use_cache:
            self._kv_cache = new_caches
            self._cache_len = pos + T

        x = self.transformer.ln_f(x)  # type: ignore[operator]

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # inference: only the final position matters
            x_last = x[:, [-1], :] if use_cache else x
            logits = self.lm_head(x_last)
            loss = None
        return logits, loss

    # -- optimisation -----------------------------------------------------
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """AdamW with weight decay on matrices only (no decay on biases / norms)."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs = {}
        if device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
            kwargs["fused"] = True
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **kwargs)

    # -- generation -------------------------------------------------------
    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: float,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        logits = logits / max(temperature, 1e-6)
        if top_k is not None and 0 < top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k)
            logits = logits.masked_fill(logits < v[..., [-1]], float("-inf"))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            # shift right so the token that crosses the threshold is kept
            remove = cum - probs > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    @torch.no_grad()
    def stream(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
        stop_sequences: Sequence[Sequence[int]] = (),
        stop_token_ids: Sequence[int] = (),
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Iterator[int]:
        """Yield generated token ids one by one (KV cached, so this is fast)."""
        device = device or next(self.parameters()).device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        self.eval()
        self.reset_cache()
        block = self.config.block_size

        ids = list(prompt_ids)[-block:]  # never feed more than the context window
        room = max(0, block - len(ids))  # how much context is left for new tokens
        max_new_tokens = max(0, min(max_new_tokens, room))
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = self.forward(idx, use_cache=True)  # prefill

        stop_token_ids = list(stop_token_ids)
        produced: List[int] = []
        for _ in range(max_new_tokens):
            next_id = self._sample(
                logits[:, -1, :], temperature, top_k, top_p, generator
            )
            new_id = int(next_id.item())
            if new_id in stop_token_ids:
                break
            produced.append(new_id)
            yield new_id
            # stop sequences are matched against the tail of the generation
            if stop_sequences and any(
                len(produced) >= len(s) and produced[len(produced) - len(s):] == list(s)
                for s in stop_sequences
            ):
                break
            idx = torch.tensor([[new_id]], dtype=torch.long, device=device)
            logits, _ = self.forward(idx, use_cache=True)  # one token at a time

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
        stop_sequences: Sequence[Sequence[int]] = (),
        stop_token_ids: Sequence[int] = (),
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        callback=None,
    ) -> List[int]:
        """Sample up to ``max_new_tokens`` ids.  ``callback(id)`` fires per token."""
        out: List[int] = []
        for new_id in self.stream(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_sequences=stop_sequences,
            stop_token_ids=stop_token_ids,
            seed=seed,
            device=device,
        ):
            out.append(new_id)
            if callback is not None:
                callback(new_id)
        return out

    # -- persistence ------------------------------------------------------
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "orbit-gpt",
            "version": 1,
            "model_config": asdict(self.config),
            "model_state": self.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str = "cpu", map_location: Optional[str] = None
    ) -> "GPT":
        """Load a model saved with :meth:`save` (handles the raw state_dict too)."""
        ckpt = torch.load(path, map_location=map_location or device)
        if "model_config" in ckpt:
            config = GPTConfig.from_dict(ckpt["model_config"])
            state = ckpt["model_state"]
        else:  # tolerate a bare state_dict
            raise ValueError(f"{path} is not an OrbitGPT checkpoint")
        model = cls(config)
        # be forgiving about a tied-weight key that may be missing
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing and any(not k.endswith("wte.weight") for k in missing):
            raise RuntimeError(f"checkpoint is missing weights: {missing}")
        return model.to(device)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.n_params/1e6:.2f}M params, {self.config}"
