# OrbitGPT 🛰️

**A tiny language model you can train from scratch on your own machine — laptop CPU or a free Google Colab GPU — in a couple of minutes.**

No API key, no internet at inference time, no 3 GB of dependencies. Just PyTorch
and about a thousand lines of readable code you can actually understand:
a byte-level BPE tokenizer, a GPT-2 style transformer, a training loop, and a
chat REPL.

```text
python train.py                                  # ~2 min on a Colab T4
python generate.py --checkpoint out/orbit --chat # talk to your model
```

---

## ⚡ Google Colab — 60 second quickstart

1. Open <https://colab.research.google.com> → **New notebook**
2. **Runtime → Change runtime type → T4 GPU** (CPU works too, just slower)
3. Paste this into **one cell** and run it:

```python
# OrbitGPT - paste into one Colab cell and run
import os, urllib.request
URLS = ["https://raw.githubusercontent.com/Orbitlol/orbit-gpt/main/colab/orbit_gpt_colab.py"]
if not (os.path.exists("orbit_gpt_colab.py") and os.path.getsize("orbit_gpt_colab.py") > 5000):
    for url in URLS:
        try:
            urllib.request.urlretrieve(url, "orbit_gpt_colab.py"); break
        except Exception as e:
            print("download failed:", e)
print("orbit_gpt_colab.py:", os.path.getsize("orbit_gpt_colab.py"), "bytes")
%run orbit_gpt_colab.py
```

(`%run` instead of `!python` is deliberate: it runs the file inside the
notebook kernel, which is what lets the chat box at the end read your typing.)

That's it. The script prints its progress, shows a couple of samples when it is
done, saves the model to `/content/orbit_model` (and offers it as a `.zip`
download), then drops you into a chat box.

**No download?** Either
upload `colab/orbit_gpt_colab.py` from this repo into Colab and run
`!python orbit_gpt_colab.py`, or open
`colab/OrbitGPT_Colab.ipynb` directly in Colab
(badge below → *Open in Colab* needs the notebook to be on GitHub).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Orbitlol/orbit-gpt/blob/main/colab/OrbitGPT_Colab.ipynb)

The single file is **self-contained**: it embeds the built-in assistant corpus
so it works even with no network access beyond the corpus download, and it
takes PyTorch as its only dependency. Tweak the `CONFIG` block at the top to
change the corpus, model size, or chat settings:

```python
CONFIG = dict(
    corpus="orbit-chat:20",             # what to learn
    preset="auto",                       # auto|nano|micro|mini|small|base
    vocab_size=1024,
    out_dir="/content/orbit_model",
    ...
)
```

It also works as a normal local script:

```bash
python colab/orbit_gpt_colab.py --preset micro --corpus ./my_notes.txt
python colab/orbit_gpt_colab.py --no-train --chat        # reload and just chat
```

---

## 💻 Run it on your own PC

```bash
git clone https://github.com/Orbitlol/orbit-gpt.git
cd orbit-gpt
pip install torch                 # that is the only hard dependency

python train.py                   # auto-picks nano on CPU / micro on GPU
python generate.py --checkpoint out/orbit --chat
```

CPU training uses the `nano` preset by default and takes a few minutes on a
modern laptop (~10 min on a 2-core machine). Add `--preset micro` if you have a
GPU and `--dataset` to point at your own text.

### Train on your own text

```bash
python train.py --dataset ./notes.txt                       # one file
python train.py --dataset ./my_folder                       # every .txt in a folder
python train.py --dataset "https://example.com/book.txt"    # or a URL
python train.py --dataset "shakespeare,orbit-chat:20"       # or mix corpora
```

Then chat with it:

```bash
python generate.py --checkpoint out/orbit --chat
# or one-shot:
python generate.py -c out/orbit -p "Once upon a time" -t 0.9 -n 3
```

In the chat box: `/reset` clears the conversation, `/temp 1.0` changes the
sampling temperature, `/tokens 300` changes the reply length, `/quit` leaves.
Long conversations are trimmed automatically so they always fit the context
window (the `nano` model remembers roughly the last exchange, `micro` a few).

---

## 📏 Model sizes

| preset | layers | heads | width | context | params | trains in |
|--------|--------|-------|-------|---------|--------|-----------|
| `nano`  | 4  | 4  | 128 | 128 | 0.8M  | ~5 min (laptop CPU) · ~1 min (T4) |
| `micro` | 6  | 6  | 192 | 192 | 2.7M  | ~2 min (T4 GPU) · ~25 min (laptop CPU) |
| `mini`  | 8  | 8  | 256 | 256 | 6.4M  | ~6 min (T4 GPU) |
| `small` | 10 | 12 | 384 | 320 | 17.8M | ~20 min (T4 GPU) |
| `base`  | 12 | 8  | 512 | 384 | 38.0M | ~1 h (needs a big corpus) |

Timings are for the default 2000 steps on the default corpus; the CPU numbers
were measured on a 2-core machine, so a typical 8-core laptop is ~4× quicker
(`nano` finishes in well under two minutes). GPU numbers are estimates.

Parameter counts assume the default 1024-token vocabulary. Memory use is tiny:
even `base` fits in a few GB with the default batch size, and `nano` runs in
under 1 GB of RAM.

Everything is configurable:

```bash
python train.py --preset micro --n-layer 8 --n-embd 256 --block-size 256 \
                --batch-size 48 --max-steps 4000 --lr 3e-3
```

Useful flags: `--tokenizer char` (baseline), `--vocab-size 2048`,
`--epochs 20` (derive steps from corpus size), `--compile` (torch.compile,
Linux/GPU), `--dtype bf16|fp16|fp32`, `--resume out/orbit/model.pt`,
`--eval-interval`, `--save-interval`, `--seed`.

---

## 📚 Built-in corpora

| name | what it is |
|------|------------|
| `shakespeare` | ~1.1 MB of Shakespeare (the classic nanoGPT corpus) |
| `orbit-chat`  | 137 short `User:` / `Assistant:` exchanges built into the repo |
| `alice`, `pride`, `shakespeare-sonnets` | Project Gutenberg books |

A corpus spec can repeat a source with `:n`. The default is `orbit-chat:20` —
the built-in assistant corpus repeated 20 times (about 500 KB of text) —
because that is what makes a *small* model behave like a chat assistant.

Repeated copies are **shuffled** (block by block, deterministically): feeding
the same 137 exchanges in the same order 20 times teaches a tiny model the
*document order*, and it then replies with whatever exchange came next in the
corpus instead of answering your question. Pass `--no-shuffle` to turn that off
(useful when repeating a novel, where order is the point). Downloads are cached
in `~/.cache/orbit-gpt`.

Which corpus should you use?

| goal | corpus | what to expect |
|------|--------|----------------|
| best chat answers | `orbit-chat:20` (default) | ~8/10 short questions answered correctly |
| writer *and* chat | `shakespeare,orbit-chat:20` | funnier, less accurate (~6/10) |
| best prose | `shakespeare` or a big book | fluent-ish pastiche, no chat ability |
| your own data | `./notes.txt,orbit-chat:10` | your text, still able to chat |

(Measured with the `nano` preset: 10 mixed questions after 2000 steps. More
capacity — the `micro`/`mini` presets — narrows the gap, so mixing is a good
deal once you are on a GPU.)

---

## 🧠 Python API

```python
from orbit_gpt import GPT, TrainConfig, Trainer, build_tokenizer
from orbit_gpt.config import get_preset
from orbit_gpt.data import load_corpus, TokenDataset
from orbit_gpt.generate import chat, generate

text = load_corpus("orbit-chat:20")
tok = build_tokenizer(text, "bpe", vocab_size=1024)
dataset = TokenDataset(tok.encode(text), val_fraction=0.1)

cfg = get_preset("micro")
cfg.vocab_size = tok.vocab_size
model = GPT(cfg)

Trainer(model, dataset, TrainConfig(max_steps=2000), tok).train()

print(generate(model, tok, "User: Hello!\nAssistant:", max_new_tokens=100))
chat(model, tok)          # interactive REPL
```

Load a trained model later with:

```python
from orbit_gpt.generate import load_model
model, tokenizer = load_model("out/orbit")
```

---

## 🗂️ Project layout

```text
orbit-gpt/
├── orbit_gpt/
│   ├── config.py      # GPTConfig, TrainConfig and the size presets
│   ├── tokenizer.py   # byte-level BPE + char tokenizer (pure Python, no deps)
│   ├── model.py       # the transformer: attention, KV cache, sampling
│   ├── data.py        # corpora, downloading, train/val split, batching
│   ├── train.py       # AdamW + cosine schedule + AMP + checkpointing
│   ├── generate.py    # sampling, stop strings, chat REPL
│   └── data/orbit_assistant.txt   # built-in assistant corpus
├── train.py           # CLI: train a model
├── generate.py        # CLI: sample / chat
├── colab/
│   ├── orbit_gpt_colab.py         # self-contained single file for Colab
│   └── OrbitGPT_Colab.ipynb       # ready-made notebook
├── tools/build_colab.py           # regenerates the single file from the package
└── tests/test_smoke.py            # 20 fast tests, no pytest needed
```

`colab/orbit_gpt_colab.py` is **generated** from the package by
`python tools/build_colab.py` so the copy-paste version can never drift from
the real code.

---

## 🛠️ Tuning tips

* **Loss plateaus too high?** More steps. Small models on small corpora need
  many epochs; 20–50 epochs over a 1 MB corpus is normal.
* **Garbage output?** Lower `--lr` to `1e-3`, raise `--batch-size`, or train
  longer. Also check that your corpus is actually big enough (aim for at least
  a few hundred KB).
* **Repetitive chat answers?** Sample with a higher `--temperature` (0.9–1.0)
  and lower `--top-p` (0.9). Generation uses top-k + nucleus sampling.
* **It answers fine at first, then drifts?** A small model reads its own
  replies, so one wrong answer poisons the next turn. Type `/reset` in the chat
  box, or train a bigger preset: more capacity means longer, more reliable
  conversations.
* **Out of memory?** Smaller `--batch-size` with `--grad-accum 4` keeps the
  effective batch size while cutting memory.
* **Faster on CPU?** Use the `nano` preset, keep `--block-size 128`, and let
  PyTorch use all cores (`torch.set_num_threads`).

Run the tests with `python tests/test_smoke.py`.

---

## 🎯 Honest expectations

OrbitGPT is a *toy* by modern standards — it is the "nanoGPT on Shakespeare"
scale of model, trained for minutes on a few hundred kilobytes of text. It will produce
recognisable, often charming, frequently wrong text.

Be aware of one thing in particular: the built-in `orbit-chat` corpus is only
26 KB, so with the default settings the model largely **memorises** those 137
exchanges. It will answer those questions well and anything you did not train
on poorly. That is normal for a model this size and is the reason to feed it
your own, larger corpus once you want something less canned.

It is genuinely useful as:

* a model you can read end-to-end and understand completely,
* a local playground for prompting, sampling and fine-tuning experiments,
* a starting point you can scale up (more data + `mini`/`small` preset = much
  better text).

It is not going to replace a hosted LLM. That's the point: it's yours, it runs
offline, and you can train it on your own notes.

---

## License

MIT — see [LICENSE](LICENSE).
