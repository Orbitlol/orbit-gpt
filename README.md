# OrbitGPT 🛰️

**A tiny language model you can train from scratch on your own machine — laptop CPU or a free Google Colab GPU — in a couple of minutes.**

No API key, no internet at inference time, no 3 GB of dependencies. Just PyTorch
and about a thousand lines of readable code you can actually understand:
a byte-level BPE tokenizer, a GPT-2 style transformer, a training loop, and a
chat REPL.

```text
python train.py                                  # ~2 min on a Colab T4, once
python generate.py --checkpoint out/orbit --chat # talk to your model, any time
```

**Train it once, then just talk to it.** The checkpoint is saved automatically
and every later run loads it instead of training again — on Colab it lives in
your Google Drive, so it survives "Runtime → Disconnect".

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
done, saves the model, then drops you into a chat box.

**The second time you run the cell it does not train.** It mounts your Google
Drive, finds `MyDrive/orbit-gpt/orbit_model`, loads it and starts chatting in a
couple of seconds:

```text
Found a trained model in /content/drive/MyDrive/orbit-gpt/orbit_model
Loading it - no waiting. (--retrain = train a new one, --resume = keep training)
```

| what you want | how |
|---------------|-----|
| just chat with the model I trained earlier | run the cell again (nothing else) |
| train a fresh model | `%run orbit_gpt_colab.py --retrain` |
| keep training the saved one longer | `%run orbit_gpt_colab.py --resume` |
| keep everything out of Google Drive | `%run orbit_gpt_colab.py --no-drive` |

(On a local machine the same thing happens in the out directory: `python
train.py` reuses `out/orbit` when it finds one.)

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
    corpus="conversation",               # what to learn (generated, no download)
    preset="auto",                       # auto|nano|micro|mini|small|base
    vocab_size=2048,
    out_dir="/content/orbit_model",
    save_to_drive=True,                  # the checkpoint survives session restarts
    retrain=False,                       # True = ignore the saved model
    ...
)
```

It also works as a normal local script:

```bash
python colab/orbit_gpt_colab.py --preset micro --corpus ./my_notes.txt
python colab/orbit_gpt_colab.py            # second run: loads, does not train
python colab/orbit_gpt_colab.py --retrain  # ignore the saved model, train again
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
sampling temperature, `/tokens 300` changes the reply length, `/rep 1.3` turns
up the repetition penalty, `/quit` leaves.
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

Parameter counts assume the default 2048-token vocabulary. Memory use is tiny:
even `base` fits in a few GB with the default batch size, and `nano` runs in
under 1 GB of RAM.

Everything is configurable:

```bash
python train.py --preset micro --n-layer 8 --n-embd 256 --block-size 256 \
                --batch-size 48 --max-steps 4000 --lr 3e-3
```

Useful flags: `--tokenizer char` (baseline), `--vocab-size`,
`--max-epochs 8` (cap the number of passes, so a small corpus is not memorised),
`--compile` (torch.compile, Linux/GPU), `--dtype bf16|fp16|fp32`,
`--resume out/orbit/model.pt`, `--eval-interval`, `--save-interval`, `--seed`.

---

## 📚 Built-in corpora

| name | what it is |
|------|------------|
| `conversation` | **default** — ~28,000 generated `User:` / `Assistant:` exchanges (3.2 MB) built on the fly, no download |
| `shakespeare` | ~1.1 MB of Shakespeare (the classic nanoGPT corpus) |
| `orbit-chat`  | 137 short `User:` / `Assistant:` exchanges built into the repo |
| `alice`, `pride`, `shakespeare-sonnets` | Project Gutenberg books |

A corpus spec can repeat a source with `:n`. The default is `conversation`.

Repeated copies are **shuffled** (block by block, deterministically): feeding
the same exchanges in the same order over and over teaches a tiny model the
*document order*, and it then replies with whatever came next in the corpus
instead of answering your question. Pass `--no-shuffle` to turn that off
(useful when repeating a novel, where order is the point). Downloads are cached
in `~/.cache/orbit-gpt`.

Which corpus should you use?

| goal | corpus | what to expect |
|------|--------|----------------|
| best chat answers | `conversation` (default) | answers phrasings it never saw during training |
| writer *and* chat | `shakespeare,conversation` | funnier, less accurate |
| best prose | `shakespeare` or a big book | fluent-ish pastiche, no chat ability |
| your own data | `./notes.txt,conversation` | your text, still able to chat |

---

## 🗣️ Why it answers instead of reciting

A 137-exchange corpus is *memorisable*: train on it and the model answers those
137 questions beautifully and everything else by falling back on the nearest
line it learned by heart. Three things stop that:

1. **A generated corpus, not a written one.** `conversation` is built by
   `orbit_gpt/corpora/conversation.py` at load time: ~28,000 exchanges, each
   intent (greetings, identity, definitions, how-tos, chitchat, refusals,
   advice, creative writing, meta) written with many question phrasings and 3-4
   answer variants, sampled and shuffled deterministically. Every arithmetic
   and unit-conversion answer is *computed*, so the training data is never
   wrong. There is no canonical sentence for the model to memorise.
2. **An epoch cap.** `--max-epochs 8` stops it grinding over the same text
   until it is word-perfect.
3. **A repetition penalty** (default 1.15) so it does not loop on one word.

Check it yourself — ask something that is *not* in the corpus:

```text
You: I've always wondered what a transformer is.
Orbit: A transformer is the architecture behind modern language models. Its
       attention layers let each token weigh how relevant the others are...
```

(Greeting/topic/answer-opener combinations are sampled independently, so the
reply is recomposed rather than copied — the phrasing above appears nowhere in
the training text.)

### Arithmetic

A 1M-parameter model cannot add two 2-digit numbers in its head — it will
invent "45 + 37 = 134". So `orbit_gpt/skills.py` answers clear calculations
with Python and leaves everything else to the model:

```text
You: What is 45 + 37?        -> 45 + 37 = 82.
You: What is 25% of 900?     -> 25% of 900 = 225.
You: 120 kg in pounds        -> 120 kg is about 264.55 pounds.
You: 100 f to c              -> 100 degrees Fahrenheit is about 37.8 degrees Celsius.
```

It is deliberately conservative: `answer_arithmetic(text)` returns `None`
unless the message is unambiguously a calculation, so "what is 2 + 2 in
Python?" still goes to the model. Verified on 12,000 random questions: 0 wrong.

---

## 🧠 Python API

```python
from orbit_gpt import GPT, TrainConfig, Trainer, build_tokenizer
from orbit_gpt.config import get_preset
from orbit_gpt.data import load_corpus, TokenDataset
from orbit_gpt.generate import chat, generate

text = load_corpus("conversation")          # or "orbit-chat:20", a file, a URL...
tok = build_tokenizer(text, "bpe", vocab_size=2048)
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
│   ├── skills.py      # deterministic arithmetic / unit conversions
│   └── corpora/
│       ├── conversation.py         # the generated dialogue corpus (default)
│       └── orbit_assistant.txt     # small hand-written exchange corpus
├── train.py           # CLI: train a model
├── generate.py        # CLI: sample / chat
├── colab/
│   ├── orbit_gpt_colab.py         # self-contained single file for Colab
│   └── OrbitGPT_Colab.ipynb       # ready-made notebook
├── tools/build_colab.py           # regenerates the single file from the package
├── tools/colab_footer.py          #   ...the Colab entry point it appends
└── tests/                         # 32 fast tests, no pytest needed
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

**On memorising:** a model this size trained on a small hand-written corpus
just recites it. That is why the default corpus is *generated* (see below) —
thousands of phrasings per intent, so there is no single sentence to memorise.
It still is not a general assistant: it has no world knowledge beyond its
training text, no memory between sessions, and it will happily be wrong with
total confidence.

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
