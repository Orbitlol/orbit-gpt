# OrbitGPT 🛰️

**A tiny language model you can train from scratch on your own machine — laptop CPU or a free Google Colab GPU — in a couple of minutes.**

No API key, no internet at inference time, no 3 GB of dependencies. Just PyTorch
and about a thousand lines of readable code you can actually understand:
a byte-level BPE tokenizer, a GPT-2 style transformer, a training loop, and a
chat REPL.

```text
python train.py                                  # ~5 min on a Colab T4, once
python generate.py --checkpoint checkpoints/orbit --chat   # talk to it, any time
```

**Train it once, then just talk to it.** Checkpoints go to
`checkpoints/orbit` inside the repo (`/content/checkpoints/orbit` when the
single-file build runs standalone in Colab) — never to Google Drive — and every
later run loads the newest one instead of training again. It can also search
the web before answering when a question needs fresh information.

---

## ⚡ Google Colab — two cells, that's the whole procedure

1. Open <https://colab.research.google.com> → **New notebook**
2. **Runtime → Change runtime type → T4 GPU** (CPU also works, ~4× slower)
3. Paste **cell 1**, run it, paste **cell 2**, run it. Done.

**Cell 1 — setup (once per session, ~20 s)**

```python
import os

if not os.path.isdir("orbit-gpt"):
    !git clone -q --depth 1 https://github.com/Orbitlol/orbit-gpt.git

%cd orbit-gpt
!pip install -q ddgs        # optional: web search. Delete this line to skip.
print("ready - now run the next cell")
```

**Cell 2 — train, then chat**

```python
%run colab/orbit_gpt_colab.py
```

That is it. The first run trains and then opens the chat box; **every later run
finds the checkpoint and goes straight to chatting.** Checkpoints live in
`/content/checkpoints/orbit` — **never on Google Drive**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Orbitlol/orbit-gpt/blob/main/colab/OrbitGPT_2_Cells.ipynb)

`colab/OrbitGPT_2_Cells.ipynb` is that same two-cell notebook, ready to open.

### What happens inside cell 2

| stage | what it learns | why |
|-------|----------------|-----|
| 1 — pre-training | ~3.6 MB of generated prose | sentence structure. Skip this and a 5 M-parameter model produces word salad |
| 2 — SFT | ~48 k `User:`/`Assistant:` exchanges | the chat format, starting from the stage-1 weights |

Then it drops you into chat. Options:

| what you want | how |
|---------------|-----|
| chat with the model I trained earlier | run cell 2 again — nothing else |
| train a fresh model | `%run colab/orbit_gpt_colab.py --retrain` |
| keep training the saved one longer | `%run colab/orbit_gpt_colab.py --resume --max-steps 4000` |
| a bigger model | `%run colab/orbit_gpt_colab.py --preset mini --max-steps 3000` |
| never touch the network | `%run colab/orbit_gpt_colab.py --no-search` |
| skip stage 1 (not recommended) | `%run colab/orbit_gpt_colab.py --skip-pretrain` |

The single file is **self-contained**: it embeds the built-in assistant corpus,
so it still works offline, and PyTorch is its only real dependency. Tweak the
`CONFIG` block at the top to change the corpora, model size, or chat settings:

```python
CONFIG = dict(
    preset="micro",           # nano|micro|mini|small|base  (micro = 4.8M)
    pretrain_corpus="prose",  # stage 1: plain prose -> sentence structure
    pretrain_fraction=0.4,    # share of the step budget spent on stage 1
    sft_corpus="conversation",  # stage 2: the User:/Assistant: format
    sft_lr=None,              # None = preset lr / 3
    out_dir="",               # "" = checkpoints/orbit (never Drive)
    use_web_search=True,      # False = never touch the network
    web_results=5,
    chat_repetition_penalty=1.15,   # >1 stops it looping on the same words
    ...
)
```

Hyper-parameters (learning rate, batch size, warmup, weight decay, gradient
clipping, epoch cap) come from `PRESET_TRAIN` in `orbit_gpt/config.py`, so
"make it bigger" is a one-line change.

It also works as a normal local script:

```bash
python colab/orbit_gpt_colab.py             # train (2 stages), then chat
python colab/orbit_gpt_colab.py             # second run: loads, does not train
python colab/orbit_gpt_colab.py --retrain   # ignore the saved model, train again
python colab/orbit_gpt_colab.py --resume    # keep training the saved model
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

## 📦 Take it home: one small file, no PyTorch

Export the trained model to **ONNX** and run it in a desktop chat window. The
app needs `onnxruntime` (~15 MB) and nothing else — no PyTorch, no internet.

```bash
python -m orbit_gpt.export checkpoints/orbit --out exports/orbit.onnx
pip install onnxruntime
python orbit_app.py --model exports/orbit-int8.onnx
```

```text
$ ls -la exports/
  orbit.onnx       25.2 MB   fp32 decoder step  (measured, `micro`)
  orbit-int8.onnx   7.6 MB   8-bit weights      <- the app prefers this one
  tokenizer.json                the vocabulary it was trained with
  config.json                   architecture + defaults
```

`orbit_app.py` opens a small desktop window. Tkinter ships with Python on
Windows and macOS; on Debian/Ubuntu it is `sudo apt install python3-tk`, and
without it the app falls back to the terminal automatically.
`--no-gui` chooses the terminal up front. Add `--check` to the
export to verify the graph against PyTorch before you ship it:

```text
$ python -m orbit_gpt.export checkpoints/orbit --out exports/orbit.onnx --check
wrote exports/orbit.onnx (19.4 MB)
wrote exports/orbit-int8.onnx (5.1 MB, 3.8x smaller)
onnx ok: 8 steps, max |onnx - torch| = 0.00001
```

**How the export stays small and fast**

* the graph is a **single decode step** — one token in, logits out, with the KV
  cache handed back — so generation costs one small forward pass per token
  instead of re-running the whole context;
* 8-bit weight quantisation makes it ~4x smaller with no measurable change in
  the replies (the `--check` tolerance for int8 is 0.5 logits).

| preset | params | fp32 | int8 | measured on |
|--------|--------|------|------|-------------|
| nano   | 0.8 M  | 4.2 MB | 1.5 MB | this repo |
| micro  | 4.8 M  | 25 MB  | 7.6 MB | this repo |
| mini   | 6.4 M  | ~32 MB | ~9 MB  | (scales with the parameter count) |

---

---

## ⚙️ Why it is cheap to train

The pipeline only spends time on work that changes the model:

| what | effect |
|------|--------|
| **tokenizer + token cache** | the 7 MB corpus is tokenised once; later runs load `tokens.npy` instead of re-doing it (~10 s saved per run) |
| **right-sized vocabulary** | `nano` uses 1024 tokens instead of 2048 — the output matrix is a third of a 0.8 M model, so every step is cheaper |
| **trimmed corpora** | 48 k exchanges instead of 74 k: the same coverage, less work per epoch |
| **epoch cap** | `--max-epochs 8` stops when the data is saturated instead of grinding on |
| **two-stage SFT** | prose first (sentence shape) then chat (format); the second stage converges in far fewer steps than learning both at once |
| **train once, chat many** | every later run loads the checkpoint — no retraining |
| **ONNX int8 + KV cache** | ~5 MB model, one small forward pass per generated token |

Measured on a **2-core laptop CPU** (no GPU):

| preset | tokens/step | ms/step | 1000 steps |
|--------|-------------|---------|------------|
| nano   | 1,024  | 150 ms   | 2.5 min |
| micro  | 2,048  | 1,250 ms | 21 min |

A free Colab **T4** is roughly 10x that, which is why the default 2,000-step
`micro` run finishes in a few minutes there.

---

---

## 📏 Model sizes

| preset | layers | heads | width | context | params | trains in |
|--------|--------|-------|-------|---------|--------|-----------|
| `nano`  | 4  | 4  | 128 | 128 | 0.80M | ~5 min (laptop CPU) · ~1 min (T4) |
| **`micro` (default)** | 6 | 8 | 256 | 384 | **4.82M** | **~5 min (T4 GPU)** · ~1 h (laptop CPU) |
| `mini`  | 8  | 8  | 256 | 256 | 6.36M | ~10 min (T4 GPU) |
| `small` | 10 | 12 | 384 | 320 | ~20M  | ~20 min (T4 GPU) |
| `base`  | 12 | 8  | 512 | 384 | ~45M  | ~1 h (needs a big corpus) |

Parameter counts are measured with the default 2048-token vocabulary; timings
are for 2000 steps on the 9.6 MB generated corpus (capped by `--max-epochs 8`).
CPU numbers are from a 2-core machine, so a typical 8-core laptop is ~4x
quicker. On CPU use `--preset nano` if you want an answer over lunch rather
than after it.

The hyper-parameters that ship with each preset live in `PRESET_TRAIN`
(`orbit_gpt/config.py`) — learning rate, warmup, weight decay, gradient
clipping, batch size and gradient accumulation — so the whole training recipe
is one dict you can edit.

Everything is configurable:

```bash
python train.py --preset micro --n-layer 8 --n-embd 256 --block-size 256 \
                --batch-size 48 --max-steps 4000 --lr 3e-3
```

Useful flags: `--tokenizer char` (baseline), `--vocab-size`,
`--max-epochs 8` (cap the number of passes, so a small corpus is not memorised),
`--compile` (torch.compile, Linux/GPU), `--dtype bf16|fp16|fp32`,
`--resume auto`, `--eval-interval`, `--save-interval`, `--seed`.

---

## 📚 Built-in corpora

| name | what it is |
|------|------------|
| `conversation` | **SFT stage 2** — ~48,000 generated `User:` / `Assistant:` exchanges (7 MB) built on the fly, no download |
| `prose` | **SFT stage 1** — 3.6 MB of plain paragraphs built from the same knowledge, no download |
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
| best chat answers | `prose` → `conversation` (the default two-stage run) | real sentences first, then the chat format |
| chat only | `conversation` | faster to train, weaker sentences |
| writer *and* chat | `shakespeare,conversation` | funnier, less accurate |
| best prose | `shakespeare` or a big book | fluent-ish pastiche, no chat ability |
| your own data | `./notes.txt,conversation` | your text, still able to chat |

---

## 🗣️ Why it answers instead of reciting

A 137-exchange corpus is *memorisable*: train on it and the model answers those
137 questions beautifully and everything else by falling back on the nearest
line it learned by heart. Three things stop that:

1. **A generated corpus, not a written one.** `conversation` is built by
   `orbit_gpt/corpora/conversation.py` at load time: ~74,000 exchanges (9.6 MB,
   2.7M tokens with the default vocabulary), each
   intent (greetings, identity, definitions, how-tos, chitchat, refusals,
   advice, creative writing, meta) written with many question phrasings and 3-4
   answer variants, sampled and shuffled deterministically. Every arithmetic
   and unit-conversion answer is *computed*, so the training data is never
   wrong. There is no canonical sentence for the model to memorise. About 3,000
   of the exchanges are "here are some web results, answer the question"
   examples in the exact layout `orbit_gpt/search.py` produces, which is what
   lets it read search results at inference time.
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

## 💾 Checkpoints (no Google Drive)

Checkpoints are written to `checkpoints/<name>` — inside the repository when
you run from a clone, `/content/checkpoints/orbit` when the single-file build
runs standalone in Colab. **Nothing is ever mounted or written to Google
Drive**, and `checkpoints/` is in `.gitignore` so a run can never push weights.

```text
checkpoints/orbit/
├── model.pt              # best val loss - what inference loads
├── model-latest.pt       # freshest save  - what --resume auto loads
├── model-step000750.pt   # periodic snapshots (newest 2 kept)
├── tokenizer.json        # the tokenizer the weights were trained with
└── train_config.json     # model + training config, for reproducibility
```

`model.pt` is written whenever validation loss improves, `model-latest.pt`
every `--save-interval` steps (250 by default) and at the end of the run, so a
run that is interrupted can always be picked up again.

```bash
# resume from the newest checkpoint, for 1000 more steps
python train.py --resume auto --max-steps 3000
# or from a specific file
python train.py --resume checkpoints/orbit/model-step000750.pt
```

Resuming restores the step counter, best loss and history but keeps *your*
`--max-steps` / `--lr`, so you can carry on with a different budget. If the
checkpoint's vocabulary does not match the corpus, training says so and starts
clean instead of crashing on a shape mismatch.

Checkpoints are ~19 MB for `micro` (4.8M params × 4 bytes) — small enough to
keep around, but too big to sprinkle through git history, so they stay on disk.
If you really want one in the repo: `python train.py --push-checkpoint`
(refuses anything over `--push-size-limit` MB, and needs working git
credentials, which Colab does not have).

---

## 🌐 Web search (optional)

A 5M-parameter model only knows its training text. For anything time-sensitive
it can read a few search results first:

```text
You: What are the latest developments in tiny language models?

[Web search enabled]
Searching for: latest developments in tiny language models
Found 5 relevant results.
Generating answer...

Orbit: ...
```

* **Off by default for ordinary questions.** `orbit_gpt/search.py` only
  searches when the message asks for current information (latest / today /
  2026 / price / weather / who won / "search for ..."). "hello", "who made
  you?" and "explain recursion" never touch the network.
* **Configurable.** `USE_WEB_SEARCH = True` at the top of
  `orbit_gpt/search.py`, `use_web_search=True` in the Colab `CONFIG` block,
  `--no-search` on the command line, `/search on|off` in the chat, or
  `ORBIT_WEB_SEARCH=0` in the environment.
* **Bounded.** 5 results by default, ~1200 characters in total, hard-capped to
  leave room for the question and the answer inside the context window.
* **Untrusted.** Snippets are cleaned, truncated and fenced as *data*;
  "ignore previous instructions"-style text is stripped and anything that
  could open a fake `User:`/`Assistant:` turn is neutralised.
* **Never fatal.** No package, no network, blocked, timed out or empty —
  the reason is printed and the model answers normally.
* **Modular.** `set_provider(fn)` swaps in any backend
  (`orbit_gpt.search.PROVIDERS` tries `ddgs`, then `duckduckgo_search`, then
  `googlesearch-python`).

```bash
pip install ddgs            # the only extra dependency, and only for search
python generate.py --checkpoint checkpoints/orbit --chat           # search on
python generate.py --checkpoint checkpoints/orbit --chat --no-search
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
│   ├── export.py      # ONNX export (+ 8-bit quantisation) for the local app
│   ├── skills.py      # deterministic arithmetic / unit conversions
│   ├── search.py      # optional web search (pluggable provider)
│   ├── checkpoints.py # where checkpoints live + resume resolution
│   └── corpora/
│       ├── conversation.py         # the generated dialogue corpus (SFT stage 2)
│       ├── prose.py                # the generated prose corpus (SFT stage 1)
│       └── orbit_assistant.txt     # small hand-written exchange corpus
├── train.py           # CLI: train a model (--init-from = fine-tune)
├── generate.py        # CLI: sample / chat
├── orbit_app.py       # desktop/terminal chat app: ONNX + onnxruntime only
├── colab/
│   ├── orbit_gpt_colab.py         # self-contained single file for Colab
│   ├── OrbitGPT_Colab.ipynb       # ready-made notebook
│   └── OrbitGPT_2_Cells.ipynb     # the two-cell quickstart
├── checkpoints/                   # gitignored: model.pt, model-latest.pt, ...
├── exports/                       # gitignored: the ONNX app bundle
├── tools/build_colab.py           # regenerates the single file from the package
├── tools/colab_footer.py          #   ...the Colab entry point it appends
└── tests/                         # 59 fast tests, no pytest needed
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
* **Want the old, tiny model?** `--preset nano` — 0.8M params, still fine for
  "does the pipeline work" runs.

Run the tests with `python tests/test_smoke.py`.

---

## 🎯 Honest expectations

OrbitGPT is a *toy* by modern standards — it is the "nanoGPT on Shakespeare"
scale of model, trained for minutes on a few hundred kilobytes of text. It will produce
recognisable, often charming, frequently wrong text.

The default model is `micro`: 4.8M parameters, 384-token context — about 6x
the old 0.8M default. It is still a toy. **On memorising:** a model this size
trained on a small hand-written corpus just recites it. That is why the default corpus is *generated* (see below) —
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
