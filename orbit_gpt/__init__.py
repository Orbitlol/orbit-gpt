"""OrbitGPT - a small, fast, dependency-light GPT you can train on a laptop.

The whole project only needs PyTorch (and optionally ``tqdm``).  Everything
else - tokenization (byte-level BPE), data loading, training, sampling and a
tiny chat loop - is implemented here in plain Python so the model can be
trained from scratch in a couple of minutes on a Colab GPU or a laptop CPU.

Typical use from the command line::

    python train.py --dataset shakespeare,orbit-chat:12 --preset micro
    python generate.py --checkpoint out/orbit-micro --prompt "User: Hello!\\nAssistant:"

Typical use from Python::

    from orbit_gpt import GPT, GPTConfig, BPETokenizer, Trainer, TrainConfig

__version__ = "0.1.0"
"""

from orbit_gpt.config import GPTConfig, TrainConfig, PRESETS, get_preset, list_presets
from orbit_gpt.tokenizer import Tokenizer, BPETokenizer, CharTokenizer, load_tokenizer
from orbit_gpt.model import GPT
from orbit_gpt.data import load_corpus, list_datasets
from orbit_gpt.train import Trainer, auto_device, set_seed

__all__ = [
    "GPT",
    "GPTConfig",
    "TrainConfig",
    "PRESETS",
    "get_preset",
    "list_presets",
    "Tokenizer",
    "BPETokenizer",
    "CharTokenizer",
    "load_tokenizer",
    "Trainer",
    "auto_device",
    "set_seed",
    "load_corpus",
    "list_datasets",
    "__version__",
]
