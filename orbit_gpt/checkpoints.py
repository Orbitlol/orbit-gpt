"""Where checkpoints live and how to find the most recent one.

Rules, so there are no surprises:

* **Google Drive is never touched.** Nothing here mounts or writes to Drive.
* In Colab (``/content`` exists and there is no repo) checkpoints go to
  ``/content/checkpoints/<name>``.
* Inside a clone of this repository they go to ``<repo>/checkpoints/<name>``
  (which on Colab would be e.g. ``/content/orbit-gpt/checkpoints/orbit``).
* Otherwise: ``./checkpoints/<name>``.

``checkpoints/`` is in ``.gitignore``, so a run can never accidentally push a
hundred megabytes of weights.

Layout inside the directory::

    model.pt              best validation loss so far  (what inference loads)
    model-latest.pt       most recent periodic save     (what --resume loads)
    model-step000500.pt   periodic saves, newest 3 kept
    tokenizer.json        the tokenizer that goes with the weights
    train_config.json     model + training config, for reproducibility
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

__all__ = [
    "DEFAULT_NAME",
    "BEST_NAME",
    "LATEST_NAME",
    "default_checkpoint_dir",
    "in_colab",
    "repo_root",
    "step_name",
    "latest_checkpoint",
    "resolve_resume",
    "keep_last_n",
    "human_size",
]

DEFAULT_NAME = "orbit"
BEST_NAME = "model.pt"          # best val loss - loaded by inference
LATEST_NAME = "model-latest.pt"  # most recent save - loaded by --resume
KEEP_STEP_CHECKPOINTS = 2       # how many periodic snapshots to keep


def in_colab() -> bool:
    """True when running inside Google Colab (no Drive mounting involved)."""
    return Path("/content").exists()


def repo_root() -> Optional[Path]:
    """The checkout this file belongs to, or ``None`` (single-file builds)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
        if (parent / "train.py").exists() and (parent / "orbit_gpt").is_dir():
            return parent
    return None


def default_checkpoint_dir(name: str = DEFAULT_NAME) -> Path:
    """Pick a checkpoint directory that needs no manual path editing."""
    root = repo_root()
    if root is not None:
        return root / "checkpoints" / name
    if in_colab():
        return Path("/content/checkpoints") / name
    return Path.cwd() / "checkpoints" / name


def step_name(step: int) -> str:
    return f"model-step{step:06d}.pt"


def _step_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def latest_checkpoint(directory) -> Optional[Path]:
    """The newest usable checkpoint in ``directory`` (``None`` if empty).

    Preference order: the periodic "latest" save (it has the freshest step
    counter and optimizer-shaped state), then "best", then any step snapshot.
    """
    d = Path(directory)
    if not d.is_dir():
        return None
    for candidate in (d / LATEST_NAME, d / BEST_NAME):
        if candidate.exists():
            return candidate
    snapshots = sorted(d.glob("model-step*.pt"), key=_step_number)
    return snapshots[-1] if snapshots else None


def resolve_resume(out_dir, resume: str = "") -> Optional[Path]:
    """Turn the ``--resume`` argument into a path (or ``None``).

    ``""``, ``"auto"``, ``"latest"`` and ``"1"`` all mean "pick the newest
    checkpoint in ``out_dir``"; anything else is used as given, and a directory
    is resolved to the newest checkpoint inside it.
    """
    if not resume or str(resume).strip().lower() in ("auto", "latest", "1", "true", "yes", "on"):
        return latest_checkpoint(out_dir)
    path = Path(str(resume)).expanduser()
    if path.is_dir():
        return latest_checkpoint(path)
    return path if path.exists() else None


def keep_last_n(directory, pattern: str = "model-step*.pt", n: int = KEEP_STEP_CHECKPOINTS) -> None:
    """Delete older periodic snapshots so a long run cannot fill the disk."""
    d = Path(directory)
    snapshots = sorted(d.glob(pattern), key=_step_number)
    for stale in snapshots[:-n] if n > 0 else snapshots:
        try:
            stale.unlink()
        except OSError:
            pass


def human_size(path) -> str:
    """``'3.4 MB'`` - used when reporting whether a checkpoint is committable."""
    size = Path(path).stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
