"""Environment detection, paths, and secrets that work both locally and on Google Colab."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# On Colab, data and outputs live on Google Drive so they survive runtime resets.
# Point every collaborator's Drive at the same folder (a shortcut to the shared folder
# in "My Drive" works), or override with the CELLCOUNT_STORAGE env var.
COLAB_STORAGE = Path("/content/drive/MyDrive/cell-image-counting")


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class Paths:
    root: Path      # where data/ and outputs/ live
    data: Path      # downloaded datasets
    outputs: Path   # figures, splits, checkpoints, results


def get_paths(create: bool = True) -> Paths:
    """Return the storage paths for the current environment."""
    if "CELLCOUNT_STORAGE" in os.environ:
        root = Path(os.environ["CELLCOUNT_STORAGE"]).expanduser()
    elif is_colab():
        root = COLAB_STORAGE
    else:
        root = REPO_ROOT
    paths = Paths(root=root, data=root / "data", outputs=root / "outputs")
    if create:
        paths.data.mkdir(parents=True, exist_ok=True)
        paths.outputs.mkdir(parents=True, exist_ok=True)
    return paths


def get_secret(name: str) -> str:
    """Read a secret from Colab Secrets, the environment, or the repo's .env file."""
    if is_colab():
        from google.colab import userdata

        try:
            return userdata.get(name)
        except Exception:  # SecretNotFoundError / NotebookAccessError
            pass
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    value = os.environ.get(name)
    if not value:
        raise KeyError(
            f"{name} is not set. Locally: add it to {REPO_ROOT / '.env'} "
            "(see .env.example). On Colab: add it under Secrets and enable notebook access."
        )
    return value
