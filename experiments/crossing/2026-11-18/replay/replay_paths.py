"""Resolve the checkout and experiment archive without relying on a particular home directory."""
import json
import os
from pathlib import Path

REPO = Path(os.environ.get("VISFLY_REPO_ROOT", Path(__file__).resolve().parents[4])).expanduser().resolve()


def archive_root():
    root = os.environ.get("VISFLY_EXPERIMENT_ARCHIVE")
    if not root:
        raise SystemExit("Set VISFLY_EXPERIMENT_ARCHIVE to the parent of crossing-corrected-20261118.")
    return Path(root).expanduser().resolve()


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
