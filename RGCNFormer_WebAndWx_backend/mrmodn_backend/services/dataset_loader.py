"""Cached loaders for the bundled NextGen datasets."""
from functools import lru_cache
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "nextgen"


@lru_cache(maxsize=8)
def load_json(name: str):
    path = (DATA_DIR / name).resolve()
    if path.parent != DATA_DIR.resolve():
        raise ValueError("invalid dataset name")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

