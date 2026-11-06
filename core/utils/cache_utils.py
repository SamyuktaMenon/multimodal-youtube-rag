"""
Caching and serialization utilities.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def hash_text(text: str) -> str:
    """Generate MD5 hash of text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def save_json(data: Any, path: Path) -> None:
    """Save serializable data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Optional[Any]:
    """Load data from JSON file if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

