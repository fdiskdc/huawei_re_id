"""Load the backend's visible ``backend.env`` configuration file."""

import os
from pathlib import Path


DEFAULT_CONFIG_FILE = Path(__file__).resolve().with_name("backend.env")


def load_backend_config() -> Path:
    """Load backend.env without overriding variables supplied by the runtime."""
    configured_path = os.getenv("BACKEND_CONFIG_FILE")
    path = Path(configured_path).expanduser() if configured_path else DEFAULT_CONFIG_FILE
    if not path.is_absolute():
        path = DEFAULT_CONFIG_FILE.parent / path
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Backend configuration file not found: {path}")

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid configuration at {path}:{line_number}")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"Invalid configuration key at {path}:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, os.path.expandvars(value))

    return path
