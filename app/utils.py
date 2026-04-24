import os
import uuid


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_temp_filename(original: str) -> str:
    safe = original.replace("/", "_").replace("\\", "_")
    return f"{uuid.uuid4().hex}_{safe}"
