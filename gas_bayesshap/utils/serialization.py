"""Atomic, reproducible serialization helpers.

All on-disk writes used by the checkpoint layer are atomic:

    temp file -> flush -> fsync -> atomic rename

so that a partially written checkpoint is never treated as valid.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # directory fsync is best-effort (e.g. some filesystems)


def write_json_atomic(path: os.PathLike, obj: Any, sort_keys: bool = True) -> Path:
    """Atomically write ``obj`` as pretty JSON to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, sort_keys=sort_keys, indent=2, default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_json(path: os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"__np__": True, "dtype": obj.dtype.str, "shape": list(obj.shape), "data": obj.tolist()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def write_npz_atomic(path: os.PathLike, **arrays: np.ndarray) -> Path:
    """Atomically write a ``.npz`` bundle of named arrays.

    Note: ``np.savez_compressed`` appends ``.npz`` to a string path that does
    not end in ``.npz``, so the temp name is given the suffix explicitly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = tmp + ".npz"  # savez_compressed appends .npz to the given name
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_npz(path: os.PathLike) -> "np.lib.npyio.NpzFile":
    return np.load(path, allow_pickle=False)


def write_text_atomic(path: os.PathLike, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def read_text(path: os.PathLike) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def ensure_dir(path: os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def jsonable(value: Any) -> Any:
    """Recursively convert numpy types to JSON-safe python types."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def optional_json_default(obj: Any) -> Optional[Any]:
    """Default hook that makes numpy arrays JSON-safe."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))
