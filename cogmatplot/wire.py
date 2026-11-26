"""Localhost message codec.

Command headers are pickled. ndarray payloads travel as raw buffers.
The server collapses back-to-back image updates for the same cell and keeps
every other command in order.
"""

from __future__ import annotations

import pickle
import threading
from typing import List

import numpy as np

from cogmatplot.api import STREAM_METHODS

_TOKEN = "__cog_ndarray__"


def encode(method: str, args, kwargs) -> List[bytes]:
    buffers: List[bytes] = []
    payload = {
        "method": method,
        "args": _pack(args, buffers),
        "kwargs": _pack(kwargs, buffers),
    }
    frames = [pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)]
    frames.extend(buffers)
    return frames


def decode(frames):
    payload = pickle.loads(frames[0])
    buffers = list(frames[1:])
    return payload["method"], _unpack(payload["args"], buffers), _unpack(payload["kwargs"], buffers)


def _pack(obj, buffers: List[bytes]):
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        buffers.append(arr.tobytes())
        return {_TOKEN: len(buffers) - 1, "shape": list(arr.shape), "dtype": str(arr.dtype)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _pack(v, buffers) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_pack(v, buffers) for v in obj)
    if isinstance(obj, list):
        return [_pack(v, buffers) for v in obj]
    return obj


def _unpack(obj, buffers):
    if isinstance(obj, dict) and _TOKEN in obj and len(obj) == 3:
        arr = np.frombuffer(buffers[obj[_TOKEN]], dtype=np.dtype(obj["dtype"]))
        return arr.reshape(obj["shape"]).copy()
    if isinstance(obj, dict):
        return {k: _unpack(v, buffers) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_unpack(v, buffers) for v in obj)
    if isinstance(obj, list):
        return [_unpack(v, buffers) for v in obj]
    return obj


class Mailbox:
    """Ordered commands. Consecutive image updates for one cell collapse."""

    def __init__(self):
        self._lock = threading.Lock()
        self._queue = []

    def push(self, method: str, args, kwargs) -> None:
        item = (method, args, kwargs)
        with self._lock:
            if (
                method in STREAM_METHODS
                and len(args) >= 2
                and self._queue
                and self._queue[-1][0] == method
                and tuple(self._queue[-1][1][:2]) == tuple(args[:2])
            ):
                self._queue[-1] = item
                return
            self._queue.append(item)

    def drain(self):
        with self._lock:
            queued = self._queue
            self._queue = []
        return queued
