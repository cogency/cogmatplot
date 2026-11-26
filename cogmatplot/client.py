"""Send FigureManager calls to a cogmatplot server.

Method names match the in-process manager. ``figure3`` returns a proxy whose
``scatter3``, ``view_xy``, and ``show_grid`` calls are forwarded.
"""

from __future__ import annotations

import time

import numpy as np
import zmq

from cogmatplot.api import METHODS, STREAM_METHODS
from cogmatplot.ply import as_ply, is_cloud
from cogmatplot.wire import encode


def _cloud_arrays(points, colors):
    """Send plain arrays, not loader structures the server may not import."""
    if not is_cloud(points):
        return points, colors
    ply = as_ply(points)
    return ply.points, ply.colors if colors is None else colors


def _mesh_arrays(points, faces, colors):
    if faces is not None:
        return points, faces, colors
    ply = as_ply(points)
    return ply.points, ply.faces, ply.colors if colors is None else colors


def _remote(name):
    def method(self, *args, **kwargs):
        return self._call(name, *args, **kwargs)

    method.__name__ = name
    method.__doc__ = f"Forward {name} to the figure server."
    return method


class RemoteCloud:
    """Stand-in for :class:`cogmatplot.view3.Fig3d` on the client side."""

    def __init__(self, client, row, col):
        self._client = client
        self.row = row
        self.col = col

    def scatter3(self, points, color=None, colors=None, point_size=3, cmap=None, hold=True, **kwargs):
        self._client.scatter3(
            self.row, self.col, points,
            color=color, colors=colors, point_size=point_size, cmap=cmap, hold=hold, **kwargs,
        )
        return self

    def view(self, which="xy"):
        self._client.view3(self.row, self.col, which)
        return self

    def view_xy(self):
        return self.view("xy")

    def view_xz(self):
        return self.view("xz")

    def view_yz(self):
        return self.view("yz")

    def view_isometric(self):
        return self.view("iso")

    def grid(self, on=True):
        self._client.grid3(self.row, self.col, on)
        return self

    def show_grid(self):
        return self.grid(True)

    def clear(self):
        self._client.clear3(self.row, self.col)
        return self

    def plot_mesh(self, points, faces=None, color=None, colors=None, style="surface", hold=True, **kwargs):
        self._client.plot_mesh(
            self.row, self.col, points, faces,
            color=color, colors=colors, style=style, hold=hold, **kwargs,
        )
        return self

    def plot_box(self, box, **kwargs):
        self._client.plot_box(self.row, self.col, box, **kwargs)
        return self


class PlotClient:
    """Debugger-side handle. Construct with ``FigureManager.connect`` too."""

    def __init__(self, host="localhost", port=5555):
        self.host = host
        self.port = int(port)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 16)
        self._socket.connect(f"tcp://{host}:{self.port}")
        self._events = None

    def _call(self, method, *args, **kwargs):
        kwargs.pop("flush", None)
        frames = encode(method, args, kwargs)
        wait = 0.25 if method in STREAM_METHODS else 2.0
        deadline = time.perf_counter() + wait
        while True:
            try:
                self._socket.send_multipart(frames, flags=zmq.NOBLOCK)
                return
            except zmq.Again:
                if time.perf_counter() > deadline:
                    if method not in STREAM_METHODS:
                        self._socket.send_multipart(frames)
                    return
                time.sleep(0.01)

    def plot(self, row, col, x, y, fmt=None, hold=False, flush=True, **kwargs):
        """``fmt`` defaults to ``'.'`` so existing debugger scripts keep their markers."""
        del flush
        self._call("plot", row, col, x, y, "." if fmt is None else fmt, hold=hold, **kwargs)

    def figure3(self, row, col, title="Fig3d", **kwargs):
        self._call("figure3", row, col, title=title, **kwargs)
        return RemoteCloud(self, row, col)

    def scatter3(self, row, col, points, color=None, colors=None, **kwargs):
        points, colors = _cloud_arrays(points, colors)
        self._call("scatter3", row, col, points, color=color, colors=colors, **kwargs)
        return RemoteCloud(self, row, col)

    def plot3(self, row, col, x, y=None, z=None, fmt=".-", colors=None, **kwargs):
        if z is None and (y is None or isinstance(y, str)):
            x, colors = _cloud_arrays(x, colors)
        self._call("plot3", row, col, x, y, z, fmt, colors=colors, **kwargs)

    def plot_mesh(self, row, col, points, faces=None, colors=None, **kwargs):
        points, faces, colors = _mesh_arrays(points, faces, colors)
        self._call("plot_mesh", row, col, points, faces, colors=colors, **kwargs)
        return RemoteCloud(self, row, col)

    def plot_box(self, row, col, box, **kwargs):
        self._call("plot_box", row, col, np.asarray(box, dtype=float), **kwargs)
        return RemoteCloud(self, row, col)

    def close_all(self, rows=None, cols=None, mon=None):
        self.reset(rows, cols, mon)

    def clf_all(self):
        self.clfAll()

    def inter(self, val=1):
        self.interactive(bool(val))

    def poll_event(self, timeout=0):
        """Return the newest key event from the server, or ``None``."""
        if self._events is None:
            self._events = self._context.socket(zmq.SUB)
            self._events.setsockopt(zmq.LINGER, 0)
            self._events.setsockopt(zmq.SUBSCRIBE, b"")
            self._events.setsockopt(zmq.CONFLATE, 1)
            self._events.connect(f"tcp://{self.host}:{self.port + 1}")
        if self._events.poll(int(timeout)):
            return self._events.recv_json()
        return None

    def close(self):
        for sock in (self._socket, self._events):
            if sock is None:
                continue
            try:
                sock.close(0)
            except Exception:
                pass
        try:
            self._context.term()
        except Exception:
            pass


for _name in METHODS:
    if not hasattr(PlotClient, _name):
        setattr(PlotClient, _name, _remote(_name))
