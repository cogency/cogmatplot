"""Monitor grids shared by matplotlib, OpenCV, and PyVista windows."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple


def _monitors():
    from screeninfo import get_monitors

    return get_monitors()


def pick_monitor(index: int = 0):
    try:
        monitors = list(_monitors() or [])
    except Exception:
        monitors = []
    if not monitors:
        return SimpleNamespace(x=0, y=0, width=1600, height=900, name="default")
    i = min(max(int(index), 0), len(monitors) - 1)
    return monitors[i]


def gridfig_box(
    row: int,
    col: int,
    rows: int,
    cols: int,
    monitor: int = 0,
    colpad: int = 50,
    rowpad: int = 50,
    rowend: int = 35,
    barpad: int = 30,
) -> Tuple[int, int, int, int]:
    """Cell rectangle for one tile. Returns ``(x, y, w, h)`` in screen pixels."""
    screen = pick_monitor(monitor)
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    row = int(max(0, min(rows - 1, row)))
    col = int(max(0, min(cols - 1, col)))
    fw = (screen.width // cols) - colpad
    fh = ((screen.height - rowend) // rows) - rowpad
    x = int(screen.x + col * fw + (col + 1) * colpad / 2)
    y = int(screen.y + row * fh + (row + 1) * rowpad / 2)
    return x, y, int(max(fw, 80)), int(max(fh - barpad, 80))


def grid_positions(
    count: int,
    cols: int,
    monitor: int = 0,
    cell_size: Optional[Sequence[int]] = None,
    title: int = 30,
    taskbar: int = 45,
) -> List[Tuple[int, int, int, int]]:
    """Named-window packing for debug streams.

    ``cell_size`` is ``(height, width)`` when cells are fixed.
    Each item is ``(x, y, w, h)``.
    """
    screen = pick_monitor(monitor)
    width = int(screen.width)
    height = int(screen.height) - int(taskbar)
    num_cols = max(1, int(cols))
    n = max(1, int(count))
    if cell_size is not None:
        dh, dw = int(cell_size[0]), int(cell_size[1])
    else:
        dw = math.floor(width / float(num_cols))
        num_rows = max(1, math.ceil(n / float(num_cols)))
        if num_rows * dw > height:
            dw = math.floor(height / float(num_rows))
            dh = math.floor((height - title * num_rows) / float(num_rows))
        else:
            dh = dw
    positions = []
    for i in range(int(count)):
        r, c = divmod(i, num_cols)
        positions.append((
            int(dw * c + screen.x),
            int((dh + title) * r + screen.y),
            int(dw),
            int(dh),
        ))
    return positions
