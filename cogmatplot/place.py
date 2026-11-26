"""Put Qt and OpenCV windows on the same screen rectangle.

Matplotlib places a window with ``move``, which anchors the visible frame,
title bar included. ``setGeometry`` anchors the client area under the title
bar, so a PyVista window sits too high. OpenCV's Win32 ``moveWindow`` anchors
the outer rectangle, which includes a thin invisible border, so that window
sits a few pixels to the right. These helpers land all three on the frame
matplotlib uses.
"""

from __future__ import annotations

import sys


def place_qt_window(window, x, y, w, h) -> None:
    """Match matplotlib: frame origin at ``(x, y)``, client size ``(w, h)``."""
    if window is None or not hasattr(window, "move"):
        return
    window.move(int(x), int(y))
    if hasattr(window, "resize"):
        window.resize(int(w), int(h))


def place_cv_window(name: str, x, y, w, h) -> None:
    """Move an OpenCV window so its visible frame starts at ``(x, y)``."""
    import cv2

    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.resizeWindow(name, max(w, 1), max(h, 1))
    cv2.moveWindow(name, x, y)
    cv2.waitKey(1)
    origin = _cv_visible_origin(name)
    if origin is None:
        return
    dx = origin[0] - x
    dy = origin[1] - y
    if dx or dy:
        cv2.moveWindow(name, x - dx, y - dy)


def _cv_visible_origin(name: str):
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
    hwnd = user32.FindWindowW(None, name)
    if not hwnd:
        return None
    rect = wintypes.RECT()
    try:
        dwmapi = ctypes.windll.dwmapi
        # DWMWA_EXTENDED_FRAME_BOUNDS: the visible frame, without the invisible border.
        hr = dwmapi.DwmGetWindowAttribute(ctypes.c_void_p(hwnd), 9, ctypes.byref(rect), ctypes.sizeof(rect))
        if hr == 0:
            return int(rect.left), int(rect.top)
    except Exception:
        pass
    if user32.GetWindowRect(ctypes.c_void_p(hwnd), ctypes.byref(rect)):
        return int(rect.left), int(rect.top)
    return None
