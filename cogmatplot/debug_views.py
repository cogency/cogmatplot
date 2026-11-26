"""OpenCV named-window grid for live debug images.

Pop-out only. Inline notebooks should use ``FigureManager.imshow``.
Publish calls are no-ops until :func:`init_debug_views`.

    from cogmatplot.debug_views import init_debug_views, debug_image, render_debug_views

    init_debug_views(cols=4, monitor=1)
    debug_image("people.mask", mask)
    debug_image("people.depth", depth)
    render_debug_views()
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Optional, Tuple

from cogmatplot.images import prepare_bgr
from cogmatplot.layout import grid_positions

log = logging.getLogger(__name__)

_viewer: Optional["DebugViews"] = None


class DebugViews:
    """OpenCV namedWindow grid for named debug image feeds."""

    def __init__(
        self,
        cols: int = 4,
        monitor: int = 0,
        size: Optional[Tuple[int, int]] = None,
        title_offset: int = 30,
        taskbar_offset: int = 45,
        enabled: bool = True,
        wait_key_ms: int = 1,
    ):
        self.cols = max(1, int(cols))
        self.monitor = int(monitor)
        self.size = size
        self.title_offset = int(title_offset)
        self.taskbar_offset = int(taskbar_offset)
        self.enabled = bool(enabled)
        self.wait_key_ms = int(wait_key_ms)
        self._feeds: OrderedDict[str, Optional[Tuple[Any, dict]]] = OrderedDict()
        self._is_setup = False

    def configure(
        self,
        cols: Optional[int] = None,
        monitor: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
        title_offset: Optional[int] = None,
        taskbar_offset: Optional[int] = None,
        enabled: Optional[bool] = None,
        wait_key_ms: Optional[int] = None,
        relayout: bool = True,
    ) -> None:
        if cols is not None:
            self.cols = max(1, int(cols))
        if monitor is not None:
            self.monitor = int(monitor)
        if size is not None:
            self.size = size
        if title_offset is not None:
            self.title_offset = int(title_offset)
        if taskbar_offset is not None:
            self.taskbar_offset = int(taskbar_offset)
        if enabled is not None:
            self.enabled = bool(enabled)
        if wait_key_ms is not None:
            self.wait_key_ms = int(wait_key_ms)
        if relayout:
            self._is_setup = False

    def publish(self, name: str, image, **kwargs) -> None:
        if not self.enabled:
            return
        if not name:
            raise ValueError("debug feed name must be non-empty")
        if name not in self._feeds:
            self._is_setup = False
        self._feeds[name] = (image, kwargs)

    def clear_images(self) -> None:
        for name in list(self._feeds.keys()):
            self._feeds[name] = None

    def clear(self) -> None:
        self._feeds.clear()
        self._is_setup = False

    def feed_names(self):
        return list(self._feeds.keys())

    def render(self, wait_key_ms: Optional[int] = None) -> None:
        if not self.enabled or not self._feeds:
            return
        import cv2

        if not self._is_setup:
            self._setup_windows()
        delay = self.wait_key_ms if wait_key_ms is None else int(wait_key_ms)
        try:
            for name, payload in self._feeds.items():
                if payload is None:
                    continue
                image, kwargs = payload
                display = prepare_bgr(image, **kwargs)
                if display is None:
                    continue
                cv2.imshow(name, display)
            cv2.waitKey(delay)
        except Exception:
            log.debug("debug view render failed", exc_info=True)

    def close(self) -> None:
        import cv2

        for name in list(self._feeds.keys()):
            try:
                cv2.destroyWindow(name)
            except Exception:
                pass
        self._feeds.clear()
        self._is_setup = False

    def _setup_windows(self) -> None:
        import cv2

        positions = grid_positions(
            len(self._feeds),
            self.cols,
            self.monitor,
            cell_size=self.size,
            title=self.title_offset,
            taskbar=self.taskbar_offset,
        )
        from cogmatplot.place import place_cv_window

        for name, (xpos, ypos, dw, dh) in zip(self._feeds, positions):
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            place_cv_window(name, xpos, ypos, dw, dh)
        self._is_setup = True
        if positions:
            _x, _y, dw, dh = positions[0]
        else:
            dw = dh = 0
        log.info(
            "DebugViews ready: %d feeds, cols=%d, monitor=%d, cell=%dx%d",
            len(self._feeds), self.cols, self.monitor, dh, dw,
        )


def init_debug_views(
    cols: int = 4,
    monitor: int = 0,
    size: Optional[Tuple[int, int]] = None,
    title_offset: int = 30,
    taskbar_offset: int = 45,
    enabled: bool = True,
    wait_key_ms: int = 1,
    recreate: bool = False,
) -> DebugViews:
    """Bootstrap the process-wide debug viewer. Pass ``recreate=True`` to replace it."""
    global _viewer
    if _viewer is not None and not recreate:
        _viewer.configure(
            cols=cols, monitor=monitor, size=size, title_offset=title_offset,
            taskbar_offset=taskbar_offset, enabled=enabled, wait_key_ms=wait_key_ms,
        )
        return _viewer
    if _viewer is not None and recreate:
        try:
            _viewer.close()
        except Exception:
            pass
    _viewer = DebugViews(
        cols=cols, monitor=monitor, size=size, title_offset=title_offset,
        taskbar_offset=taskbar_offset, enabled=enabled, wait_key_ms=wait_key_ms,
    )
    return _viewer


def get_debug_views() -> Optional[DebugViews]:
    return _viewer


def is_debug_views_active() -> bool:
    return _viewer is not None and _viewer.enabled


def debug_image(name: str, image, auto_render: bool = False, **kwargs) -> None:
    """Publish ``image`` on the singleton. No-op until :func:`init_debug_views`."""
    if _viewer is None or not _viewer.enabled:
        return
    _viewer.publish(name, image, **kwargs)
    if auto_render:
        _viewer.render()


def render_debug_views(wait_key_ms: Optional[int] = None) -> None:
    if _viewer is None or not _viewer.enabled:
        return
    _viewer.render(wait_key_ms=wait_key_ms)


def clear_debug_images() -> None:
    if _viewer is not None:
        _viewer.clear_images()


def close_debug_views() -> None:
    global _viewer
    if _viewer is not None:
        try:
            _viewer.close()
        finally:
            _viewer = None
