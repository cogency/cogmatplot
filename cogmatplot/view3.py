"""3D point-cloud views.

Pop-out mode uses a PyVista Qt window (orbit, zoom, grid). Inline mode keeps
the same ``scatter3`` calls and renders a static snapshot, or a matplotlib
3D scatter when PyVista is not available.

Clouds are ``(N, 3)`` xyz. Pass ``colors`` as ``(N, 3)`` rgb, or ``color``
as one solid color. An object with ``.points`` and ``.colors`` (Open3D) is
accepted.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cogmatplot.layout import gridfig_box
from cogmatplot.mode import current_mode

_VIEWS = {
    "xy": "view_xy",
    "xz": "view_xz",
    "yz": "view_yz",
    "iso": "view_isometric",
    "isometric": "view_isometric",
}


def coerce_cloud(points, colors=None):
    """Return ``(pts Nx3 float, colors or None)``."""
    if hasattr(points, "points"):
        raw = points
        pts = np.asarray(raw.points)
        if colors is None:
            raw_colors = getattr(raw, "colors", None)
            if raw_colors is not None:
                raw_colors = np.asarray(raw_colors)
                if getattr(raw_colors, "size", 0):
                    colors = raw_colors
    else:
        pts = np.asarray(points, dtype=float)

    if pts.ndim != 2:
        raise ValueError("point cloud must have shape (N, 3)")
    if pts.shape[1] != 3 and pts.shape[0] == 3:
        pts = pts.T
    if pts.shape[1] != 3:
        raise ValueError("point cloud must have shape (N, 3)")

    if colors is not None:
        colors = np.asarray(colors)
        if colors.size == 0:
            colors = None
        elif (
            colors.ndim == 2
            and colors.shape[0] == 3
            and colors.shape[1] == len(pts)
            and colors.shape[1] != 3
        ):
            colors = colors.T
    return np.ascontiguousarray(pts, dtype=float), colors


def add_cloud(plotter, points, color=None, colors=None, point_size=3, cmap=None, **kwargs):
    """Add one xyz or xyz+rgb cloud to a PyVista plotter."""
    pts, colors = coerce_cloud(points, colors)
    point_size = kwargs.pop("point_size", point_size)
    if colors is None:
        plotter.add_points(pts, color=color or "silver", point_size=point_size, **kwargs)
        return
    col = np.asarray(colors)
    if col.ndim == 2 and col.shape[1] == 3:
        rgb = col.astype(np.float64)
        peak = np.nanmax(rgb) if rgb.size else 0
        if peak > 1.0:
            rgb = rgb / 255.0
        plotter.add_points(pts, scalars=np.ascontiguousarray(rgb), rgb=True, point_size=point_size, **kwargs)
        return
    scalars = np.ascontiguousarray(col).reshape(-1)
    if cmap is None:
        plotter.add_points(pts, scalars=scalars, point_size=point_size, **kwargs)
    else:
        plotter.add_points(pts, scalars=scalars, cmap=cmap, point_size=point_size, **kwargs)


def add_mesh(plotter, points, faces, color=None, colors=None, style="surface", **kwargs):
    """Draw triangle faces. ``style`` is ``surface``, ``wireframe``, or ``both``."""
    import pyvista as pv

    pts = np.asarray(points, dtype=float)
    tris = np.asarray(faces, dtype=np.int64)
    if tris.ndim != 2 or tris.shape[1] < 3:
        raise ValueError("faces must have shape (M, 3)")
    tris = tris[:, :3]
    packed = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris]).ravel()
    poly = pv.PolyData(np.ascontiguousarray(pts), packed)
    style = str(style or "surface").lower()
    if style == "wireframe":
        plotter.add_mesh(poly, style="wireframe", color=color or "black", line_width=kwargs.pop("line_width", 2), **kwargs)
        return
    rgb = _vertex_rgb(colors, len(pts))
    show_edges = style in ("both", "shaded+edges", "wire")
    if rgb is not None:
        plotter.add_mesh(poly, scalars=rgb, rgb=True, show_edges=show_edges, edge_color=kwargs.pop("edge_color", "black"), **kwargs)
        return
    plotter.add_mesh(
        poly,
        color=color or "lightsteelblue",
        show_edges=show_edges,
        edge_color=kwargs.pop("edge_color", "black"),
        **kwargs,
    )


def _vertex_rgb(colors, count):
    if colors is None:
        return None
    col = np.asarray(colors)
    if col.ndim != 2 or col.shape[1] != 3 or len(col) != count:
        return None
    rgb = col.astype(np.float64)
    if rgb.size and np.nanmax(rgb) > 1.0:
        rgb = rgb / 255.0
    return np.ascontiguousarray(rgb)


class Fig3d:
    """One 3D view. ``.pl`` is the live PyVista plotter in pop-out mode."""

    def __init__(self, title="Fig3d", show=True, grid_pos=None, off_screen: Optional[bool] = None, on_change=None):
        self.title = title
        self.row = None
        self.col = None
        self._inline = current_mode() == "inline" if off_screen is None else bool(off_screen)
        self._grid_pos = grid_pos
        self._on_change = on_change
        self._layers = []
        self._view = "xy"
        self._grid = True
        self._grid_shown = False
        self._pl = None
        if not self._inline:
            self._open_qt()
            if show:
                self.show()

    @staticmethod
    def close_all():
        try:
            import pyvista as pv

            pv.close_all()
        except Exception:
            pass

    def place_in_grid(self, row, col, nrows, ncols, screen_index=0, margin=0):
        """Move the pop-out window into a monitor cell.

        ``grid_pos`` is ``(row, col, nrows, ncols, screen_index)``.
        """
        del margin
        if self._inline or self._pl is None:
            return
        x, y, w, h = gridfig_box(row, col, nrows, ncols, screen_index)
        from cogmatplot.place import place_qt_window

        place_qt_window(getattr(self._pl, "app_window", None), x, y, w, h)

    def show(self, grid=True, axes=True):
        if self._inline:
            self._notify()
            return self
        if self._pl is None:
            self._open_qt()
        self._pl.set_background("white")
        if axes:
            self._pl.show_axes()
        if grid:
            self.grid(True)
        self._pl.show()
        if self._grid_pos is not None:
            self.place_in_grid(*self._grid_pos)
        return self

    def close(self):
        if self._pl is not None:
            try:
                self._pl.close()
            except Exception:
                pass
            self._pl = None

    def clear(self):
        self._layers = []
        self._grid_shown = False
        if self._pl is not None:
            self._pl.clear()
            self._style_qt()
        self._notify()
        return self

    @property
    def pl(self):
        if self._pl is None and not self._inline:
            self._open_qt()
        return self._pl

    @property
    def layers(self):
        return list(self._layers)

    def scatter3(self, points, color=None, colors=None, point_size=3, cmap=None, hold=True, **kwargs):
        """Add a cloud. ``hold=False`` replaces clouds already in this view."""
        if not hold:
            self._layers = []
            self._grid_shown = False
            if self._pl is not None:
                self._pl.clear()
                self._style_qt()
        layer = {
            "points": points,
            "color": color,
            "colors": colors,
            "point_size": point_size,
            "cmap": cmap,
            "kwargs": kwargs,
        }
        self._layers.append(layer)
        if self._pl is not None:
            self._draw_layer(self._pl, layer)
        self._notify()
        return self

    def plot_mesh(self, points, faces, color=None, colors=None, style="surface", hold=True, **kwargs):
        """Add a triangle mesh. ``style`` is ``surface``, ``wireframe``, or ``both``."""
        if not hold:
            self._layers = []
            self._grid_shown = False
            if self._pl is not None:
                self._pl.clear()
                self._style_qt()
        layer = {
            "kind": "mesh",
            "points": points,
            "faces": faces,
            "color": color,
            "colors": colors,
            "style": style,
            "kwargs": kwargs,
        }
        self._layers.append(layer)
        if self._pl is not None:
            self._draw_layer(self._pl, layer)
        self._notify()
        return self

    def view(self, which="xy"):
        self._view = which
        if self._pl is not None:
            self._apply_view(self._pl)
        self._notify()
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
        self._grid = bool(on) and str(on).lower() != "off"
        if self._pl is not None and self._grid and not self._grid_shown:
            self._pl.show_grid()
            self._grid_shown = True
        self._notify()
        return self

    def show_grid(self):
        return self.grid(True)

    def snapshot(self, size=(720, 540)):
        """Render the current clouds to an RGB image."""
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True, window_size=tuple(size))
        try:
            plotter.set_background("white")
            for layer in self._layers:
                self._draw_layer(plotter, layer)
            plotter.show_axes()
            if self._grid:
                plotter.show_grid()
            self._apply_view(plotter)
            return plotter.screenshot(return_img=True)
        finally:
            plotter.close()

    def _open_qt(self):
        from pyvistaqt import BackgroundPlotter

        self._pl = BackgroundPlotter(title=self.title or "Fig3d", auto_update=True)
        if self._grid_pos is not None:
            self.place_in_grid(*self._grid_pos)
        self._style_qt()
        for layer in self._layers:
            self._draw_layer(self._pl, layer)

    def _style_qt(self):
        if self._pl is None:
            return
        self._pl.set_background("white")
        self._pl.show_axes()
        if self._grid and not self._grid_shown:
            self._pl.show_grid()
            self._grid_shown = True
        self._apply_view(self._pl)

    def _draw_layer(self, plotter, layer):
        if layer.get("kind") == "mesh":
            add_mesh(
                plotter,
                layer["points"],
                layer["faces"],
                color=layer.get("color"),
                colors=layer.get("colors"),
                style=layer.get("style") or "surface",
                **(layer.get("kwargs") or {}),
            )
            return
        add_cloud(
            plotter,
            layer["points"],
            color=layer["color"],
            colors=layer["colors"],
            point_size=layer["point_size"],
            cmap=layer["cmap"],
            **layer["kwargs"],
        )

    def _apply_view(self, plotter):
        name = _VIEWS.get(str(self._view).lower(), "view_xy")
        getattr(plotter, name)()

    def _notify(self):
        if self._on_change is not None:
            self._on_change(self)
