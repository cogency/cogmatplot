"""MATLAB-style plotting helpers and :class:`FigureManager`.

In-process::

    fm = FigureManager(2, 2, mon=0, pause=False)
    fm.imshow(0, 0, image)
    fm.plot(0, 1, x, y, '.-')
    fm.plot3(1, 0, x, y, z)
    cloud = fm.figure3(1, 1, title='points')
    cloud.scatter3(pts, colors=rgb)
    cloud.scatter3(tag, color='red')
    cloud.view_xy()

``mode='inline'`` draws that grid into the notebook as static figures.
``mode='qt'`` opens an interactive window per tile on ``mon``.
``FigureManager.connect(host, port)`` sends the same calls to a server.
"""

from __future__ import annotations

import warnings
import weakref

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from cogmatplot.images import prepare_bgr
from cogmatplot.layout import gridfig_box
from cogmatplot.mode import current_mode
from cogmatplot.view3 import coerce_cloud

_MANAGERS = weakref.WeakSet()


def _as_on(state) -> bool:
    if isinstance(state, str):
        return state.lower() in ("on", "true", "1", "yes")
    return bool(state)


def saveFig(fig, filename):
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    fig.savefig(filename)


def closeAll():
    """Close every figure this process opened. MATLAB ``close all``."""
    plt.ioff()
    for manager in list(_MANAGERS):
        manager.closeAll()
    plt.close("all")
    try:
        import cv2

        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        import pyvista as pv

        pv.close_all()
    except Exception:
        pass
    if current_mode() == "qt":
        plt.ion()


def clfAll():
    """Clear every open axes. MATLAB ``clf`` across figures."""
    for manager in list(_MANAGERS):
        manager.clfAll()
    for num in list(plt.get_fignums()):
        for ax in plt.figure(num).axes:
            ax.cla()


def figure(**kwargs):
    fig, ax = plt.subplots(**kwargs)
    return fig, ax


def _place_window(fig, x, y, w, h):
    manager = getattr(fig.canvas, "manager", None)
    window = getattr(manager, "window", None)
    if window is None:
        return
    from cogmatplot.place import place_qt_window

    place_qt_window(window, x, y, w, h)
    for attr in ("header_visible", "footer_visible"):
        if hasattr(fig.canvas, attr):
            setattr(fig.canvas, attr, False)


def gridfig(row=0, col=0, rows=3, cols=3, mon=0, colpad=50, rowpad=50, rowend=35, barpad=30, bar=False):
    """One matplotlib window placed in a monitor grid."""
    x, y, w, h = gridfig_box(row, col, rows, cols, mon, colpad=colpad, rowpad=rowpad, rowend=rowend, barpad=barpad)
    prev = plt.rcParams.get("toolbar")
    plt.rcParams["toolbar"] = "toolbar2" if bar else "None"
    try:
        fig, ax = figure()
    finally:
        if prev is not None:
            plt.rcParams["toolbar"] = prev
    _place_window(fig, x, y, w, h)
    try:
        fig.tight_layout()
    except Exception:
        pass
    _flush_fig(fig)
    return fig, ax


def fit(fig):
    fig.tight_layout()


def axfit(ax):
    ax.get_figure().tight_layout()


def _flush_fig(fig):
    if fig is None:
        return
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except Exception:
        pass


def _display_inline(fig):
    if fig is None or not getattr(fig, "_cog_inline", False):
        return
    try:
        fig.tight_layout()
    except Exception:
        pass
    try:
        from IPython import get_ipython
        from IPython.display import display, update_display

        if get_ipython() is None:
            raise RuntimeError("no ipython")
    except Exception:
        _flush_fig(fig)
        return
    did = f"cogmatplot-{id(fig)}"
    seen = getattr(_display_inline, "_seen", set())
    if did in seen:
        update_display(fig, display_id=did)
    else:
        display(fig, display_id=did)
        seen.add(did)
        _display_inline._seen = seen


def _cv_alive():
    for manager in list(_MANAGERS):
        for cell in manager._cells.values():
            if cell.cv_name:
                return True
    try:
        from cogmatplot.debug_views import get_debug_views

        view = get_debug_views()
        return view is not None and bool(view.feed_names())
    except Exception:
        return False


def _pump_cv():
    if not _cv_alive():
        return
    import cv2

    key = cv2.waitKey(1)
    if key in (-1, 255) or key is None:
        return
    event = {"type": "key", "key": int(key) & 0xFF, "source": "cv"}
    for manager in list(_MANAGERS):
        manager._emit(event)


def _pump_qt():
    if not any(getattr(manager, "mode", "") == "qt" for manager in list(_MANAGERS)):
        return
    try:
        from PyQt5.QtWidgets import QApplication
    except Exception:
        return
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def drawnow(ax=None):
    """Flush open figures. MATLAB ``drawnow``.

    With an axes, flush that figure. With no argument, flush every manager.
    Also pumps OpenCV and Qt so mixed tiles stay responsive.
    """
    if ax is not None:
        fig = ax.get_figure()
        _flush_fig(fig)
        _display_inline(fig)
    else:
        for manager in list(_MANAGERS):
            manager._present()
        for num in list(plt.get_fignums()):
            _flush_fig(plt.figure(num))
    _pump_cv()
    _pump_qt()


_drawnow = drawnow


def labels(ax, title=None, x=None, y=None):
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fit(ax.get_figure())


def imshow(ax, im, cmap="jet", vmin=None, vmax=None, cbar=False, **kwargs):
    artist = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    if cbar:
        ax.get_figure().colorbar(artist, ax=ax)
    return artist


_show_image = imshow


def imagesc(ax, im, **kwargs):
    """Scaled image. MATLAB ``imagesc``."""
    data = np.asarray(im, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(finite.min()), float(finite.max())
    kwargs.setdefault("vmin", lo)
    kwargs.setdefault("vmax", hi)
    kwargs.setdefault("cmap", "jet")
    return _show_image(ax, im, **kwargs)


def updateImage(h, im, auto=True, flush=True):
    h.set_data(im)
    h.set_extent((0, im.shape[1], im.shape[0], 0))
    h.axes.set_xlim(0, im.shape[1])
    h.axes.set_ylim(im.shape[0], 0)
    if auto:
        h.autoscale()
    if flush:
        _flush_fig(h.get_figure())


def colorbar(h):
    return h.axes.get_figure().colorbar(h, ax=h.axes)


def gridon(ax):
    ax.xaxis.grid(True, which="both")
    ax.yaxis.grid(True, which="both")
    zaxis = getattr(ax, "zaxis", None)
    if zaxis is not None:
        zaxis.grid(True, which="both")


def gridoff(ax):
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")
    zaxis = getattr(ax, "zaxis", None)
    if zaxis is not None:
        zaxis.grid(False, which="both")


def hold(ax, state="on"):
    """MATLAB ``hold on`` / ``hold off`` for an axes you already have."""
    ax._cog_hold = _as_on(state)
    return ax


class Cell:
    def __init__(self, kind="mpl"):
        self.kind = kind
        self.fig = None
        self.ax = None
        self.image = None
        self.line = None
        self.hold = False
        self.cbar = False
        self.cloud = None
        self.cv_name = None
        self.owns_fig = False
        self.figbar = False


def _plot3_points(x, y, z, fmt, colors):
    """Return an ``(N, 3)`` cloud when this ``plot3`` call is markers, not a line."""
    if y is None and z is None:
        arr = np.asarray(x)
        if getattr(arr, "ndim", 0) == 2 and arr.shape[1] == 3:
            return arr
        return None
    marker = isinstance(fmt, str) and "-" not in fmt and any(ch in fmt for ch in ".o+x*sd^v<>")
    if marker or colors is not None:
        return np.column_stack((np.ravel(np.asarray(x)), np.ravel(np.asarray(y)), np.ravel(np.asarray(z))))
    return None


def _draw_mpl_layer(ax, layer):
    if layer.get("kind") == "mesh":
        _mesh_layer(ax, layer)
    else:
        _scatter_layer(ax, layer)


def _mesh_layer(ax, layer):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    pts = np.asarray(layer["points"], dtype=float)
    faces = np.asarray(layer["faces"], dtype=int)[:, :3]
    style = str(layer.get("style") or "surface").lower()
    color = layer.get("color") or "lightsteelblue"
    tris = pts[faces]
    if style in ("surface", "both", "shaded"):
        edge = "black" if style == "both" else "none"
        ax.add_collection3d(Poly3DCollection(tris, facecolor=color, edgecolor=edge, linewidths=0.4))
    if style == "wireframe":
        loops = [np.vstack((pts[tri], pts[tri[0]])) for tri in faces]
        ax.add_collection3d(Line3DCollection(loops, colors="black", linewidths=0.8))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])


def _scatter_layer(ax, layer):
    pts, colors = coerce_cloud(layer["points"], layer["colors"])
    size = layer["point_size"]
    if colors is None:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=layer["color"] or "silver", s=size)
        return
    col = np.asarray(colors)
    if col.ndim == 2 and col.shape[1] == 3:
        rgb = col.astype(float)
        if rgb.size and np.nanmax(rgb) > 1.0:
            rgb /= 255.0
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rgb, s=size)
        return
    kw = {}
    if layer["cmap"] is not None:
        kw["cmap"] = layer["cmap"]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=col.reshape(-1), s=size, **kw)


class FigureManager:
    """Grid of tiles. Each tile is matplotlib, OpenCV, or a 3D point cloud.

    ``FigureManager(rows, cols, mon)`` matches the positional form used in
    notebooks. ``connect`` returns a client with the same method names.
    """

    def __init__(self, rows, cols, mon=0, pause=True, mode=None, figsize=None):
        self.rows = max(1, int(rows))
        self.cols = max(1, int(cols))
        self.mon = int(mon)
        self._pause = bool(pause)
        self.mode = mode or current_mode()
        self.figsize = figsize
        self._cells = {}
        self._inline_fig = None
        self._axes = None
        self._on_event = None
        self._cloud_warned = False
        _MANAGERS.add(self)

    @classmethod
    def connect(cls, host="localhost", port=5555):
        from cogmatplot.client import PlotClient

        return PlotClient(host, port)

    def _is_inline(self):
        return self.mode == "inline"

    def _rc(self, row, col):
        row = int(max(0, min(self.rows - 1, int(row))))
        col = int(max(0, min(self.cols - 1, int(col))))
        return row, col

    def _emit(self, event):
        if self._on_event is not None:
            self._on_event(event)

    def _figsize_of(self):
        if self.figsize is not None:
            return self.figsize
        return (4.8 * self.cols, 3.6 * self.rows)

    def _ensure_inline(self):
        if self._inline_fig is None:
            fig, axes = plt.subplots(self.rows, self.cols, squeeze=False, figsize=self._figsize_of())
            fig._cog_inline = True
            self._inline_fig = fig
            self._axes = [[axes[r][c] for c in range(self.cols)] for r in range(self.rows)]
            self._bind_keys(fig)
        return self._inline_fig

    def _inline_axis(self, row, col, projection):
        fig = self._ensure_inline()
        ax = self._axes[row][col]
        is3 = getattr(ax, "name", "") == "3d"
        want3 = projection == "3d"
        if is3 != want3:
            ax.remove()
            index = row * self.cols + col + 1
            if want3:
                ax = fig.add_subplot(self.rows, self.cols, index, projection="3d")
            else:
                ax = fig.add_subplot(self.rows, self.cols, index)
            self._axes[row][col] = ax
        return fig, ax

    def _bind_keys(self, fig):
        def on_key(event):
            self._emit({"type": "key", "key": getattr(event, "key", None), "source": "mpl"})

        try:
            fig.canvas.mpl_connect("key_press_event", on_key)
        except Exception:
            pass

    def _new_window(self, row, col, figbar, projection):
        x, y, w, h = gridfig_box(row, col, self.rows, self.cols, self.mon)
        prev = plt.rcParams.get("toolbar")
        plt.rcParams["toolbar"] = "toolbar2" if figbar else "None"
        try:
            if projection:
                fig, ax = plt.subplots(subplot_kw={"projection": projection})
            else:
                fig, ax = plt.subplots()
        finally:
            if prev is not None:
                plt.rcParams["toolbar"] = prev
        _place_window(fig, x, y, w, h)
        self._bind_keys(fig)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return fig, ax

    def _detach(self, cell):
        if cell is None:
            return
        if cell.cv_name:
            try:
                import cv2

                cv2.destroyWindow(cell.cv_name)
            except Exception:
                pass
            cell.cv_name = None
        if cell.cloud is not None:
            cell.cloud._on_change = None
            cell.cloud.close()
            cell.cloud = None
        if cell.owns_fig and cell.fig is not None:
            plt.close(cell.fig)
            cell.fig = None
            cell.ax = None

    def _forget_artists(self, cell):
        cell.line = None
        cell.image = None
        cell.cbar = False

    def _mpl(self, row, col, projection, figbar=False):
        row, col = self._rc(row, col)
        key = (row, col)
        cell = self._cells.get(key)
        if cell is not None and cell.kind == "cv":
            self._detach(cell)
            self._cells.pop(key, None)
            cell = None
        if cell is not None and cell.kind == "3d":
            self._detach(cell)
            self._cells.pop(key, None)
            cell = None
        if self._is_inline():
            fig, ax = self._inline_axis(row, col, projection)
            switched = cell is not None and cell.ax is not None and cell.ax is not ax
            if cell is None:
                cell = Cell("mpl")
                self._cells[key] = cell
            elif switched:
                self._forget_artists(cell)
            cell.kind = "mpl"
            cell.fig = fig
            cell.ax = ax
            cell.owns_fig = False
            return cell

        want3 = projection == "3d"
        if cell is not None and cell.kind == "mpl" and cell.ax is not None:
            is3 = getattr(cell.ax, "name", "") == "3d"
            if is3 == want3:
                return cell
            self._detach(cell)
            self._cells.pop(key, None)
        elif cell is not None:
            self._detach(cell)
            self._cells.pop(key, None)
        fig, ax = self._new_window(row, col, figbar, projection)
        cell = Cell("mpl")
        cell.fig = fig
        cell.ax = ax
        cell.owns_fig = True
        cell.figbar = bool(figbar)
        self._cells[key] = cell
        return cell

    def _mpl2d(self, row, col, figbar=False):
        return self._mpl(row, col, None, figbar=figbar)

    def _mpl3d(self, row, col):
        return self._mpl(row, col, "3d")

    def _require_ax(self, row, col):
        """Axes for titles and limits, without flipping 2D/3D."""
        row, col = self._rc(row, col)
        cell = self._cells.get((row, col))
        if cell is not None and cell.ax is not None:
            return cell
        if self._is_inline() and self._axes is not None:
            ax = self._axes[row][col]
            cell = Cell("mpl")
            cell.fig = self._inline_fig
            cell.ax = ax
            self._cells[(row, col)] = cell
            return cell
        return self._mpl2d(row, col)

    def _touch(self, fig=None):
        fig = fig or self._inline_fig
        if self._is_inline():
            _display_inline(fig)
            return
        if self._pause:
            _flush_fig(fig)
            plt.pause(0.0001)

    def _present(self):
        if self._inline_fig is not None:
            _display_inline(self._inline_fig)
        for cell in self._cells.values():
            if cell.owns_fig and cell.fig is not None:
                _flush_fig(cell.fig)

    def drawnow(self):
        _drawnow()

    def interactive(self, val):
        if _as_on(val):
            plt.ion()
        else:
            plt.ioff()

    def pause(self, val=0.0001):
        if self._is_inline():
            self.drawnow()
            return
        plt.pause(val)

    def reset(self, rows=None, cols=None, mon=None):
        self.closeAll()
        if rows is not None:
            self.rows = max(1, int(rows))
        if cols is not None:
            self.cols = max(1, int(cols))
        if mon is not None:
            self.mon = int(mon)

    def close(self, row, col):
        key = self._rc(row, col)
        cell = self._cells.pop(key, None)
        if cell is not None:
            self._detach(cell)
        if self._is_inline() and self._axes is not None:
            try:
                self._axes[key[0]][key[1]].cla()
            except Exception:
                pass

    def closeAll(self):
        for key in list(self._cells):
            self.close(*key)
        if self._inline_fig is not None:
            plt.close(self._inline_fig)
            self._inline_fig = None
            self._axes = None

    def clf(self, row, col):
        key = self._rc(row, col)
        cell = self._cells.get(key)
        if cell is None:
            if self._is_inline() and self._axes is not None:
                self._axes[key[0]][key[1]].cla()
            return
        cell.hold = False
        self._forget_artists(cell)
        if cell.kind == "3d" and cell.cloud is not None:
            cell.cloud.clear()
            return
        if cell.ax is not None:
            cell.ax.cla()
            self._touch(cell.fig)

    def cla(self, row, col):
        self.clf(row, col)

    def clfAll(self):
        for key in list(self._cells):
            self.clf(*key)
        if self._is_inline() and self._axes is not None:
            for row in self._axes:
                for ax in row:
                    ax.cla()

    def hold(self, row, col, state="on"):
        cell = self._require_ax(row, col)
        cell.hold = _as_on(state)
        cell.ax._cog_hold = cell.hold
        return cell.ax

    def title(self, row, col, text):
        row, col = self._rc(row, col)
        cell = self._cells.get((row, col))
        if cell is not None and cell.kind == "3d" and cell.cloud is not None:
            cell.cloud.title = text
            if self._is_inline() and self._axes is not None:
                self._axes[row][col].set_title(text)
                self._touch(self._inline_fig)
            return
        if cell is not None and cell.kind == "cv" and cell.cv_name:
            import cv2

            cv2.setWindowTitle(cell.cv_name, str(text))
            return
        cell = self._require_ax(row, col)
        cell.ax.set_title(text)
        self._touch(cell.fig)
        return cell.ax

    def xlabel(self, row, col, text):
        cell = self._require_ax(row, col)
        cell.ax.set_xlabel(text)
        self._touch(cell.fig)
        return cell.ax

    def ylabel(self, row, col, text):
        cell = self._require_ax(row, col)
        cell.ax.set_ylabel(text)
        self._touch(cell.fig)
        return cell.ax

    def axis(self, row, col, *args, **kwargs):
        cell = self._require_ax(row, col)
        result = cell.ax.axis(*args, **kwargs)
        self._touch(cell.fig)
        return result

    def grid(self, row, col, on=True):
        cell = self._require_ax(row, col)
        if _as_on(on):
            gridon(cell.ax)
        else:
            gridoff(cell.ax)
        self._touch(cell.fig)
        return cell.ax

    def xlim(self, row, col, *args, **kwargs):
        cell = self._require_ax(row, col)
        result = cell.ax.set_xlim(*args, **kwargs)
        self._touch(cell.fig)
        return result

    def ylim(self, row, col, *args, **kwargs):
        cell = self._require_ax(row, col)
        result = cell.ax.set_ylim(*args, **kwargs)
        self._touch(cell.fig)
        return result

    def zlim(self, row, col, *args, **kwargs):
        cell = self._mpl3d(row, col)
        result = cell.ax.set_zlim(*args, **kwargs)
        self._touch(cell.fig)
        return result

    def legend(self, row, col, *args, **kwargs):
        cell = self._require_ax(row, col)
        result = cell.ax.legend(*args, **kwargs)
        self._touch(cell.fig)
        return result

    def colorbar(self, row, col, **kwargs):
        cell = self._require_ax(row, col)
        if cell.image is None:
            return None
        bar = cell.fig.colorbar(cell.image, ax=cell.ax, **kwargs)
        cell.cbar = True
        self._touch(cell.fig)
        return bar

    def text(self, row, col, txt, xy=(0, 0), **kwargs):
        kwargs.pop("flush", None)
        cell = self._mpl2d(row, col)
        kwargs.setdefault("ha", "left")
        kwargs.setdefault("va", "top")
        artist = cell.ax.text(xy[0], xy[1], txt, **kwargs)
        self._touch(cell.fig)
        return artist

    def rectangle(self, row, col, x, y, w, h, **kwargs):
        cell = self._mpl2d(row, col)
        kwargs.setdefault("fill", False)
        patch = Rectangle((x, y), w, h, **kwargs)
        cell.ax.add_patch(patch)
        self._touch(cell.fig)
        return patch

    def box(self, row, col, x, y, hold=False, **kwargs):
        kwargs.pop("flush", None)
        if hold:
            self.hold(row, col, "on")
        xs = [x[0], x[1], x[1], x[0], x[0]]
        ys = [y[0], y[0], y[1], y[1], y[0]]
        fmt = kwargs.pop("fmt", "-")
        return self.plot(row, col, xs, ys, fmt, grid=kwargs.pop("grid", 0), **kwargs)

    def imshow(self, row, col, im, auto=0, title=None, figbar=False, viewer=None, cbar=False, **kwargs):
        kwargs.pop("flush", None)
        if str(viewer or "mpl").lower() in ("cv", "cv2", "opencv"):
            prep = {key: kwargs.pop(key) for key in ("normalize", "colormap", "rgb") if key in kwargs}
            bgr = prepare_bgr(im, **prep)
            if bgr is None:
                return None
            if not self._is_inline():
                return self._show_cv(row, col, bgr, title=title)
            im = np.ascontiguousarray(bgr[:, :, ::-1])
        cell = self._mpl2d(row, col, figbar=figbar)
        ax = cell.ax
        if cell.image is None or getattr(cell.image, "axes", None) is None:
            cell.image = _show_image(ax, im, **kwargs)
        else:
            updateImage(cell.image, im, auto=bool(auto), flush=False)
        if cbar and not cell.cbar and cell.image is not None:
            ax.get_figure().colorbar(cell.image, ax=ax)
            cell.cbar = True
        if title is not None:
            ax.set_title(title)
        self._touch(cell.fig)
        return ax

    def imagesc(self, row, col, im, **kwargs):
        kwargs.pop("flush", None)
        data = np.asarray(im, dtype=float)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = float(finite.min()), float(finite.max())
        kwargs.setdefault("vmin", lo)
        kwargs.setdefault("vmax", hi)
        kwargs.setdefault("cmap", "jet")
        return self.imshow(row, col, im, **kwargs)

    def _show_cv(self, row, col, bgr, title=None):
        import cv2

        row, col = self._rc(row, col)
        key = (row, col)
        cell = self._cells.get(key)
        if cell is None or cell.kind != "cv":
            if cell is not None:
                self._detach(cell)
            name = f"cog{id(self) % 100000}:{row},{col}"
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            x, y, w, h = gridfig_box(row, col, self.rows, self.cols, self.mon)
            from cogmatplot.place import place_cv_window

            place_cv_window(name, x, y, w, h)
            cell = Cell("cv")
            cell.cv_name = name
            self._cells[key] = cell
        cv2.imshow(cell.cv_name, bgr)
        if title:
            cv2.setWindowTitle(cell.cv_name, str(title))
        self._touch()
        return None

    def plot(self, row, col, x, y, fmt=".-", grid=1, ylabel=None, xlabel=None, figbar=False, hold=None, **kwargs):
        kwargs.pop("flush", None)
        cell = self._mpl2d(row, col, figbar=figbar)
        ax = cell.ax
        if hold is not None:
            cell.hold = _as_on(hold)
        if cell.line is not None and not cell.hold:
            cell.line.set_data(x, y)
            ax.relim()
            ax.autoscale_view()
        else:
            (line,) = ax.plot(x, y, fmt, **kwargs)
            if not cell.hold:
                cell.line = line
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if grid:
            gridon(ax)
        self._touch(cell.fig)
        return ax

    def plot3(self, row, col, x, y=None, z=None, fmt=".-", hold=None, color=None, colors=None, point_size=3, **kwargs):
        """MATLAB ``plot3(X, Y, Z)`` and ``plot3(X, Y, Z, LineSpec)``.

        A line spec such as ``'-'`` or ``'.-'`` draws a 3D line. A marker spec
        such as ``'.'``, or an ``(N, 3)`` point array, draws a point cloud.
        ``colors`` is per-point rgb.
        """
        kwargs.pop("flush", None)
        if isinstance(y, str) and z is None:
            fmt = y
            y = None
        points = _plot3_points(x, y, z, fmt, colors)
        if points is not None:
            return self.scatter3(
                row, col, points, color=color, colors=colors, point_size=point_size,
                hold=True if hold is None else hold, **kwargs,
            )
        cell = self._mpl3d(row, col)
        ax = cell.ax
        if hold is not None:
            cell.hold = _as_on(hold)
        if cell.line is not None and not cell.hold and hasattr(cell.line, "set_data_3d"):
            cell.line.set_data_3d(x, y, z)
        else:
            (line,) = ax.plot(x, y, z, fmt, **kwargs)
            if not cell.hold:
                cell.line = line
        self._touch(cell.fig)
        return ax

    def figure3(self, row, col, title=None, **kwargs):
        del kwargs
        from cogmatplot.view3 import Fig3d

        row, col = self._rc(row, col)
        key = (row, col)
        cell = self._cells.get(key)
        if cell is not None and cell.kind == "3d" and cell.cloud is not None:
            if title:
                cell.cloud.title = title
            return cell.cloud
        if cell is not None:
            self._detach(cell)
            self._cells.pop(key, None)
        cloud = Fig3d(
            title=title or "Fig3d",
            show=not self._is_inline(),
            grid_pos=(row, col, self.rows, self.cols, self.mon),
            off_screen=self._is_inline(),
            on_change=self._on_cloud if self._is_inline() else None,
        )
        cloud.row = row
        cloud.col = col
        cell = Cell("3d")
        cell.cloud = cloud
        self._cells[key] = cell
        return cloud

    def scatter3(self, row, col, points, color=None, colors=None, point_size=3, cmap=None, hold=True, **kwargs):
        cloud = self.figure3(row, col)
        cloud.scatter3(points, color=color, colors=colors, point_size=point_size, cmap=cmap, hold=hold, **kwargs)
        return cloud

    def plot_mesh(self, row, col, points, faces, color=None, colors=None, style="surface", hold=True, **kwargs):
        """Draw a triangle mesh. ``style`` is ``surface``, ``wireframe``, or ``both``."""
        cloud = self.figure3(row, col)
        cloud.plot_mesh(points, faces, color=color, colors=colors, style=style, hold=hold, **kwargs)
        return cloud

    def view3(self, row, col, which="xy"):
        self.figure3(row, col).view(which)

    def grid3(self, row, col, on=True):
        self.figure3(row, col).grid(on)

    def clear3(self, row, col):
        key = self._rc(row, col)
        cell = self._cells.get(key)
        if cell is not None and cell.cloud is not None:
            cell.cloud.clear()

    def _on_cloud(self, cloud):
        if not self._is_inline():
            self._touch()
            return
        row, col = cloud.row, cloud.col
        fig = self._ensure_inline()
        if not cloud.layers:
            _fig, ax = self._inline_axis(row, col, None)
            ax.cla()
            if cloud.title:
                ax.set_title(cloud.title)
            self._touch(fig)
            return
        try:
            image = cloud.snapshot()
            _fig, ax = self._inline_axis(row, col, None)
            ax.clear()
            ax.imshow(image)
            ax.axis("off")
        except Exception as exc:
            _fig, ax = self._inline_axis(row, col, "3d")
            ax.cla()
            for layer in cloud.layers:
                _draw_mpl_layer(ax, layer)
            if not self._cloud_warned:
                self._cloud_warned = True
                warnings.warn(
                    f"inline 3d render failed ({exc}); drew a matplotlib view instead",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if cloud.title:
            ax.set_title(cloud.title)
        self._touch(fig)


__all__ = [
    "FigureManager",
    "saveFig",
    "closeAll",
    "clfAll",
    "figure",
    "gridfig",
    "fit",
    "axfit",
    "drawnow",
    "labels",
    "imshow",
    "imagesc",
    "updateImage",
    "colorbar",
    "gridon",
    "gridoff",
    "hold",
]
