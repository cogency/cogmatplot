"""Session display mode.

``qt`` opens interactive windows on a monitor, outside the notebook.
``inline`` draws static figures into the interactive viewer.
Call :func:`use` before creating figures.
"""

from __future__ import annotations

import sys

_mode = None


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython
    except Exception:
        return False
    return get_ipython() is not None


def use(mode: str = "auto") -> str:
    """Select ``'qt'``, ``'inline'``, or ``'auto'``.

    ``auto`` is inline inside IPython and Qt otherwise.
    Qt figures are interactive and can be placed on another monitor.
    Inline figures are static: images, axes, 2D plots, 3D lines, and a
    rendered 3D view in the notebook output.
    """
    global _mode
    mode = str(mode).lower()
    if mode == "auto":
        mode = "inline" if _in_ipython() else "qt"
    if mode not in ("inline", "qt"):
        raise ValueError("mode must be 'qt', 'inline', or 'auto'")

    import matplotlib

    current = matplotlib.get_backend().lower()
    loaded = "matplotlib.pyplot" in sys.modules
    if mode == "inline":
        if "inline" not in current and current != "agg":
            target = "module://matplotlib_inline.backend_inline" if _in_ipython() else "Agg"
            try:
                matplotlib.use(target, force=loaded)
            except Exception:
                if not loaded:
                    matplotlib.use("Agg")
    elif "qt" not in current:
        try:
            matplotlib.use("QtAgg", force=loaded)
        except Exception:
            if not loaded:
                matplotlib.use("QtAgg")

    _mode = mode
    if loaded or mode == "qt":
        import matplotlib.pyplot as plt

        if mode == "qt":
            plt.ion()
            _enable_qt_loop()
        else:
            plt.ioff()
    return mode


def _enable_qt_loop():
    """Run the Qt loop between cells, as ``%matplotlib qt`` does.

    matplotlib only does this the first time it hooks into IPython, so a
    kernel that started inline would otherwise leave Qt windows frozen.
    """
    try:
        from IPython import get_ipython
    except Exception:
        return
    shell = get_ipython()
    if shell is None or getattr(shell, "active_eventloop", None) == "qt":
        return
    try:
        shell.enable_gui("qt")
    except Exception:
        pass


def current_mode() -> str:
    """Active mode. An explicit :func:`use` wins; otherwise follow the backend."""
    if _mode:
        return _mode
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if any(name in backend for name in ("qt", "wx", "tk", "gtk", "macosx")):
        return "qt"
    return "inline"
