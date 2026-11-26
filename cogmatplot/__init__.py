"""MATLAB-style figures for notebooks, pop-out monitors, and a debug server.

Pick the display before importing the plotting classes when you can::

    import cogmatplot
    cogmatplot.use("inline")  # static figures in the interactive viewer
    cogmatplot.use("qt")      # interactive windows on a monitor

``FigureManager`` is loaded lazily so ``use`` can select the backend first.
"""

from cogmatplot.mode import current_mode, use

__all__ = [
    "use",
    "current_mode",
    "FigureManager",
    "closeAll",
    "drawnow",
    "Fig3d",
    "PlotClient",
    "Ply",
    "load_ply",
    "save_ply",
    "rectangular_prism",
]


def __getattr__(name):
    if name in ("FigureManager", "closeAll", "drawnow"):
        from cogmatplot import matplot

        return getattr(matplot, name)
    if name == "Fig3d":
        from cogmatplot.view3 import Fig3d

        return Fig3d
    if name == "PlotClient":
        from cogmatplot.client import PlotClient

        return PlotClient
    if name in ("Ply", "load_ply", "save_ply", "rectangular_prism"):
        from cogmatplot import ply

        return getattr(ply, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
