"""Method names shared by FigureManager and the remote client.

The server calls ``getattr(figure, method)(*args, **kwargs)``. A new plotting
helper is added here and on ``FigureManager`` once.
"""

STREAM_METHODS = ("imshow", "imagesc")

METHODS = (
    "imshow",
    "imagesc",
    "plot",
    "plot3",
    "scatter3",
    "plot_mesh",
    "plot_box",
    "figure3",
    "title",
    "xlabel",
    "ylabel",
    "axis",
    "hold",
    "grid",
    "xlim",
    "ylim",
    "zlim",
    "legend",
    "colorbar",
    "text",
    "box",
    "rectangle",
    "clf",
    "cla",
    "close",
    "closeAll",
    "clfAll",
    "reset",
    "drawnow",
    "pause",
    "interactive",
    "view3",
    "grid3",
    "clear3",
)
