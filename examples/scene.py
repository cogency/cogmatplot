"""Shared 2D / 3D scene for the examples.

``draw(fm)`` uses only FigureManager methods, so the same function runs
in-process and through PlotClient. The cloud shows xyz+rgb points and a
solid-color xyz tag, which is the bagtag depth view.
"""

from __future__ import annotations

import numpy as np


def make(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    height, width = 180, 240
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.stack([
        (xx * 255 // width).astype(np.uint8),
        (yy * 255 // height).astype(np.uint8),
        np.full((height, width), 90, np.uint8),
    ], axis=-1)
    poly = np.array([[30, 30], [160, 40], [150, 120], [40, 110], [30, 30]], float)
    x = np.linspace(0, 4 * np.pi, 200)
    y1 = np.sin(x)
    y2 = 0.6 * np.cos(x)
    t = np.linspace(0, 6 * np.pi, 300)
    helix = (np.cos(t), np.sin(t), t / 4)
    pts = np.column_stack([
        rng.uniform(0, 800, n),
        rng.uniform(0, 400, n),
        rng.normal(40, 25, n),
    ])
    rgb = np.zeros((n, 3), np.uint8)
    rgb[:, 0] = np.clip(pts[:, 0] / 800 * 255, 0, 255).astype(np.uint8)
    rgb[:, 1] = np.clip(pts[:, 1] / 400 * 255, 0, 255).astype(np.uint8)
    rgb[:, 2] = 160
    tag = pts[pts[:, 2] > 70]
    return {
        "image": image,
        "poly": poly,
        "x": x,
        "y1": y1,
        "y2": y2,
        "helix": helix,
        "pts": pts,
        "rgb": rgb,
        "tag": tag,
    }


def draw(fm, n=1200):
    """Draw a 2x2 grid: image, 2D plot, 3D line, point cloud."""
    data = make(n=n)
    for row, col in ((0, 0), (0, 1), (1, 0), (1, 1)):
        fm.clf(row, col)

    fm.imshow(0, 0, data["image"], title="image")
    fm.hold(0, 0, "on")
    fm.plot(0, 0, data["poly"][:, 0], data["poly"][:, 1], "y-", linewidth=2, grid=0)
    fm.axis(0, 0, "off")

    fm.plot(0, 1, data["x"], data["y1"], ".-", label="sin", xlabel="x", ylabel="y")
    fm.hold(0, 1, "on")
    fm.plot(0, 1, data["x"], data["y2"], "-", label="cos")
    fm.legend(0, 1)
    fm.title(0, 1, "plot")

    fm.plot3(1, 0, *data["helix"], "-")
    fm.title(1, 0, "plot3")

    # xyz+rgb cloud, then an xyz tag in one color (same pattern as bagtag depth).
    cloud = fm.figure3(1, 1, title="points")
    cloud.scatter3(data["pts"], colors=data["rgb"], point_size=3)
    cloud.scatter3(data["tag"], color="red", point_size=8)
    cloud.show_grid()
    cloud.view_xy()
    fm.drawnow()
    return data
