"""3D boxes: corners, a quad mesh for plotting, and cropping a cloud to a box.

Corner order is the scanner's: 0-3 go round the bottom face and 4-7 are the
same corners on the top face, so corner ``i + 4`` sits above corner ``i``.
A box may be rotated. Anything that takes a box accepts those 8 corners or
axis-aligned bounds ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
"""

from __future__ import annotations

import numpy as np

from cogmatplot.ply import Ply

BOX_FACES = np.array(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ],
    dtype=np.int64,
)


def box_corners(box) -> np.ndarray:
    """Return the ``(8, 3)`` corners of a box given as corners or as 6 bounds."""
    arr = np.asarray(box, dtype=np.float64)
    if arr.shape == (8, 3):
        return arr
    if arr.size != 6:
        raise ValueError("box must be (8, 3) corners or (xmin, xmax, ymin, ymax, zmin, zmax)")
    x0, x1, y0, y1, z0, z1 = arr.ravel()
    bottom = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]
    top = [[x, y, z1] for x, y, _ in bottom]
    return np.array(bottom + top, dtype=np.float64)


def box_mesh(box) -> Ply:
    """A box as a :class:`Ply` with 8 points and 6 quad faces, ready for ``plot_mesh``."""
    return Ply(points=box_corners(box), faces=BOX_FACES.copy())


def crop_box(points, box, margin_xy=0.0, margin_z=0.0, up=(0.0, 0.0, 1.0)):
    """Keep the points inside a box grown by margins. Returns ``(keep, grown_box)``.

    The box may be rotated. Its own axes are used: ``margin_xy`` is
    ``(mx, my)`` added on both sides along the box's two horizontal axes, and
    ``margin_z`` is ``(floor, ceiling)`` along the box axis closest to ``up``.
    A single number applies to both. ``keep`` is a boolean mask over
    ``points``, so ``points[keep]`` and ``colors[keep]`` stay aligned.
    ``grown_box`` is the 8 grown corners in the same order as ``box``.
    """
    from cogmatplot.ply import as_ply, is_cloud

    pts = as_ply(points).points if is_cloud(points) else np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    corners = box_corners(box)
    mx, my = np.broadcast_to(np.asarray(margin_xy, dtype=np.float64), (2,))
    floor, ceiling = np.broadcast_to(np.asarray(margin_z, dtype=np.float64), (2,))

    center = corners.mean(axis=0)
    axes = _box_axes(corners - center, up)
    local = (corners - center) @ axes
    lo = local.min(axis=0)
    hi = local.max(axis=0)
    grow_lo = lo - [mx, my, floor]
    grow_hi = hi + [mx, my, ceiling]

    eps = 1e-9
    inside = (pts - center) @ axes
    keep = np.all((inside >= grow_lo - eps) & (inside <= grow_hi + eps), axis=1)

    upper = local >= 0.5 * (lo + hi) - eps
    grown = np.where(upper, grow_hi, grow_lo)
    return keep, center + grown @ axes.T


def _box_axes(centered, up):
    """Orthonormal box axes as columns ``(x, y, z)``, with ``z`` the one nearest ``up``."""
    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    candidates = vt.T
    z_idx = int(np.argmax(np.abs(candidates.T @ up)))
    z = candidates[:, z_idx].copy()
    if z @ up < 0:
        z = -z
    rest = [i for i in range(3) if i != z_idx]
    x = candidates[:, rest[0]] - (candidates[:, rest[0]] @ z) * z
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
