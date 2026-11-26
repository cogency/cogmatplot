# %% 3D box: draw it over a cloud, then crop the cloud to the box plus margins.
# A box is 8 corners (0-3 bottom, 4-7 top) or bounds (xmin, xmax, ymin, ymax, zmin, zmax).
# crop_box works in the box's own axes, so rotated boxes crop correctly.
#
#   python examples/box_crop.py

import cogmatplot

cogmatplot.use("qt")

import numpy as np

from cogmatplot import FigureManager, closeAll
from cogmatplot.box3 import box_corners, crop_box
from cogmatplot.ply import rectangular_prism

MON = 0


def rotate_z(points, degrees, about):
    a = np.radians(degrees)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return (points - about) @ rot.T + about


rng = np.random.default_rng(0)
size = (500.0, 300.0, 250.0)
offset = np.array([450.0, 200.0, 0.0])
center = offset + np.array(size) / 2

bag, _ = rectangular_prism(size=size, step=12)
bag_points = rotate_z(bag.points + offset, 25, center)
belt = np.column_stack([rng.uniform(0, 1400, 20000), rng.uniform(0, 700, 20000), rng.normal(0, 2, 20000)])
points = np.vstack([belt, bag_points])
colors = np.vstack([np.full((len(belt), 3), 150, np.uint8), bag.colors])

box = rotate_z(box_corners((0, size[0], 0, size[1], 0, size[2])) + offset, 25, center)
keep, grown = crop_box(points, box, margin_xy=(40, 10), margin_z=(0, 150))

# %%
closeAll()
fm = FigureManager(1, 2, MON, pause=False)

view = fm.plot3(0, 0, points, colors=colors, point_size=3)
fm.plot_box(0, 0, box, color="blue")
view.view_isometric()
fm.title(0, 0, "cloud and box")

cropped = fm.plot3(0, 1, points[keep], colors=colors[keep], point_size=3)
fm.plot_box(0, 1, box, color="blue")
fm.plot_box(0, 1, grown, color="green")
cropped.view_isometric()
fm.title(0, 1, f"cropped: {keep.sum()} of {len(points)} points")
fm.drawnow()
