# %% Rectangular prism from PLY files.
# load_ply only reads the file. plot3 draws the point cloud (MATLAB style).
# plot_mesh draws the faces: wireframe, shaded, or both.
#
#   python examples/ply_prism.py

import cogmatplot

cogmatplot.use("qt")

from pathlib import Path

from cogmatplot import FigureManager, closeAll
from cogmatplot.ply import load_ply

DATA = Path(__file__).resolve().parent / "data"
MON = 0

# %%
closeAll()
fm = FigureManager(2, 2, MON, pause=False)

cloud = load_ply(DATA / "prism_cloud.ply")
x, y, z = cloud.points.T
view = fm.plot3(0, 0, x, y, z, ".", colors=cloud.colors, point_size=8)
view.view_isometric()
fm.title(0, 0, "point cloud")

mesh = load_ply(DATA / "prism_mesh.ply")
wire = fm.plot_mesh(0, 1, mesh.points, mesh.faces, style="wireframe")
wire.view_isometric()
fm.title(0, 1, "wireframe")

shaded = fm.plot_mesh(1, 0, mesh.points, mesh.faces, style="surface", color="lightsteelblue")
shaded.view_isometric()
fm.title(1, 0, "shaded")

both = fm.plot_mesh(1, 1, mesh.points, mesh.faces, style="both", color="lightsteelblue")
both.view_isometric()
fm.title(1, 1, "shaded + edges")
fm.drawnow()
