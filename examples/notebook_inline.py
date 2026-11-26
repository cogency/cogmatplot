# %% Static figures inside the interactive viewer.
# These are not interactive. The same calls draw images, 2D axes,
# a 3D line, and a rendered point cloud.

import cogmatplot

cogmatplot.use("inline")

from cogmatplot import FigureManager, closeAll
from scene import draw, make

# %%
closeAll()
fm = FigureManager(2, 2, pause=False)
draw(fm, n=800)

# %% A cv-style image is drawn into the output as well.
live = FigureManager(1, 1, pause=False)
live.imshow(0, 0, make(n=10)["image"], viewer="cv", title="inline image")
live.drawnow()
