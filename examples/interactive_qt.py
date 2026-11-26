# %% Interactive pop-out figures.
# Run in the interactive window. Windows are live: pan, zoom, orbit.
# If this kernel was previously inline, switch first:
# %matplotlib qt

import cogmatplot

cogmatplot.use("qt")

from cogmatplot import FigureManager, closeAll
from scene import draw

MON = 0  # monitor index; set this to the screen you want

# %%
closeAll()
fm = FigureManager(2, 2, MON, pause=False)
draw(fm)

# %% OpenCV tile in the same kind of grid. This window is the fast live path.
import numpy as np

live = FigureManager(1, 1, MON, pause=False)
frame = np.zeros((180, 240, 3), np.uint8)
frame[:, :, 1] = 180
live.imshow(0, 0, frame, viewer="cv", title="cv")
live.drawnow()
