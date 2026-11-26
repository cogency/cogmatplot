# %% Fast OpenCV debug streams.
# Pop-out named windows. Call render once per frame after publishing.

import numpy as np

from cogmatplot.debug_views import (
    close_debug_views,
    debug_image,
    init_debug_views,
    render_debug_views,
)

init_debug_views(cols=2, monitor=0)

# %%
for i in range(40):
    gray = np.full((120, 160), (i * 6) % 256, np.uint8)
    color = np.zeros((120, 160, 3), np.uint8)
    color[:, :, 1] = gray
    debug_image("gray", gray)
    debug_image("color", color)
    render_debug_views()

close_debug_views()
