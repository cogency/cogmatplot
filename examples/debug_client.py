# %% Plot into a cogmatplot server from another process.
# Start the server first, in its own terminal:
#   python -m cogmatplot.server -m 0 -r 2 -c 2
# The server windows are the interactive ones. This process only sends calls.

from cogmatplot import FigureManager, closeAll
from scene import draw

fm = FigureManager.connect("127.0.0.1", 5555)
fm.reset(2, 2, 2)
draw(fm, n=800)
print("sent. poll_event() reads key presses from the server on port 5556.")
