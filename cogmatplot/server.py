"""Figure server.

``python -m cogmatplot.server`` owns the windows. A debugger uses
``PlotClient`` or ``FigureManager.connect``. Each public FigureManager
method is called with the arguments the client sent. Image updates for one
cell collapse to the latest frame. Key presses are published on ``port + 1``.
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback

import zmq

from cogmatplot.api import METHODS
from cogmatplot.wire import Mailbox, decode


def run_server(fm, port=5555, stop_event=None, host="127.0.0.1", ready=None):
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.LINGER, 0)
    pull.setsockopt(zmq.RCVHWM, 64)
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    try:
        pull.bind(f"tcp://{host}:{int(port)}")
        pub.bind(f"tcp://{host}:{int(port) + 1}")
    except Exception:
        if ready is not None:
            ready.set()
        pull.close(0)
        pub.close(0)
        ctx.term()
        raise

    mailbox = Mailbox()
    stopping = stop_event if stop_event is not None else threading.Event()

    def emit(event):
        try:
            pub.send_json(event, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    fm._on_event = emit

    def recv_loop():
        while not stopping.is_set():
            if not pull.poll(100):
                continue
            try:
                method, args, kwargs = decode(pull.recv_multipart())
            except Exception:
                traceback.print_exc()
                continue
            if method not in METHODS:
                print(f"cogmatplot server: ignored {method}")
                continue
            mailbox.push(method, args, kwargs)

    thread = threading.Thread(target=recv_loop, name="cogmatplot-recv", daemon=True)
    thread.start()
    if ready is None:
        print(f"cogmatplot server on tcp://{host}:{port}  events on {int(port) + 1}")
    else:
        ready.set()

    try:
        while not stopping.is_set():
            batch = mailbox.drain()
            for method, args, kwargs in batch:
                try:
                    getattr(fm, method)(*args, **kwargs)
                except Exception:
                    print(f"cogmatplot server: {method} failed")
                    traceback.print_exc()
            if batch or _needs_pump(fm):
                try:
                    fm.drawnow()
                except Exception:
                    traceback.print_exc()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        thread.join(timeout=1)
        pull.close(0)
        pub.close(0)
        ctx.term()


def _needs_pump(fm) -> bool:
    if getattr(fm, "mode", None) != "qt":
        return False
    return bool(getattr(fm, "_cells", None)) or getattr(fm, "_inline_fig", None) is not None


def main(raw_args=None):
    if isinstance(raw_args, str):
        raw_args = [part for part in raw_args.split(" ") if part]
    parser = argparse.ArgumentParser(description="cogmatplot figure server")
    parser.add_argument("-p", "--port", type=int, default=5555)
    parser.add_argument("-m", "--monitor", type=int, default=2)
    parser.add_argument("-r", "--rows", type=int, default=2)
    parser.add_argument("-c", "--cols", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(raw_args)

    from cogmatplot.mode import use

    use("qt")
    from cogmatplot.matplot import FigureManager

    figure = FigureManager(args.rows, args.cols, mon=args.monitor, pause=False)
    run_server(figure, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
