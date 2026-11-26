"""Unit tests for the cogmatplot library. Headless: Agg backend, no Qt windows."""

from __future__ import annotations

import socket
import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np

import cogmatplot

cogmatplot.use("inline")

from cogmatplot.api import METHODS
from cogmatplot.layout import grid_positions, gridfig_box
from cogmatplot.matplot import FigureManager, closeAll, gridfig
from cogmatplot.view3 import coerce_cloud
from cogmatplot.wire import Mailbox, decode, encode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from scene import draw, make  # noqa: E402


SCREEN = SimpleNamespace(x=100, y=50, width=1200, height=800, name="test")


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class TestPackage(unittest.TestCase):
    def test_import_is_this_repo(self):
        path = Path(cogmatplot.__file__).resolve()
        self.assertEqual(path, ROOT / "cogmatplot" / "__init__.py")
        self.assertNotIn("cog_matplot", str(path))

    def test_version_is_set(self):
        self.assertRegex(cogmatplot.__version__, r"^\d+\.\d+")

    def test_manager_has_remote_methods(self):
        missing = [name for name in METHODS if not hasattr(FigureManager, name)]
        self.assertEqual(missing, [])


class TestWire(unittest.TestCase):
    def test_roundtrip_arrays(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        points = np.asfortranarray(np.arange(9, dtype=np.float32).reshape(3, 3))
        method, args, kwargs = decode(encode(
            "imshow",
            (0, 1, image),
            {"title": "frame", "empty": np.zeros((0, 3), np.float64)},
        ))
        self.assertEqual(method, "imshow")
        self.assertEqual(args[0], 0)
        self.assertEqual(args[1], 1)
        self.assertEqual(args[2].dtype, np.uint8)
        np.testing.assert_array_equal(args[2], image)
        np.testing.assert_array_equal(kwargs["empty"], np.zeros((0, 3)))
        self.assertEqual(kwargs["title"], "frame")
        _method, packed, _kw = decode(encode("scatter3", (points,), {}))
        np.testing.assert_array_equal(packed[0], np.arange(9, dtype=np.float32).reshape(3, 3))

    def test_mailbox_keeps_latest_image_only(self):
        box = Mailbox()
        box.push("imshow", (0, 0, "a"), {})
        box.push("imshow", (0, 0, "b"), {})
        box.push("imshow", (0, 0, "c"), {})
        queued = box.drain()
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0][1][2], "c")

    def test_mailbox_keeps_commands_between_images(self):
        box = Mailbox()
        box.push("imshow", (0, 0, "a"), {})
        box.push("plot", (0, 0, [1], [2]), {})
        box.push("imshow", (0, 0, "b"), {})
        kinds = [item[0] for item in box.drain()]
        self.assertEqual(kinds, ["imshow", "plot", "imshow"])


class TestLayout(unittest.TestCase):
    def test_gridfig_box_matches_monitor_math(self):
        with patch("cogmatplot.layout._monitors", return_value=[SCREEN]):
            x, y, w, h = gridfig_box(1, 1, 2, 2, 0)
        self.assertEqual((x, y, w, h), (700, 432, 550, 302))

    def test_grid_positions_are_row_major(self):
        with patch("cogmatplot.layout._monitors", return_value=[SCREEN]):
            positions = grid_positions(3, cols=2, cell_size=(10, 20), title=5, taskbar=0)
        self.assertEqual(positions[0][:2], (100, 50))
        self.assertEqual(positions[1][0], 100 + 20)
        self.assertEqual(positions[2][1], 50 + 10 + 5)
        self.assertEqual(positions[0][2:], (20, 10))


class TestClouds(unittest.TestCase):
    def test_coerce_transposes_channel_first(self):
        pts = np.arange(12, dtype=float).reshape(3, 4)
        out, colors = coerce_cloud(pts)
        self.assertEqual(out.shape, (4, 3))
        self.assertIsNone(colors)

    def test_coerce_open3d_like_object(self):
        raw = SimpleNamespace(
            points=np.ones((5, 3)),
            colors=np.full((5, 3), 255),
        )
        pts, colors = coerce_cloud(raw)
        self.assertEqual(pts.shape, (5, 3))
        np.testing.assert_array_equal(colors, raw.colors)

    def test_square_cloud_is_not_transposed(self):
        pts = np.eye(3)
        out, _colors = coerce_cloud(pts)
        np.testing.assert_array_equal(out, pts)


class TestImages(unittest.TestCase):
    def test_prepare_normalizes_gray(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("opencv is not installed")
        from cogmatplot.images import prepare_bgr

        bgr = prepare_bgr(np.array([[0.0, 1.0], [0.5, 0.5]]))
        self.assertEqual(bgr.shape, (2, 2, 3))
        self.assertEqual(bgr.dtype, np.uint8)
        self.assertEqual(int(bgr[0, 0, 0]), 0)
        self.assertEqual(int(bgr[0, 1, 0]), 255)

    def test_prepare_rgb_flag_swaps_channels(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("opencv is not installed")
        from cogmatplot.images import prepare_bgr

        rgb = np.zeros((2, 2, 3), np.uint8)
        rgb[..., 0] = 10
        rgb[..., 2] = 30
        swapped = prepare_bgr(rgb, rgb=True)
        self.assertEqual(int(swapped[0, 0, 0]), 30)
        self.assertEqual(int(swapped[0, 0, 2]), 10)


class TestFigures(unittest.TestCase):
    def setUp(self):
        self.fm = FigureManager(2, 2, mon=0, pause=False, mode="inline")

    def tearDown(self):
        self.fm.closeAll()
        closeAll()

    def test_plot_replaces_until_hold(self):
        ax = self.fm.plot(0, 0, [0, 1, 2], [0, 1, 0], "o-")
        self.fm.plot(0, 0, [0, 1, 2], [2, 1, 0], "o-")
        self.assertEqual(len(ax.lines), 1)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [2, 1, 0])
        self.fm.hold(0, 0, "on")
        self.fm.plot(0, 0, [0, 1], [0, 1], "--")
        self.assertEqual(len(ax.lines), 2)

    def test_imshow_updates_in_place_and_imagesc_scales(self):
        image = np.arange(16, dtype=np.uint8).reshape(4, 4)
        ax = self.fm.imshow(0, 1, image, title="first")
        self.assertEqual(len(ax.images), 1)
        self.assertEqual(ax.get_title(), "first")
        self.fm.imshow(0, 1, image + 1, title="second")
        self.assertEqual(len(ax.images), 1)
        self.assertEqual(ax.get_title(), "second")
        depth = np.array([[5.0, 15.0], [np.nan, 25.0]])
        ax2 = self.fm.imagesc(1, 1, depth)
        shown = ax2.images[0]
        self.assertAlmostEqual(shown.get_clim()[0], 5.0)
        self.assertAlmostEqual(shown.get_clim()[1], 25.0)

    def test_plot3_keeps_its_title(self):
        ax = self.fm.plot3(1, 0, [0, 1], [0, 1], [0, 2], "-")
        self.fm.title(1, 0, "plot3")
        self.assertEqual(ax.name, "3d")
        self.assertEqual(ax.get_title(), "plot3")
        self.assertEqual(len(ax.lines), 1)

    def test_overlays_and_markers(self):
        self.fm.imshow(0, 0, np.zeros((8, 8, 3), np.uint8))
        self.fm.hold(0, 0, "on")
        self.fm.plot(0, 0, [1, 2, 3], [1, 2, 1], "y-", grid=0)
        self.fm.axis(0, 0, "off")
        self.fm.text(0, 0, "tag", (1, 1))
        self.fm.rectangle(0, 0, 1, 1, 3, 2)
        self.fm.box(0, 0, [0, 4], [0, 4], hold=True)
        ax = self.fm._axes[0][0]
        self.assertTrue(ax.images)
        self.assertGreaterEqual(len(ax.lines), 2)
        self.assertTrue(ax.texts)
        self.assertTrue(ax.patches)

    def test_cv_viewer_draws_inline(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("opencv is not installed")
        ax = self.fm.imshow(0, 0, np.full((6, 5), 3.0), viewer="cv", title="cv")
        self.assertEqual(ax.images[0].get_array().shape[-1], 3)
        self.assertEqual(ax.get_title(), "cv")

    def test_scatter_xyz_and_rgb_without_popout(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [0.2, 1.0, 0.1], [0.4, 0.4, 1.0]])
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cloud = self.fm.scatter3(1, 1, pts, colors=rgb, point_size=4)
            cloud.scatter3(pts[:2], color="red", point_size=8)
            cloud.view_xy()
        self.assertEqual(len(cloud.layers), 2)
        ax = self.fm._axes[1][1]
        self.assertTrue(ax.name == "3d" or ax.images)
        if ax.name == "3d":
            self.assertGreaterEqual(len(ax.collections), 1)
        self.assertEqual(ax.get_title(), "Fig3d")

    def test_scene_draws_every_tile(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            draw(self.fm, n=80)
        self.assertTrue(self.fm._axes[0][0].images)
        self.assertTrue(self.fm._axes[0][0].lines)
        self.assertGreaterEqual(len(self.fm._axes[0][1].lines), 2)
        self.assertEqual(self.fm._axes[1][0].name, "3d")
        self.assertEqual(self.fm._axes[1][0].get_title(), "plot3")
        cloud_ax = self.fm._axes[1][1]
        self.assertTrue(cloud_ax.name == "3d" or cloud_ax.images)
        self.assertEqual(cloud_ax.get_title(), "points")

    def test_gridfig_creates_axes_without_a_window_manager(self):
        _fig, ax = gridfig(0, 0, rows=1, cols=1, mon=0)
        self.assertIsNotNone(ax)
        import matplotlib.pyplot as plt

        plt.close(_fig)


class TestDebugViews(unittest.TestCase):
    def tearDown(self):
        from cogmatplot.debug_views import close_debug_views

        close_debug_views()

    def test_publish_is_a_noop_until_init(self):
        from cogmatplot.debug_views import debug_image, get_debug_views

        debug_image("missing", np.zeros((2, 2)))
        self.assertIsNone(get_debug_views())

    def test_named_feeds_keep_order_and_can_disable(self):
        from cogmatplot.debug_views import (
            clear_debug_images,
            debug_image,
            init_debug_views,
        )

        view = init_debug_views(cols=2, enabled=True, recreate=True)
        debug_image("people.mask", np.zeros((4, 4), np.uint8))
        debug_image("people.depth", np.ones((4, 4), np.float32))
        self.assertEqual(view.feed_names(), ["people.mask", "people.depth"])
        clear_debug_images()
        self.assertIsNone(view._feeds["people.mask"])
        view.configure(enabled=False)
        debug_image("people.mask", np.ones((4, 4), np.uint8))
        self.assertIsNone(view._feeds["people.mask"])


class TestPly(unittest.TestCase):
    def test_prism_files_roundtrip(self):
        from cogmatplot.ply import load_ply, rectangular_prism, save_ply

        cloud, mesh = rectangular_prism()
        loaded_cloud = load_ply(ROOT / "examples" / "data" / "prism_cloud.ply")
        loaded_mesh = load_ply(ROOT / "examples" / "data" / "prism_mesh.ply")
        self.assertEqual(loaded_cloud.points.shape, cloud.points.shape)
        self.assertEqual(loaded_cloud.colors.shape, cloud.colors.shape)
        self.assertIsNone(loaded_cloud.faces)
        np.testing.assert_allclose(loaded_cloud.points, cloud.points, rtol=0, atol=1e-4)
        np.testing.assert_array_equal(loaded_cloud.colors, cloud.colors)
        np.testing.assert_array_equal(loaded_mesh.points, mesh.points)
        np.testing.assert_array_equal(loaded_mesh.faces, mesh.faces)
        self.assertEqual(loaded_mesh.faces.shape, (12, 3))

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "box.ply"
            save_ply(path, mesh.points, faces=mesh.faces)
            again = load_ply(path)
        np.testing.assert_array_equal(again.faces, mesh.faces)
        np.testing.assert_allclose(again.points, mesh.points, atol=1e-5)

    def test_plot3_cloud_and_plot_mesh(self):
        from cogmatplot.ply import load_ply

        fm = FigureManager(2, 2, pause=False, mode="inline")
        try:
            cloud = load_ply(ROOT / "examples" / "data" / "prism_cloud.ply")
            mesh = load_ply(ROOT / "examples" / "data" / "prism_mesh.ply")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                x, y, z = cloud.points.T
                fm.plot3(0, 0, x, y, z, ".", colors=cloud.colors, point_size=4)
                fm.plot_mesh(0, 1, mesh.points, mesh.faces, style="wireframe")
                fm.plot_mesh(1, 0, mesh.points, mesh.faces, style="surface", color="lightsteelblue")
                fm.plot_mesh(1, 1, mesh.points, mesh.faces, style="both", color="lightsteelblue")
            self.assertNotIn("kind", fm._cells[(0, 0)].cloud.layers[0])
            original = cloud.points
            stored = fm._cells[(0, 0)].cloud.layers[0]["points"]
            self.assertTrue(np.shares_memory(stored, original))
            for key, style in (((0, 1), "wireframe"), ((1, 0), "surface"), ((1, 1), "both")):
                layer = fm._cells[key].cloud.layers[0]
                self.assertEqual(layer["kind"], "mesh")
                self.assertEqual(layer["style"], style)
                self.assertEqual(fm._axes[key[0]][key[1]].name, "3d")
        finally:
            fm.closeAll()

    def test_subplot_hands_out_tile_axes(self):
        fm = FigureManager(2, 2, pause=False, mode="inline")
        try:
            axes = [[fm.subplot(r, c) for c in range(2)] for r in range(2)]
            self.assertIs(axes[0][1], fm._axes[0][1])
            axes[1][0].plot([0, 1], [1, 0])
            self.assertEqual(len(fm._axes[1][0].lines), 1)
            self.assertEqual(fm.subplot(1, 1, projection="3d").name, "3d")
            fm.drawnow()
        finally:
            fm.closeAll()

    def test_as_ply_reads_loader_structures(self):
        from cogmatplot.ply import as_ply, is_cloud, load_ply

        cloud = load_ply(ROOT / "examples" / "data" / "prism_cloud.ply")
        n = len(cloud.points)
        rgb = cloud.colors.astype(float)

        class Struct(dict):
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError as e:
                    raise AttributeError(key) from e

        abd = Struct(
            vertices=cloud.points.copy(),
            normals=np.zeros((n, 3)),
            alpha=np.full(n, 255.0),
            red=rgb[:, 0], green=rgb[:, 1], blue=rgb[:, 2],
            curvature=np.zeros(n),
        )
        table = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        for i, key in enumerate("xyz"):
            table[key] = cloud.points[:, i]
        for i, key in enumerate(("red", "green", "blue")):
            table[key] = cloud.colors[:, i]
        open3d_like = SimpleNamespace(points=cloud.points, colors=cloud.colors / 255.0)
        xyz_dict = {"x": cloud.points[:, 0], "y": cloud.points[:, 1], "z": cloud.points[:, 2],
                    "colors": cloud.colors}

        for source in (abd, table, open3d_like, xyz_dict, cloud):
            self.assertTrue(is_cloud(source))
            ply = as_ply(source)
            np.testing.assert_allclose(ply.points, cloud.points, atol=1e-4)
            self.assertEqual(ply.colors.dtype, np.uint8)
            np.testing.assert_array_equal(ply.colors, cloud.colors)
        self.assertEqual(as_ply(abd).normals.shape, (n, 3))
        self.assertFalse(is_cloud(cloud.points))
        self.assertIsNone(as_ply(cloud.points).colors)

        mesh = as_ply(load_ply(ROOT / "examples" / "data" / "prism_mesh.ply"))
        self.assertEqual(mesh.faces.shape, (12, 3))

    def test_plot3_and_plot_mesh_take_whole_structures(self):
        from cogmatplot.ply import load_ply

        cloud = load_ply(ROOT / "examples" / "data" / "prism_cloud.ply")
        mesh = load_ply(ROOT / "examples" / "data" / "prism_mesh.ply")
        abd = {"vertices": cloud.points, "red": cloud.colors[:, 0].astype(float),
               "green": cloud.colors[:, 1].astype(float), "blue": cloud.colors[:, 2].astype(float)}
        fm = FigureManager(1, 2, pause=False, mode="inline")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fm.plot3(0, 0, abd, point_size=4)
                fm.plot_mesh(0, 1, mesh, style="both")
            pts, colors = coerce_cloud(fm._cells[(0, 0)].cloud.layers[0]["points"])
            np.testing.assert_array_equal(colors, cloud.colors)
            self.assertEqual(len(pts), len(cloud.points))
            layer = fm._cells[(0, 1)].cloud.layers[0]
            self.assertEqual(layer["kind"], "mesh")
            self.assertEqual(np.asarray(layer["faces"]).shape, (12, 3))
        finally:
            fm.closeAll()


class TestServer(unittest.TestCase):
    def test_client_roundtrip(self):
        from threading import Event, Thread

        import zmq  # noqa: F401

        from cogmatplot.client import PlotClient
        from cogmatplot.server import run_server

        fm = FigureManager(2, 2, pause=False, mode="inline")
        port = _free_port()
        stop = Event()
        ready = Event()
        thread = Thread(target=run_server, args=(fm, port, stop), kwargs={"ready": ready}, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(3), "server did not bind")
        client = PlotClient("127.0.0.1", port)
        try:
            image = np.zeros((6, 8, 3), np.uint8)
            image[..., 2] = 255
            client.imshow(0, 0, image, title="remote")
            client.plot(1, 0, [0, 1, 2], [0, 1, 0], "o-")
            client.hold(1, 0, "on")
            client.plot(1, 0, [0, 1, 2], [1, 0, 1], "-", hold=True)
            pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.4], [0.3, 1.0, 0.8]])
            rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
            cloud = client.figure3(0, 1, title="cloud")
            cloud.scatter3(pts, colors=rgb, point_size=4)
            cloud.scatter3(pts[:1], color="silver", point_size=8)
            cloud.view_xy()
            client.drawnow()
            ok = False
            for _ in range(80):
                axes = fm._axes
                cell = fm._cells.get((0, 1)) if axes else None
                if (
                    axes
                    and axes[0][0].images
                    and len(axes[1][0].lines) >= 2
                    and cell is not None
                    and cell.cloud is not None
                    and len(cell.cloud.layers) >= 2
                ):
                    ok = True
                    break
                stop.wait(0.05)
            self.assertTrue(ok, "server did not apply imshow and both plot calls")
            shown = np.asarray(axes[0][0].images[0].get_array())
            self.assertEqual(int(np.max(shown[..., 2])), 255)
            self.assertEqual(axes[0][0].get_title(), "remote")
            cloud_cell = fm._cells[(0, 1)]
            self.assertEqual(cloud_cell.kind, "3d")
            self.assertEqual(len(cloud_cell.cloud.layers), 2)
            self.assertEqual(cloud_cell.cloud.title, "cloud")
            self.assertIsNone(client.poll_event(timeout=10))
            linked = FigureManager.connect("127.0.0.1", port)
            self.assertIsInstance(linked, PlotClient)
            linked.close()
        finally:
            client.close()
            stop.set()
            thread.join(2)
            fm.closeAll()

    def test_client_sends_cloud_structures_as_arrays(self):
        from threading import Event, Thread

        from cogmatplot.client import PlotClient
        from cogmatplot.ply import load_ply
        from cogmatplot.server import run_server

        cloud = load_ply(ROOT / "examples" / "data" / "prism_cloud.ply")
        mesh = load_ply(ROOT / "examples" / "data" / "prism_mesh.ply")
        rgb = cloud.colors.astype(float)
        abd = {"vertices": cloud.points, "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2]}

        fm = FigureManager(1, 2, pause=False, mode="inline")
        port = _free_port()
        stop = Event()
        ready = Event()
        thread = Thread(target=run_server, args=(fm, port, stop), kwargs={"ready": ready}, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(3), "server did not bind")
        client = PlotClient("127.0.0.1", port)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                client.plot3(0, 0, abd, point_size=4)
                client.plot_mesh(0, 1, mesh, style="wireframe")
                cells = {}
                for _ in range(80):
                    cells = {key: fm._cells.get(key) for key in ((0, 0), (0, 1))}
                    if all(cell is not None and cell.cloud is not None and cell.cloud.layers for cell in cells.values()):
                        break
                    stop.wait(0.05)
            points_layer = cells[(0, 0)].cloud.layers[0]
            self.assertIsInstance(points_layer["points"], np.ndarray)
            np.testing.assert_allclose(points_layer["points"], cloud.points)
            np.testing.assert_array_equal(points_layer["colors"], cloud.colors)
            mesh_layer = cells[(0, 1)].cloud.layers[0]
            self.assertEqual(np.asarray(mesh_layer["faces"]).shape, (12, 3))
        finally:
            client.close()
            stop.set()
            thread.join(2)
            fm.closeAll()

    def test_make_scene_shapes(self):
        data = make(n=20)
        self.assertEqual(data["pts"].shape, (20, 3))
        self.assertEqual(data["rgb"].shape, (20, 3))
        self.assertEqual(data["image"].shape[2], 3)
        self.assertEqual(len(data["helix"]), 3)


if __name__ == "__main__":
    unittest.main()
