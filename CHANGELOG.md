# Changelog

## [0.3.0] - 2026-11-27

### Added

- 3D boxes: `box_corners`, `box_mesh`, `crop_box` (rotated box plus margins), and `plot_box`.
- Quad faces in `plot_mesh`, `load_ply`, and `save_ply`.
- `examples/box_crop.py`; README sections on boxes and direct PyVista access via `.pl`.

### Fixed

- Binary PLY faces load in one pass.
- Inline 3D views accept 0-255 color tuples.

## [0.2.0] - 2026-11-27

### Added

- `as_ply` / `is_cloud`: read clouds from other loaders (abd3d `load_cloud`, plyfile, Open3D, structured arrays, dicts).
- `plot3`, `scatter3`, and `plot_mesh` accept a whole cloud or mesh structure.
- `FigureManager.subplot(row, col)` returns a tile's matplotlib axes.
- `cogmatplot.__version__`.
- Test workflow on Windows for Python 3.10, 3.12, and 3.13.
- Pushing a version tag runs the tests, publishes to PyPI, and creates the GitHub release.
- Project URLs and changelog link in the package metadata.

### Changed

- Requires Python 3.10 or later.
- Pin `pyvistaqt<0.13`; newer versions hang on secondary monitors on Windows.
- `plot3` passes point clouds to PyVista without copying.
- The 3D grid follows the data as layers are added.

### Fixed

- `drawnow` no longer blocks notebook kernels.
- `use("qt")` enables the kernel's Qt loop; `%matplotlib qt` is not needed.
- 0-1 float colors round correctly to `uint8`.

## [0.1.0] - 2026-11-26

First release: `FigureManager` grid with matplotlib, OpenCV, and PyVista tiles on any monitor; Qt and inline modes; MATLAB-style helpers; debug server and client; `debug_views`; PLY load and save; examples.

[0.3.0]: https://github.com/cogency/cogmatplot/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/cogency/cogmatplot/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/cogency/cogmatplot/releases/tag/0.1.0
