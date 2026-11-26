"""Binary PLY loader and a small rectangular-prism sample.

Binary little-endian files are read with ``numpy.frombuffer`` after the
header. That is the fast path: the vertex block is one array, not a Python
loop. Vertex colors may be ``red/green/blue`` or ``diffuse_red/green/blue``.
Faces are triangle or quad lists (``property list uchar int vertex_indices``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

_SCALAR = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}

PathLike = Union[str, Path]


@dataclass
class Ply:
    """Points are ``(N, 3)``. ``faces`` is ``(M, 3)`` when the file has a mesh."""

    points: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None


def load_ply(path: PathLike) -> Ply:
    """Load a PLY file. Binary little-endian is the fast path."""
    data = Path(path).read_bytes()
    header, offset = _split_header(data)
    elements, endian = _parse_header(header)
    buf = data[offset:]
    vertices = None
    faces = None
    cursor = 0
    for element in elements:
        if element["name"] == "vertex":
            vertices, cursor = _read_fixed(buf, cursor, element, endian)
        elif element["name"] == "face":
            faces, cursor = _read_faces(buf, cursor, element, endian)
        else:
            cursor = _skip_element(buf, cursor, element, endian)
    if vertices is None:
        raise ValueError(f"{path} has no vertex element")
    names = vertices.dtype.names
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64, copy=False)
    return Ply(points=points, colors=_colors(vertices, names), normals=_normals(vertices, names), faces=faces)


def as_ply(cloud) -> Ply:
    """Turn a cloud from another loader into a :class:`Ply`.

    Accepts a path, a :class:`Ply`, an ``(N, 3)`` array, a structured array
    or plyfile element with ``x/y/z`` fields, or any dict or object with
    ``points``/``vertices`` or ``x, y, z``; ``colors`` or ``red, green, blue``;
    ``normals`` or ``nx, ny, nz``; and ``faces``/``triangles``. That covers
    ``abd3d_algo.load_cloud``, ``plyfile.PlyData`` and Open3D clouds. Colors
    come back as ``uint8`` 0-255.
    """
    if isinstance(cloud, Ply):
        return cloud
    if isinstance(cloud, (str, Path)):
        return load_ply(cloud)
    if not isinstance(cloud, dict) and hasattr(cloud, "elements"):
        names = [element.name for element in cloud.elements]
        ply = _from_table(cloud["vertex"].data)
        if "face" in names:
            ply.faces = _faces(cloud["face"].data["vertex_indices"])
        return ply
    if getattr(getattr(cloud, "data", None), "dtype", None) is not None:
        cloud = cloud.data
    if isinstance(cloud, np.ndarray):
        if cloud.dtype.names:
            return _from_table(cloud)
        pts = np.asarray(cloud, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("a plain array cloud must have shape (N, 3)")
        return Ply(points=pts)

    points = _first(cloud, "points", "vertices")
    if points is None:
        xyz = [_field(cloud, key) for key in ("x", "y", "z")]
        if any(v is None for v in xyz):
            raise ValueError("cloud needs points, vertices, or x, y, z")
        points = np.column_stack(xyz)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    colors = _first(cloud, "colors", "colours")
    if colors is None:
        colors = _stack(cloud, ("red", "green", "blue")) or _stack(cloud, ("diffuse_red", "diffuse_green", "diffuse_blue"))
    normals = _first(cloud, "normals")
    if normals is None:
        normals = _stack(cloud, ("nx", "ny", "nz"))
    return Ply(
        points=pts,
        colors=_cloud_colors(colors, len(pts)),
        normals=_cloud_normals(normals, len(pts)),
        faces=_faces(_first(cloud, "faces", "triangles")),
    )


def is_cloud(obj) -> bool:
    """True for structures :func:`as_ply` reads; False for plain arrays and lists."""
    if obj is None or isinstance(obj, (str, Path, list, tuple)):
        return False
    if isinstance(obj, np.ndarray):
        return bool(obj.dtype.names)
    if isinstance(obj, (Ply, dict)) or hasattr(obj, "elements"):
        return True
    data = getattr(obj, "data", None)
    if getattr(getattr(data, "dtype", None), "names", None):
        return True
    return hasattr(obj, "points") or hasattr(obj, "vertices") or all(hasattr(obj, key) for key in ("x", "y", "z"))


def _field(obj, key):
    if isinstance(obj, np.ndarray):
        return obj[key] if obj.dtype.names and key in obj.dtype.names else None
    if isinstance(obj, dict):
        return obj.get(key)
    try:
        return obj[key]
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return getattr(obj, key, None)


def _first(obj, *keys):
    for key in keys:
        value = _field(obj, key)
        if value is not None:
            return value
    return None


def _stack(obj, keys):
    parts = [_field(obj, key) for key in keys]
    if any(part is None for part in parts):
        return None
    return [np.asarray(part) for part in parts]


def _from_table(table):
    names = table.dtype.names
    for key in ("x", "y", "z"):
        if key not in names:
            raise ValueError("structured cloud needs x, y, z fields")
    points = np.column_stack([table["x"], table["y"], table["z"]]).astype(np.float64, copy=False)
    return Ply(points=points, colors=_colors(table, names), normals=_normals(table, names))


def _cloud_colors(colors, n):
    if colors is None:
        return None
    col = np.column_stack(colors) if isinstance(colors, list) else np.asarray(colors)
    if col.size == 0:
        return None
    if col.ndim == 2 and col.shape[0] == 3 and col.shape[1] == n and n != 3:
        col = col.T
    if col.shape != (n, 3):
        return None
    if col.dtype == np.uint8:
        return col
    return _as_u8(col, n)


def _cloud_normals(normals, n):
    if normals is None:
        return None
    nrm = np.column_stack(normals) if isinstance(normals, list) else np.asarray(normals)
    if nrm.shape != (n, 3):
        return None
    return nrm.astype(np.float64, copy=False)


def _faces(faces):
    if faces is None:
        return None
    arr = np.asarray(faces)
    if arr.dtype == object:
        sizes = {len(face) for face in arr}
        width = 4 if sizes == {4} else 3
        arr = np.array([np.asarray(face)[:width] for face in arr], dtype=np.int64)
    if arr.size == 0:
        return None
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    width = 4 if arr.shape[1] == 4 else 3
    return arr[:, :width].astype(np.int64, copy=False)


def save_ply(path: PathLike, points, colors=None, normals=None, faces=None) -> None:
    """Write a binary little-endian PLY file."""
    pts = np.asarray(points, dtype="<f4")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    cols = None if colors is None else _as_u8(colors, len(pts))
    norms = None if normals is None else np.asarray(normals, dtype="<f4")
    tris = None if faces is None else np.asarray(faces, dtype="<i4")
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {len(pts)}",
             "property float x", "property float y", "property float z"]
    if cols is not None:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    if norms is not None:
        lines += ["property float nx", "property float ny", "property float nz"]
    if tris is not None:
        lines += [f"element face {len(tris)}", "property list uchar int vertex_indices"]
    lines.append("end_header")
    chunks = ["\n".join(lines).encode("ascii") + b"\n"]
    names = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if cols is not None:
        names += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if norms is not None:
        names += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    table = np.empty(len(pts), dtype=names)
    table["x"], table["y"], table["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if cols is not None:
        table["red"], table["green"], table["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    if norms is not None:
        table["nx"], table["ny"], table["nz"] = norms[:, 0], norms[:, 1], norms[:, 2]
    chunks.append(table.tobytes())
    if tris is not None:
        # One count byte plus 3 or 4 little-endian ints per face. No struct padding.
        width = 4 if tris.shape[1] == 4 else 3
        blob = np.empty((len(tris), 1 + 4 * width), dtype=np.uint8)
        blob[:, 0] = width
        blob[:, 1:] = np.ascontiguousarray(tris[:, :width], dtype="<i4").view(np.uint8).reshape(len(tris), 4 * width)
        chunks.append(blob.tobytes())
    Path(path).write_bytes(b"".join(chunks))


def rectangular_prism(size=(200.0, 120.0, 80.0), step=10.0):
    """A box as a colored point cloud and as a 12-triangle mesh.

    ``size`` is ``(length, width, height)``. ``step`` is the cloud spacing.
    Both results are :class:`Ply` objects. The cloud samples the six faces.
    The mesh is the eight corners and the shaded sides.
    """
    length, width, height = (float(v) for v in size)
    step = float(step)
    corners = np.array([
        [0, 0, 0],
        [length, 0, 0],
        [length, width, 0],
        [0, width, 0],
        [0, 0, height],
        [length, 0, height],
        [length, width, height],
        [0, width, height],
    ], dtype=np.float64)
    # Outward winding, two triangles per side.
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    samples = []
    colors = []
    palette = np.array([
        [180, 80, 70],
        [70, 140, 190],
        [80, 170, 90],
        [200, 160, 60],
        [140, 90, 180],
        [90, 90, 90],
    ], dtype=np.uint8)
    spans = (
        (np.array([0, 0, 0.0]), np.array([length, 0, 0.0]), np.array([0, width, 0.0])),
        (np.array([0, 0, height]), np.array([length, 0, 0.0]), np.array([0, width, 0.0])),
        (np.array([0, 0, 0.0]), np.array([length, 0, 0.0]), np.array([0, 0, height])),
        (np.array([0, width, 0.0]), np.array([length, 0, 0.0]), np.array([0, 0, height])),
        (np.array([0, 0, 0.0]), np.array([0, width, 0.0]), np.array([0, 0, height])),
        (np.array([length, 0, 0.0]), np.array([0, width, 0.0]), np.array([0, 0, height])),
    )
    for (origin, axis_u, axis_v), color in zip(spans, palette):
        nu = max(2, int(round(np.linalg.norm(axis_u) / step)) + 1)
        nv = max(2, int(round(np.linalg.norm(axis_v) / step)) + 1)
        uu, vv = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="ij")
        pts = origin + uu[..., None] * axis_u + vv[..., None] * axis_v
        flat = pts.reshape(-1, 3)
        samples.append(flat)
        colors.append(np.broadcast_to(color, (len(flat), 3)).copy())
    cloud = Ply(points=np.vstack(samples), colors=np.vstack(colors))
    mesh = Ply(points=corners, faces=faces)
    return cloud, mesh


def _split_header(data: bytes):
    marker = b"end_header"
    at = data.find(marker)
    if at < 0:
        raise ValueError("PLY header has no end_header")
    end = at + len(marker)
    if data[end:end + 1] == b"\r":
        end += 1
    if data[end:end + 1] == b"\n":
        end += 1
    return data[:at].decode("ascii", errors="replace"), end


def _parse_header(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("comment")]
    if not lines or lines[0] != "ply":
        raise ValueError("not a PLY file")
    fmt = lines[1]
    if fmt.startswith("format binary_little_endian"):
        endian = "<"
    elif fmt.startswith("format binary_big_endian"):
        endian = ">"
    elif fmt.startswith("format ascii"):
        endian = "ascii"
    else:
        raise ValueError(f"unsupported PLY format: {fmt}")
    elements = []
    current = None
    for line in lines[2:]:
        parts = line.split()
        if parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                current["props"].append(("list", _dtype(parts[2], endian), _dtype(parts[3], endian), parts[4]))
            else:
                current["props"].append((parts[2], _dtype(parts[1], endian)))
    return elements, endian


def _dtype(name: str, endian: str) -> str:
    base = _SCALAR[name]
    if endian == ">" and base.startswith("<"):
        return ">" + base[1:]
    if endian == "ascii":
        return base[1:] if base[:1] in "<>" else base
    return base


def _read_fixed(buf, cursor, element, endian):
    if any(prop[0] == "list" for prop in element["props"]):
        raise ValueError(f"{element['name']} has a list property; only face lists are supported")
    count = element["count"]
    if endian == "ascii":
        rows = []
        text = buf[cursor:].decode("ascii")
        lines = text.splitlines()
        for line in lines[:count]:
            rows.append([float(x) for x in line.split()])
        dtype = np.dtype([(name, np.dtype(dt)) for name, dt in element["props"]])
        table = np.empty(count, dtype=dtype)
        raw = np.asarray(rows, dtype=np.float64)
        for i, (name, dt) in enumerate(element["props"]):
            table[name] = raw[:, i].astype(dt)
        consumed = sum(len(line) + 1 for line in lines[:count])
        return table, cursor + consumed
    dtype = np.dtype([(name, np.dtype(dt)) for name, dt in element["props"]])
    nbytes = count * dtype.itemsize
    raw = np.frombuffer(buf, dtype=np.uint8, count=nbytes, offset=cursor).copy()
    table = raw.view(dtype).reshape(count)
    return table, cursor + nbytes


def _read_faces(buf, cursor, element, endian):
    lists = [prop for prop in element["props"] if prop[0] == "list"]
    if len(lists) != 1:
        raise ValueError("face element needs one list property")
    _tag, count_dt, index_dt, _name = lists[0]
    count = element["count"]
    if endian == "ascii":
        text = buf[cursor:].decode("ascii")
        lines = text.splitlines()
        faces = []
        for line in lines[:count]:
            nums = [int(x) for x in line.split()]
            n = nums[0]
            faces.append(nums[1:1 + n])
        width = max(len(face) for face in faces)
        out = np.full((count, width), -1, dtype=np.int64)
        for i, face in enumerate(faces):
            out[i, :len(face)] = face
        consumed = sum(len(line) + 1 for line in lines[:count])
        if width == 4 and all(len(face) == 4 for face in faces):
            return out, cursor + consumed
        return out[:, :3] if width >= 3 else out, cursor + consumed
    count_size = np.dtype(count_dt).itemsize
    index_size = np.dtype(index_dt).itemsize
    first = int(np.frombuffer(buf, dtype=np.dtype(count_dt), count=1, offset=cursor)[0])
    stride = count_size + first * index_size
    if count and cursor + count * stride <= len(buf):
        # Fixed-stride rows only line up when every face has the same vertex count.
        rows = np.frombuffer(buf, dtype=np.uint8, count=count * stride, offset=cursor).reshape(count, stride)
        counts = rows[:, :count_size].copy().view(np.dtype(count_dt)).ravel()
        if np.all(counts == first):
            indices = rows[:, count_size:].copy().view(np.dtype(index_dt))
            indices = indices.reshape(count, first).astype(np.int64, copy=False)
            return indices, cursor + count * stride
    faces = []
    pos = cursor
    for _ in range(count):
        n = int(np.frombuffer(buf, dtype=np.dtype(count_dt), count=1, offset=pos)[0])
        pos += count_size
        idx = np.frombuffer(buf, dtype=np.dtype(index_dt), count=n, offset=pos).astype(np.int64)
        pos += n * index_size
        faces.append(idx)
    width = max(len(face) for face in faces) if faces else 0
    out = np.full((count, max(width, 3)), -1, dtype=np.int64)
    for i, face in enumerate(faces):
        out[i, :len(face)] = face
    return out[:, :3], pos


def _skip_element(buf, cursor, element, endian):
    if endian == "ascii":
        text = buf[cursor:].decode("ascii")
        lines = text.splitlines()
        return cursor + sum(len(line) + 1 for line in lines[:element["count"]])
    if any(prop[0] == "list" for prop in element["props"]):
        _faces, cursor = _read_faces(buf, cursor, element, endian)
        return cursor
    dtype = np.dtype([(name, np.dtype(dt)) for name, dt in element["props"] if name != "list"])
    return cursor + element["count"] * dtype.itemsize


def _colors(table, names):
    if names is None:
        return None
    triples = (("red", "green", "blue"), ("diffuse_red", "diffuse_green", "diffuse_blue"))
    for keys in triples:
        if all(key in names for key in keys):
            return np.column_stack([table[key] for key in keys]).astype(np.uint8, copy=False)
    return None


def _normals(table, names):
    if names is None or not all(key in names for key in ("nx", "ny", "nz")):
        return None
    return np.column_stack([table["nx"], table["ny"], table["nz"]]).astype(np.float64, copy=False)


def _as_u8(colors, n):
    col = np.asarray(colors)
    if col.shape != (n, 3):
        raise ValueError("colors must have shape (N, 3)")
    if col.dtype != np.uint8 and np.nanmax(col) <= 1.0:
        col = col * 255.0
    return np.clip(np.rint(col), 0, 255).astype(np.uint8)
