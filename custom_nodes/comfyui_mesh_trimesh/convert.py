"""Conversion between the native ComfyUI MESH type and trimesh geometry.

This module deliberately imports nothing from ComfyUI: it takes plain tensors/arrays in
and gives plain tensors/arrays back, so the conversion can be exercised on its own.
`nodes.py` adapts the `Types.MESH` container to these helpers.

ComfyUI's MESH is a *batch*: vertices are (B, N, 3), faces are (B, M, 3) and the
per-vertex attributes are (B, N, ...). Variable-size batches are zero-padded to a common
N/M and carry the real lengths in `vertex_counts` / `face_counts`. trimesh has no batch
concept, so one mesh item maps to one `trimesh.Trimesh`.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import torch
import trimesh

# ComfyUI's mesh loaders produce float32 positions and int64 face indices; match them so
# meshes round-trip through the rest of the 3D nodes unchanged.
VERTEX_DTYPE = np.float32
FACE_DTYPE = np.int64

# MESH fields trimesh geometry cannot carry. They are reported instead of dropped silently.
UNSUPPORTED_FIELDS = ("texture", "metallic_roughness", "normal_map", "emissive", "material", "tangents")


class MeshItem(NamedTuple):
    """One unbatched mesh, as plain numpy arrays."""

    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray | None = None
    vertex_colors: np.ndarray | None = None
    normals: np.ndarray | None = None


class PackedMesh(NamedTuple):
    """A batched mesh ready to be handed to `Types.MESH`."""

    vertices: torch.Tensor
    faces: torch.Tensor
    uvs: torch.Tensor | None = None
    vertex_colors: torch.Tensor | None = None
    normals: torch.Tensor | None = None
    vertex_counts: torch.Tensor | None = None
    face_counts: torch.Tensor | None = None


def _numpy(value, dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def _optional(value, dtype) -> np.ndarray | None:
    return None if value is None else _numpy(value, dtype)


def _counts(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return [int(item) for item in value]


def _per_item(container, count: int) -> list:
    """View a batched attribute as one entry per item."""
    if container is None:
        return [None] * count
    if isinstance(container, torch.Tensor):
        return [container[i] for i in range(count)]
    return list(container)


def extract_items(mesh) -> list[MeshItem]:
    """Split a MESH into one `MeshItem` per batch entry.

    Handles the padded batched form and the plain list form a few core nodes produce, and
    trims padded rows down to their real length when the mesh carries per-item counts.
    """
    unsupported = [name for name in UNSUPPORTED_FIELDS if getattr(mesh, name, None) is not None]
    if unsupported:
        logging.warning("MeshToTrimesh: dropping %s, which trimesh geometry has no place for",
                        ", ".join(unsupported))

    vertices = mesh.vertices
    if isinstance(vertices, (list, tuple)):
        return _items_from_lists(mesh)

    count = len(vertices)
    vertex_counts = _counts(mesh.vertex_counts)
    face_counts = _counts(mesh.face_counts)

    items = []
    for index in range(count):
        vertex_count = vertex_counts[index] if vertex_counts is not None else None
        face_count = face_counts[index] if face_counts is not None else None
        items.append(MeshItem(
            vertices=_numpy(_row(vertices, index, vertex_count), VERTEX_DTYPE),
            faces=_numpy(_row(mesh.faces, index, face_count), FACE_DTYPE),
            uvs=_optional(_row(mesh.uvs, index, vertex_count), VERTEX_DTYPE),
            vertex_colors=_optional(_row(mesh.vertex_colors, index, vertex_count), VERTEX_DTYPE),
            normals=_optional(_row(mesh.normals, index, vertex_count), VERTEX_DTYPE),
        ))
    return items


def _row(container, index: int, length: int | None):
    if container is None:
        return None
    return container[index] if length is None else container[index, :length]


def _items_from_lists(mesh) -> list[MeshItem]:
    vertices = list(mesh.vertices)
    count = len(vertices)
    faces = _per_item(mesh.faces, count)
    uvs = _per_item(mesh.uvs, count)
    colors = _per_item(mesh.vertex_colors, count)
    normals = _per_item(mesh.normals, count)
    return [MeshItem(
        vertices=_numpy(vertices[index], VERTEX_DTYPE),
        faces=_numpy(faces[index], FACE_DTYPE),
        uvs=_optional(uvs[index], VERTEX_DTYPE),
        vertex_colors=_optional(colors[index], VERTEX_DTYPE),
        normals=_optional(normals[index], VERTEX_DTYPE),
    ) for index in range(count)]


def item_to_trimesh(item: MeshItem, process: bool = False) -> trimesh.Trimesh:
    """Build one `trimesh.Trimesh`, carrying UVs, vertex colors and normals when present.

    `process` is handed to trimesh, which then merges duplicate vertices and drops
    degenerate faces. Attributes are attached through the constructor so trimesh reindexes
    them along with the vertices instead of leaving them pointing at the old numbering.
    """
    geometry = trimesh.Trimesh(
        vertices=item.vertices,
        faces=item.faces,
        visual=_visual(item),
        process=process,
    )
    _prune_stale_visual(geometry, process)
    # With `process` trimesh may have changed the vertex count, so only restore normals that
    # still line up. Otherwise trimesh derives them from the geometry on demand.
    if item.normals is not None and len(item.normals) == len(geometry.vertices):
        geometry.vertex_normals = item.normals
    return geometry


def _visual(item: MeshItem) -> trimesh.visual.Visuals | None:
    vertex_count = len(item.vertices)
    uv = item.uvs if item.uvs is not None and len(item.uvs) == vertex_count else None
    colors = item.vertex_colors if item.vertex_colors is not None and len(item.vertex_colors) == vertex_count else None
    return _make_visual(uv, None if colors is None else _to_rgba(colors))


def _make_visual(uv: np.ndarray | None, colors: np.ndarray | None) -> trimesh.visual.Visuals | None:
    """UVs need a texture visual; colors on their own get a color visual."""
    visual = None
    if uv is not None:
        visual = trimesh.visual.TextureVisuals(uv=uv)
    if colors is not None:
        if visual is None:
            visual = trimesh.visual.ColorVisuals(vertex_colors=colors)
        else:
            visual.vertex_colors = colors
    return visual


def _prune_stale_visual(geometry: trimesh.Trimesh, process: bool) -> None:
    """Keep only the per-vertex attributes that still line up with the vertex list.

    Merging reindexes UVs but not always vertex colors, which can leave an attribute longer
    than the vertices it describes. Handing that out would be a corrupt trimesh, so the
    stale attribute is dropped and reported rather than passed on.
    """
    visual = geometry.visual
    if not visual.defined:
        return
    vertex_count = len(geometry.vertices)
    uv = getattr(visual, "uv", None)
    colors = getattr(visual, "vertex_colors", None)
    keep_uv = uv if uv is not None and len(uv) == vertex_count else None
    keep_colors = colors if colors is not None and len(colors) == vertex_count else None
    if keep_uv is uv and keep_colors is colors:
        return
    geometry.visual = _make_visual(keep_uv, keep_colors)
    if process:
        lost = [name for name, before, after in
                (("UVs", uv, keep_uv), ("vertex colors", colors, keep_colors))
                if before is not None and after is None]
        logging.warning("MeshToTrimesh: merging vertices left %s out of step with the vertex "
                        "list, so they were dropped. Turn 'process' off to keep them.",
                        " and ".join(lost))


def _to_rgba(colors: np.ndarray) -> np.ndarray:
    """MESH keeps vertex colors as float 0-1 with 3 or 4 channels; trimesh wants uint8 RGBA."""
    rgba = np.round(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    if rgba.shape[1] == 3:
        rgba = np.concatenate([rgba, np.full((len(rgba), 1), 255, np.uint8)], axis=1)
    return rgba


def items_to_geometry(items: list[MeshItem], process: bool = False):
    """One item becomes a `Trimesh`; a batch becomes a `Scene` holding one geometry per item."""
    geometries = [item_to_trimesh(item, process) for item in items]
    if len(geometries) == 1:
        return geometries[0]
    return trimesh.Scene({f"mesh_{index:03d}": geometry for index, geometry in enumerate(geometries)})


def flatten_geometry(value) -> list[trimesh.Trimesh]:
    """Normalize a Trimesh, a Scene or a list of either into a flat list of meshes.

    A Scene is dumped rather than read through `Scene.geometry`, so per-geometry transforms
    from an imported scene graph are baked into the vertices.
    """
    if isinstance(value, trimesh.Trimesh):
        return [value]
    if isinstance(value, trimesh.Scene):
        dumped = value.dump(concatenate=False)
        return [dumped] if isinstance(dumped, trimesh.Trimesh) else list(dumped)
    if isinstance(value, (list, tuple)):
        meshes: list[trimesh.Trimesh] = []
        for entry in value:
            meshes.extend(flatten_geometry(entry))
        return meshes
    raise ValueError(
        "TrimeshToMesh: expected a trimesh.Trimesh, a trimesh.Scene or a list of them, "
        f"got {type(value).__name__}"
    )


def geometry_to_item(geometry: trimesh.Trimesh) -> MeshItem:
    """Read one `trimesh.Trimesh` back into a `MeshItem`."""
    vertices = _numpy(geometry.vertices, VERTEX_DTYPE)
    return MeshItem(
        vertices=vertices,
        faces=_numpy(geometry.faces, FACE_DTYPE),
        uvs=_geometry_uv(geometry, len(vertices)),
        vertex_colors=_geometry_colors(geometry, len(vertices)),
        # trimesh always has vertex normals, so they are handed on: ComfyUI's SaveGLB writes
        # them and its own loaders read them back.
        normals=_numpy(geometry.vertex_normals, VERTEX_DTYPE) if len(geometry.faces) > 0 else None,
    )


def _defined_visual(geometry: trimesh.Trimesh):
    """The visual only carries real attributes when it is defined: on a plain mesh trimesh
    synthesises placeholder values, so reading them would invent vertex colors out of nothing."""
    visual = geometry.visual
    return visual if visual is not None and visual.defined else None


def _geometry_uv(geometry: trimesh.Trimesh, vertex_count: int) -> np.ndarray | None:
    visual = _defined_visual(geometry)
    uv = getattr(visual, "uv", None) if visual is not None else None
    if uv is None or len(uv) != vertex_count:
        return None
    return _numpy(uv, VERTEX_DTYPE)


def _geometry_colors(geometry: trimesh.Trimesh, vertex_count: int) -> np.ndarray | None:
    visual = _defined_visual(geometry)
    colors = getattr(visual, "vertex_colors", None) if visual is not None else None
    if colors is None or len(colors) != vertex_count:
        return None
    colors = _numpy(colors, VERTEX_DTYPE) / 255.0
    # Drop a fully opaque alpha channel so plain RGB meshes match what ComfyUI's own loaders emit.
    if colors.shape[1] == 4 and bool(np.all(colors[:, 3] == 1.0)):
        colors = colors[:, :3]
    return colors


def pack_items(items: list[MeshItem]) -> PackedMesh:
    """Stack mesh items back into the batched form `Types.MESH` expects.

    Uniform batches stack directly. Ragged ones are zero-padded and record their real
    lengths in `vertex_counts` / `face_counts`, the same way ComfyUI's variable-size
    batches do. Per-vertex attributes are padded to a common vertex count; a missing
    attribute is filled with zeros, or with opaque white for colors, and normals are only
    carried when every item has them.
    """
    if not items:
        raise ValueError("TrimeshToMesh: the input held no geometry to convert")

    vertex_lengths = [len(item.vertices) for item in items]
    face_lengths = [len(item.faces) for item in items]
    ragged = len(set(vertex_lengths)) > 1 or len(set(face_lengths)) > 1

    uvs = _uniform_uvs(items)
    colors = _uniform_colors(items)
    # Normals are only carried when every item has one and each still matches its vertices.
    normals = [item.normals for item in items]
    if not all(normal is not None and len(normal) == len(item.vertices)
               for item, normal in zip(items, normals)):
        normals = None

    return PackedMesh(
        vertices=_stack([item.vertices for item in items], vertex_lengths, ragged, VERTEX_DTYPE, 0),
        faces=_stack([item.faces for item in items], face_lengths, ragged, FACE_DTYPE, 0),
        uvs=_stack(uvs, vertex_lengths, ragged, VERTEX_DTYPE, 0) if uvs is not None else None,
        vertex_colors=_stack(colors, vertex_lengths, ragged, VERTEX_DTYPE, 1) if colors is not None else None,
        normals=_stack(normals, vertex_lengths, ragged, VERTEX_DTYPE, 0) if normals is not None else None,
        # MESH asserts that both counts are given together, so they are set only as a pair.
        vertex_counts=_counts_tensor(vertex_lengths) if ragged else None,
        face_counts=_counts_tensor(face_lengths) if ragged else None,
    )


def _stack(arrays: list[np.ndarray], lengths: list[int], ragged: bool, dtype, fill) -> torch.Tensor:
    if not ragged:
        return torch.from_numpy(np.stack([np.asarray(array, dtype=dtype) for array in arrays], axis=0))
    # Ragged arrays cannot be stacked to read their width, so take it from the first item.
    width = np.asarray(arrays[0], dtype=dtype).shape[1]
    packed = np.full((len(arrays), max(lengths), width), fill, dtype=dtype)
    for index, array in enumerate(arrays):
        packed[index, :len(array)] = array
    return torch.from_numpy(packed)


def _counts_tensor(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(lengths, dtype=torch.int64)


def _uniform_uvs(items: list[MeshItem]) -> list[np.ndarray] | None:
    if all(item.uvs is None for item in items):
        return None
    return [item.uvs if item.uvs is not None else np.zeros((len(item.vertices), 2), VERTEX_DTYPE)
            for item in items]


def _uniform_colors(items: list[MeshItem]) -> list[np.ndarray] | None:
    present = [item.vertex_colors for item in items if item.vertex_colors is not None]
    if not present:
        return None
    width = max(colors.shape[1] for colors in present)
    uniform = []
    for item in items:
        colors = item.vertex_colors
        if colors is None:
            uniform.append(np.ones((len(item.vertices), width), VERTEX_DTYPE))
        elif colors.shape[1] < width:
            padding = np.ones((len(colors), width - colors.shape[1]), VERTEX_DTYPE)
            uniform.append(np.concatenate([colors, padding], axis=1))
        else:
            uniform.append(colors)
    return uniform
