"""Nodes converting between ComfyUI's native MESH type and trimesh geometry."""

from comfy_api.latest import Types, io

from . import convert

# One shared type handle so both nodes advertise the same TRIMESH connection.
TrimeshType = io.Custom("TRIMESH")


class MeshToTrimesh(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MeshTrimesh_MeshToTrimesh",
            display_name="Mesh to Trimesh",
            category="3d/trimesh",
            description=(
                "Convert a ComfyUI MESH into trimesh geometry for the Python trimesh ecosystem. "
                "A single-item mesh becomes a trimesh.Trimesh; a batch becomes a trimesh.Scene "
                "with one geometry per item. UVs, vertex colors and normals are carried over; "
                "textures and material factors have no place on trimesh geometry and are dropped."
            ),
            search_aliases=["trimesh", "mesh to trimesh", "convert mesh to trimesh", "export trimesh"],
            is_experimental=True,
            inputs=[
                io.Mesh.Input("mesh", tooltip="Mesh to convert"),
                io.Boolean.Input(
                    "process", default=False,
                    tooltip="Let trimesh merge duplicate vertices and drop degenerate faces. "
                            "Off converts the mesh exactly as it is.",
                ),
            ],
            outputs=[TrimeshType.Output(display_name="trimesh")],
        )

    @classmethod
    def execute(cls, mesh, process):
        items = convert.extract_items(mesh)
        return io.NodeOutput(convert.items_to_geometry(items, process))


class TrimeshToMesh(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MeshTrimesh_TrimeshToMesh",
            display_name="Trimesh to Mesh",
            category="3d/trimesh",
            description=(
                "Convert trimesh geometry into a ComfyUI MESH. Accepts a trimesh.Trimesh, a "
                "trimesh.Scene or a list of either; each geometry becomes one batch item, so a "
                "Scene keeps its parts separate. UVs, vertex colors and normals are carried over."
            ),
            search_aliases=["trimesh to mesh", "convert trimesh", "import trimesh"],
            is_experimental=True,
            inputs=[
                TrimeshType.Input(
                    "trimesh",
                    tooltip="trimesh.Trimesh, trimesh.Scene, or a list of them",
                ),
            ],
            outputs=[io.Mesh.Output(display_name="mesh")],
        )

    @classmethod
    def execute(cls, trimesh):
        geometries = convert.flatten_geometry(trimesh)
        packed = convert.pack_items([convert.geometry_to_item(geometry) for geometry in geometries])
        return io.NodeOutput(Types.MESH(
            vertices=packed.vertices,
            faces=packed.faces,
            uvs=packed.uvs,
            vertex_colors=packed.vertex_colors,
            normals=packed.normals,
            vertex_counts=packed.vertex_counts,
            face_counts=packed.face_counts,
        ))
