from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import MeshToTrimesh, TrimeshToMesh


class MeshTrimeshExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MeshToTrimesh, TrimeshToMesh]


async def comfy_entrypoint() -> MeshTrimeshExtension:
    return MeshTrimeshExtension()
