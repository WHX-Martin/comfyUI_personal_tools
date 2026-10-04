from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import ObjectExtractBBoxesToMasks, ObjectExtractPromptsFromVLM

__all__ = ["ObjectExtractExtension"]


class ObjectExtractExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ObjectExtractPromptsFromVLM, ObjectExtractBBoxesToMasks]


async def comfy_entrypoint() -> ObjectExtractExtension:
    return ObjectExtractExtension()
