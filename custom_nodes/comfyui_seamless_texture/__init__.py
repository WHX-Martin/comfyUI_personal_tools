from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import (
    SeamlessModelPadding,
    SeamlessVAEDecode,
    SeamlessVAEEncode,
    TilePreview,
)
from .materials import SeamlessMaterialsFromAlbedo
from .geometry import (
    SeamlessDelightAlbedo,
    SeamlessDSINENormal,
    SeamlessFlattenNormal,
    SeamlessHeightFromNormal,
    SeamlessMoGeInference,
    SeamlessRoughnessFromHeight,
)
from .pbr import (
    PackPBRChannels,
    SeamlessAOFromHeight,
    SeamlessHeightFromAlbedo,
    SeamlessNormalFromHeight,
    SeamlessRoughnessFromAlbedo,
)
from .upscale import SeamlessResize, SeamlessUpscale

__all__ = ["SeamlessTextureExtension"]


class SeamlessTextureExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SeamlessModelPadding,
            SeamlessVAEDecode,
            SeamlessVAEEncode,
            TilePreview,
            SeamlessHeightFromAlbedo,
            SeamlessNormalFromHeight,
            SeamlessAOFromHeight,
            SeamlessRoughnessFromAlbedo,
            PackPBRChannels,
            SeamlessUpscale,
            SeamlessResize,
            SeamlessMaterialsFromAlbedo,
            SeamlessMoGeInference,
            SeamlessDSINENormal,
            SeamlessFlattenNormal,
            SeamlessHeightFromNormal,
            SeamlessDelightAlbedo,
            SeamlessRoughnessFromHeight,
        ]


async def comfy_entrypoint() -> SeamlessTextureExtension:
    return SeamlessTextureExtension()
