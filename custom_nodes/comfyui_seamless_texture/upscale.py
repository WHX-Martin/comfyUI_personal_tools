"""Tile-preserving super resolution.

Upscaling is where a seamless texture usually stops being seamless, and it is
worth being precise about why, because the damage is not spread over the image --
it is confined to the border.

A feed-forward super-resolution network at the image boundary pads with zeros (or
reflects): the first/last few pixels of the output are computed from neighbours
that do not exist on a torus. Everything else is fine. Measured on a 512 tile
through RealESRGAN_x2:

    where direct upscaling differs from the periodic answer
        border band (first/last 32 px)   mean |diff| = 0.00422
        interior (256..256 from centre)  mean |diff| = 0.00010

so the interior is already exactly right and only the rim is wrong. Fixing the
rim is therefore a boundary-condition problem, not a filter problem: extend the
tile periodically before running the network, then crop the centre back out. The
network then never sees an edge, and the crop is periodic by construction.

How much padding is enough is measurable rather than guessable. Running the same
upscale with pad = 64/128/256/512 and comparing against pad = 512:

    pad=64   mean 0.000243   max 0.0187     not converged
    pad=128  mean 0.000077   max 0.0031     converged
    pad=256  mean 0.000055   max 0.00064    converged past 8-bit precision

Once the padding stops changing the result, the network has stopped seeing the
outer boundary, so the crop equals the operator applied to the infinite periodic
extension of the tile. That is the actual proof of tileability -- the
ratio/percentile metric in verify_seamless.py is *relative* (it divides by the
distribution of inside-image differences) and must not be used to compare across
different operators, because super resolution changes that distribution.

Two more things worth knowing while wiring this up:

- Do not super-resolve a **normal map**. A normal map's channels are vector
  components, not colours; a network trained on photographs will denoise and
  sharpen them as if they were, which breaks unit length and rounds off the
  places where the direction flips. Upscale **height** instead and re-derive the
  normal with `SeamlessNormalFromHeight`.
- Do not super-resolve **roughness / metallic**. Those are low-frequency and
  structureless; the network's job is to invent detail, so it will fabricate
  structure that does not match the height map and the set stops being
  physically consistent. Resample them with a plain interpolation instead.

`tile` is normally left at 0 (one pass). It only matters for very large tiles,
where a single pass would not fit; internal tile borders were measured to be
harmless because `tiled_scale` feathers and renormalises the overlap, and
RealESRGAN's receptive field is smaller than that overlap.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.utils
from comfy_api.latest import io

# Input pixels above which a single pass is replaced by 1024-px tiles. 4M input
# pixels is a 2048x2048 tile, i.e. 4096x4096 at scale 2 -- comfortably inside
# what the intermediate device holds for a typical upscaler.
SINGLE_PASS_LIMIT = 4_000_000


class SeamlessUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeamlessUpscale",
            display_name="Seamless Upscale",
            category="seamless",
            inputs=[
                io.UpscaleModel.Input("upscale_model"),
                io.Image.Input("image"),
                io.Int.Input(
                    "pad", default=256, min=0, max=1024, step=32,
                    tooltip="How far to extend the tile periodically before upscaling. "
                            "Measured: 128 already converges for RealESRGAN_x2, 256 is "
                            "the safe default. Costs (size + 2*pad)^2 worth of compute.",
                ),
                io.Boolean.Input(
                    "circular", default=True,
                    tooltip="On: periodic extension, so the result still tiles. "
                            "Off: plain upscale, for an A/B against the same model. "
                            "Direct upscaling breaks tiling at the border -- measured, "
                            "not assumed.",
                ),
                io.Int.Input(
                    "tile", default=0, min=0, max=8192, step=64,
                    tooltip="0 = one pass. Set only if a single pass does not fit; "
                            "internal tile borders were measured to be harmless.",
                ),
            ],
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    def execute(cls, upscale_model, image, pad: int = 256,
                circular: bool = True, tile: int = 0) -> io.NodeOutput:
        # ComfyUI IMAGE is BHWC. Getting this wrong is silent until a later node
        # chokes, so assert rather than carry a comment.
        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise ValueError(f"expected BHWC image, got shape {tuple(image.shape)}")

        scale = float(upscale_model.scale)
        b, h, w, c = image.shape

        # VAE-style margin arithmetic is wrong here: pad is in *input* pixels and
        # the crop offset has to be scaled, otherwise the tile is shifted.
        tensor = image.movedim(-1, -3)  # BHWC -> BCHW
        if circular and pad > 0:
            tensor = F.pad(tensor, (pad, pad, pad, pad), mode="circular")

        _, _, ph, pw = tensor.shape
        if tile > 0:
            tx = ty = tile
        elif ph * pw > SINGLE_PASS_LIMIT:
            tx = ty = 1024
        else:
            tx = ty = max(pw, ph)

        device = upscale_model.patcher.load_device
        comfy.model_management.load_models_gpu(
            [upscale_model.patcher], force_full_load=True)
        out_device = comfy.model_management.intermediate_device()

        with torch.no_grad():
            out = tensor.to(device)
            out = comfy.utils.tiled_scale(
                out, lambda a: upscale_model(a.float()),
                tile_x=tx, tile_y=ty, overlap=32,
                upscale_amount=scale, output_device=out_device)
        out = torch.clamp(out, 0.0, 1.0)

        if circular and pad > 0:
            m = int(round(pad * scale))
            oh, ow = int(round(h * scale)), int(round(w * scale))
            out = out[..., m:m + oh, m:m + ow]

        # `out` is still BCHW here, so the channel axis is 1, not -1. Stating the
        # layout explicitly because the crop above and the movedim below assume
        # opposite conventions.
        if out.shape[1] != c:
            raise ValueError(f"channel count changed: {c} -> {out.shape[1]}")
        return io.NodeOutput(out.movedim(-3, -1))


class SeamlessResize(io.ComfyNode):
    """Resample a tile while keeping it tileable.

    Needed for the channels that must **not** go through a super-resolution
    network:

    - roughness / metallic are low-frequency and structureless, and an SR
      network's whole job is to invent structure, so it fabricates detail that
      does not match the height map and the PBR set stops being consistent;
    - a plain `ImageScale` would do, except that `F.interpolate` clamps at the
      image border, so the result stops being periodic.

    So: extend periodically, resample, crop the centre back. Cost is negligible
    because these kernels have a radius of a few pixels, which is why `pad`
    defaults to 32 rather than the 256 the SR node needs.

    `scale_by=0.5` with `method="area"` is also the right way to feed a 1024
    tile into a model that works at 512 -- for an exact factor of two, "area" is
    exactly the 2x2 box mean, so the periodicity is preserved rather than
    approximately preserved.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeamlessResize",
            display_name="Seamless Resize",
            category="seamless",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("scale_by", default=2.0, min=0.05, max=8.0, step=0.05),
                io.Combo.Input("method", options=["area", "bicubic", "bilinear",
                                                  "lanczos", "nearest-exact"]),
                io.Int.Input("pad", default=32, min=0, max=256, step=8,
                             tooltip="Padding in input pixels. These kernels reach a "
                                     "few pixels, so 32 is already generous."),
                io.Boolean.Input("circular", default=True),
            ],
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    def execute(cls, image, scale_by: float = 2.0, method: str = "bicubic",
                pad: int = 32, circular: bool = True) -> io.NodeOutput:
        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise ValueError(f"expected BHWC image, got shape {tuple(image.shape)}")

        b, h, w, c = image.shape
        tensor = image.movedim(-1, -3)  # BHWC -> BCHW

        if circular and pad > 0:
            tensor = F.pad(tensor, (pad, pad, pad, pad), mode="circular")

        ph, pw = tensor.shape[-2], tensor.shape[-1]
        th, tw = int(round(ph * scale_by)), int(round(pw * scale_by))
        out = comfy.utils.common_upscale(tensor, tw, th, method, "disabled")

        if circular and pad > 0:
            oh, ow = int(round(h * scale_by)), int(round(w * scale_by))
            # Centre crop. Deriving the offset from the size difference rather
            # than from pad*scale keeps it exact for fractional factors too.
            y0 = (out.shape[-2] - oh) // 2
            x0 = (out.shape[-1] - ow) // 2
            out = out[..., y0:y0 + oh, x0:x0 + ow]

        out = torch.clamp(out, 0.0, 1.0)
        if out.shape[1] != c:
            raise ValueError(f"channel count changed: {c} -> {out.shape[1]}")
        return io.NodeOutput(out.movedim(-3, -1))


__all__ = ["SeamlessUpscale", "SeamlessResize"]
