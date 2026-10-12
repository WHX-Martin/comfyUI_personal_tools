"""comfyui_seamless_texture -- make diffusion output tile by *padding*, not by blending.

Why this exists
---------------
A texture is only usable if it is 四方连续 (seamless in both axes). ComfyUI used to
ship ``CircularVAEDecode`` for this; it is gone from core, so nothing in a stock
0.38.2 install can produce mathematically tileable output.

The mechanism here is the one A1111 calls ``--tiling``: switch every ``Conv2d``
in the network to ``padding_mode="circular"``. That is not a post-process and not
a blend -- it changes the *topology the convolution sees*. With zero padding the
feature map is a rectangle with an artificial border, and the model paints a
visible border layer at the edges. With circular padding the feature map becomes
a **torus**: column 0's left neighbour is column W-1, and there is no border to
paint. Every downstream op in the pipeline (pointwise samplers, nearest-neighbour
upsampling, attention) commutes with that periodicity, so the finished latent is
a function *on the torus* and the decoded image tiles exactly -- repeating it 2x2
has no seam at all, not merely a faint one.

Three facts about this install were checked before relying on the above, because
each one silently breaks the scheme if wrong:

* ``comfy.ops.Conv2d.forward`` eventually calls ``torch.nn.Conv2d._conv_forward``,
  which is the path that honours ``padding_mode``. (A subclass that called
  ``F.conv2d`` directly would ignore it.)
* The SDXL UNet (``comfy/ldm/modules/diffusionmodules/openaimodel.py``) contains
  no hand-rolled ``F.pad``; its downsampler is ``conv 3x3 / stride 2 / pad 1``,
  which wraps correctly.
* The SDXL VAE *decoder* is nearest-upsample + ``conv 3x3 / pad 1``, i.e. clean.
  Only the *encoder's* ``Downsample`` hardcodes an asymmetric zero pad
  (``F.pad(x, (0, 1, 0, 1))``), so ``SeamlessVAEEncode`` additionally pads the
  image circularly and crops the contaminated margin back off.

Keeping ComfyUI's model cache clean
-----------------------------------
Mutating a cached model's convs in place would leak: ComfyUI keeps the patcher
alive between runs, so a later, unrelated workflow would silently keep generating
on a torus. So the UNet patch is not a mutation at all -- it is installed as a
``model_function_wrapper`` on a *clone* of the patcher and applied for the
duration of each UNet call only, then reverted. Same idea for the VAE nodes: a
scoped context manager, restored on exit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

__all__ = [
    "SeamlessModelPadding",
    "SeamlessVAEDecode",
    "SeamlessVAEEncode",
    "TilePreview",
    "circular_padding",
]

# Spatial compression of the SD / SDXL VAE. `margin` is expressed in latent cells
# so it stays meaningful if someone reuses these nodes with another latent_dim=2
# VAE, but the pixel-space crop has to know the ratio.
VAE_SCALE = 8


def _conv2d_list(root) -> list[torch.nn.Conv2d]:
    return [m for m in root.modules() if isinstance(m, torch.nn.Conv2d)]


class circular_padding:
    """Switch every ``Conv2d`` under ``root`` to circular padding, then restore.

    Used as a context manager. The restore is what makes this safe to call on a
    model ComfyUI has cached: on exit every conv is put back exactly as found.

    Convs with ``padding == 0`` are skipped rather than patched. They are
    unaffected by ``padding_mode`` either way, and the SDXL UNet holds well over
    a thousand convs, so the common case stays cheap. (``padding`` is a 2-tuple
    on ``nn.Conv2d``, hence the ``all()``.)
    """

    def __init__(self, root):
        self._root = root
        self._saved: list[tuple[torch.nn.Conv2d, str]] = []

    def __enter__(self):
        if self._root is None:
            return self
        saved = []
        for conv in _conv2d_list(self._root):
            if all(p == 0 for p in conv.padding):
                continue
            saved.append((conv, conv.padding_mode))
            conv.padding_mode = "circular"
        self._saved = saved
        return self

    def __exit__(self, *exc_info):
        for conv, padding_mode in self._saved:
            conv.padding_mode = padding_mode
        self._saved = []
        return False


def _diffusion_root(model):
    """Reach the ``nn.Module`` that actually runs, from a ModelPatcher."""
    inner = getattr(model, "model", None)
    return getattr(inner, "diffusion_model", None) or inner


def _unwrap_latent(samples):
    latent = samples["samples"]
    if getattr(latent, "is_nested", False):
        latent = latent.unbind()[0]
    return latent


def _flatten_batch(images):
    # Mirrors core VAEDecode: a video VAE can hand back B,T,C,H,W.
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _pad_circular(tensor, pad):
    """Circular-pad the last two dims. ``F.pad`` pads trailing dims, so a BHWC
    image has to be moved to BCHW first -- easy to get wrong silently."""
    return F.pad(tensor, (pad, pad, pad, pad), mode="circular")


class SeamlessModelPadding(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessModelPadding",
            display_name="Seamless Model Padding (Circular)",
            description=(
                "Run this model on a torus: every convolution wraps around instead of "
                "seeing zero padding. This is the step that makes the generated texture "
                "tile seamlessly -- it is a change to the sampling padding, not a "
                "post-process blend. Place it after the LoRA loader and before the "
                "sampler. The patch is scoped to the UNet call and reverted afterwards, "
                "so the checkpoint ComfyUI has cached is left untouched."
            ),
            category="seamless",
            inputs=[
                io.Model.Input("model", tooltip="The model to run with circular padding."),
                io.Combo.Input(
                    "mode",
                    options=["circular", "off"],
                    default="circular",
                    tooltip=(
                        "'circular' wraps the convolution (seamless output). "
                        "'off' passes the model through, useful for A/B comparing a seam."
                    ),
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, mode: str) -> io.NodeOutput:
        if mode != "circular":
            return io.NodeOutput(model)

        model = model.clone()
        diffusion_model = _diffusion_root(model)

        def unet_wrapper(apply_model, args):
            with circular_padding(diffusion_model):
                return apply_model(args["input"], args["timestep"], **args["c"])

        model.set_model_unet_function_wrapper(unet_wrapper)
        return io.NodeOutput(model)


class SeamlessVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessVAEDecode",
            display_name="Seamless VAE Decode (Circular Padding)",
            description=(
                "Decode with circular padding in the VAE decoder, then discard a margin. "
                "Pairs with Seamless Model Padding; without it the UNet still paints an "
                "edge layer and a faint border survives. The margin exists so the few "
                "rows the decoder cannot resolve toroidally (latent edges) are cropped "
                "off instead of shipped inside the tile."
            ),
            category="seamless",
            inputs=[
                io.Latent.Input("samples", tooltip="The latent to decode."),
                io.Vae.Input("vae", tooltip="The VAE to decode with."),
                io.Int.Input(
                    "margin",
                    default=8,
                    min=0,
                    max=64,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Latent cells of circular padding added before decoding and cropped "
                        "back off afterwards (8 cells = 64 px at 1024). Larger margins cost "
                        "decode time and buy nothing once above the decoder's receptive "
                        "field. Set 0 to decode at the exact latent size."
                    ),
                ),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, samples, vae, margin: int) -> io.NodeOutput:
        latent = _unwrap_latent(samples)
        pad_px = margin * VAE_SCALE

        if margin > 0:
            if min(latent.shape[-2:]) <= margin:
                raise ValueError(
                    f"margin={margin} does not fit a {tuple(latent.shape[-2:])} latent; "
                    "lower margin."
                )
            latent = _pad_circular(latent, margin)

        with circular_padding(vae.first_stage_model):
            images = vae.decode(latent)

        # vae.decode hands back **BHWC** -- it finishes with `.movedim(1, -1)`
        # (comfy/sd.py). Slicing that as if it were BCHW does not crop the spatial
        # margin at all; it eats the channel axis instead, which is how the first
        # version of this node handed SaveImage a `(1, H, W, 0)` tensor. So assert
        # the layout rather than trusting the comment.
        images = _flatten_batch(images)
        if margin > 0:
            if images.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    "decoded image does not end in a channel axis: got "
                    f"{tuple(images.shape)}. ComfyUI IMAGE tensors are BHWC."
                )
            height, width = images.shape[1], images.shape[2]
            images = images[:, pad_px:height - pad_px, pad_px:width - pad_px, :]
        return io.NodeOutput(images)


class SeamlessVAEEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessVAEEncode",
            display_name="Seamless VAE Encode (Circular Padding)",
            description=(
                "Encode with circular padding, for img2img / hires-fix / repaint passes "
                "that must stay tileable. Needed because the VAE encoder's downsampler "
                "hardcodes an asymmetric zero pad, which is not toroidal -- the circular "
                "margin hides that, and the margin is cropped off the latent afterwards."
            ),
            category="seamless",
            inputs=[
                io.Image.Input("pixels", tooltip="The image to encode."),
                io.Vae.Input("vae", tooltip="The VAE to encode with."),
                io.Int.Input(
                    "margin",
                    default=8,
                    min=0,
                    max=64,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Latent cells of circular padding around the image, cropped back "
                        "off the latent afterwards (8 cells = 64 px at 1024)."
                    ),
                ),
            ],
            outputs=[io.Latent.Output(display_name="LATENT")],
        )

    @classmethod
    def execute(cls, pixels, vae, margin: int) -> io.NodeOutput:
        pad_px = margin * VAE_SCALE

        if margin > 0:
            if min(pixels.shape[1:3]) <= pad_px:
                raise ValueError(
                    f"margin={margin} ({pad_px} px) does not fit a "
                    f"{tuple(pixels.shape[1:3])} image; lower margin."
                )
            pixels = _pad_circular(pixels.permute(0, 3, 1, 2), pad_px).permute(0, 2, 3, 1)

        with circular_padding(vae.first_stage_model):
            latent = vae.encode(pixels)

        # Unlike the decode direction, vae.encode takes BHWC and gives back BCHW
        # (comfy/sd.py `movedim(-1, 1)` on the way in), so dims 2/3 really are the
        # spatial ones here. This crop sits one node away from the decode crop,
        # which had it backwards -- hence the cheap layout guard.
        if latent.shape[1] > 32:
            raise ValueError(f"encoded latent looks transposed: {tuple(latent.shape)}")
        if margin > 0:
            latent = latent[:, :, margin:-margin, margin:-margin]
        return io.NodeOutput({"samples": latent})


class TilePreview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TilePreview",
            display_name="Tile Preview (2x2 check)",
            description=(
                "Repeat the image in a grid and save that instead of the single tile. "
                "A seamless texture is indistinguishable from its own repetition, so any "
                "seam shows up as a visible cross here while remaining invisible in the "
                "single tile. Wire it to a second SaveImage to eyeball every run."
            ),
            category="seamless",
            inputs=[
                io.Image.Input("image", tooltip="The tile to repeat."),
                io.Int.Input("tiles_x", default=2, min=1, max=8, step=1),
                io.Int.Input("tiles_y", default=2, min=1, max=8, step=1),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, image, tiles_x: int, tiles_y: int) -> io.NodeOutput:
        return io.NodeOutput(image.repeat(1, tiles_y, tiles_x, 1))
