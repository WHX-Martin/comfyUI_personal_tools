"""StableMaterials integration: tileable albedo (+ optional text) -> aligned PBR set.

StableMaterials (Vecchio et al., CVPR 2026, arXiv 2406.09293) is a latent diffusion
model that predicts five aligned material maps in one pass -- basecolor, normal,
height, roughness, metallic.  This is exactly what a deterministic derivation chain
(from ``pbr.py``) cannot do: roughness/metallic are *learned* from ~6200 real PBR
materials rather than guessed from albedo luminance.

Why this module exists instead of using the upstream ComfyUI node
-----------------------------------------------------------------
The upstream pack (``P5ina/comfyui-stable-materials``) ships a
``StableMaterialsFromImage`` node that calls::

    pipeline(prompt=prompt if prompt else None, image=input_image,
             image_guidance_scale=..., ...)

The pipeline's ``__call__`` accepts neither ``image`` nor ``image_guidance_scale``
(only ``**kwargs``), so both are silently swallowed and the image conditioning is
dropped.  With the default empty prompt that also passes ``prompt=None``, which
makes ``check_inputs`` raise.  So the upstream image path is unusable, and the
correct conditioning -- ``prompt=<PIL image>``, per the model card -- is supplied
here.

Two things are added on top of the model card's usage:

* **Text fusion.**  Text and image both project into the same 768-d CLIP space,
  so they can be concatenated along the *sequence* dimension, giving cross-attention
  two keys (one from the albedo, one from the prompt) instead of one.  This is an
  extrapolation beyond the released usage (`prompt=image` *or* `prompt="text"`),
  hence it is a switch (``fuse_text``) rather than a default behaviour that cannot
  be turned off.
* **Strict torus.**  StableMaterials' own tileability rolls feature maps inside
  every conv, but only for ``i >= len(timesteps) // 4`` -- the first quarter of the
  steps still run on a rectangular feature map with zero padding.  Forcing
  ``padding_mode="circular"`` on every UNet conv for the whole run closes that gap.
  Roll and circular padding are both period-preserving, so the two compose.

Note that classifier-free guidance is *inactive* for this model: the LCM UNet sets
``time_cond_proj_dim=256``, and ``do_classifier_free_guidance`` requires it to be
``None``.  ``guidance_scale`` instead feeds a guidance embedding as a timestep
condition, and the negative prompt is computed but never used -- see
``StableMaterialsPipeline.do_classifier_free_guidance``.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
import torch

from comfy_api.latest import io

__all__ = ["SeamlessMaterialsFromAlbedo", "circular_conv_padding"]

LOGGER = logging.getLogger("comfyui_seamless_texture")

MAPS = ("basecolor", "normal", "height", "roughness", "metallic")


@contextlib.contextmanager
def circular_conv_padding(module, enabled: bool = True):
    """Temporarily force every ``Conv2d`` in ``module`` to pad circularly.

    ``torch.nn.Conv2d._conv_forward`` only honours ``padding_mode`` when it is not
    ``"zeros"``, in which case it pads explicitly and runs the convolution with
    ``padding=0``.  ``_reversed_padding_repeated_twice`` was computed from the
    original ``padding`` at construction time and stays valid, so simply swapping
    the attribute is enough -- no re-instantiation needed.

    The previous modes are restored on exit, including on exception, so the cached
    pipeline is never left running on a torus by accident.
    """
    if not enabled:
        yield
        return

    saved: list[tuple[torch.nn.Conv2d, str]] = []
    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d) and child.padding_mode != "circular":
            saved.append((child, child.padding_mode))
            child.padding_mode = "circular"
    LOGGER.info("[seamless] strict torus: %d conv layers switched to circular padding", len(saved))
    try:
        yield
    finally:
        for child, mode in saved:
            child.padding_mode = mode


def _comfy_to_pil(image_hwc: torch.Tensor):
    """ComfyUI IMAGE slice (H, W, 3) float [0,1] -> PIL RGB."""
    from PIL import Image

    arr = image_hwc[..., :3].detach().float().cpu().numpy()
    arr = np.clip(arr * 255.0, 0.0, 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _pil_to_comfy(image) -> torch.Tensor:
    """PIL image -> ComfyUI IMAGE tensor (1, H, W, 3) float [0,1].

    ``gray`` maps come back from StableMaterials in mode ``"L"``; broadcasting to
    three channels is what every consumer downstream expects.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    tensor = torch.from_numpy(arr[..., :3].astype(np.float32) / 255.0)
    return tensor.unsqueeze(0)


def _fuse_embeddings(pipeline, pil_image, text: str) -> torch.Tensor:
    """Concatenate the albedo embedding and the prompt embedding along the sequence axis.

    Both encoders are CLIP models projecting into the same 768-d space that the
    UNet cross-attends to, so the two pooled vectors can sit side by side as two
    keys.  Sequence concatenation (dim=1) is used rather than batch concatenation
    (dim=0) precisely because the latter would run two *independent* generations
    instead of one conditioned on both modalities.
    """
    device = pipeline.device

    with torch.inference_mode():
        image_inputs = pipeline.processor(images=pil_image, return_tensors="pt")
        image_inputs["pixel_values"] = image_inputs["pixel_values"].to(device)
        image_embed = pipeline.vision_encoder(**image_inputs).image_embeds.unsqueeze(1)

        text_inputs = pipeline.tokenizer(text, padding=True, return_tensors="pt")
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        text_embed = pipeline.text_encoder(**text_inputs).text_embeds.unsqueeze(1)

    embed = torch.cat([image_embed, text_embed], dim=1)
    LOGGER.info("[seamless] conditioning: albedo %s + text %s -> %s",
                tuple(image_embed.shape[1:]), tuple(text_embed.shape[1:]), tuple(embed.shape[1:]))
    return embed


class SeamlessMaterialsFromAlbedo(io.ComfyNode):
    """Albedo -> five aligned PBR maps, via StableMaterials."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeamlessMaterialsFromAlbedo",
            display_name="StableMaterials: Albedo -> PBR (seamless)",
            category="Seamless Texture/materials",
            description=(
                "Run StableMaterials on a tileable albedo to get basecolor / normal / "
                "height / roughness / metallic in one pass. Tileability is preserved "
                "twice over: by the model's own feature rolling, and by forcing every "
                "UNet convolution to pad circularly for the whole run. Supply an "
                "already-resampled albedo whose size matches `size` (use Seamless "
                "Resize) -- this node does not resample, because a naive resize would "
                "break the tile's periodicity."
            ),
            inputs=[
                io.Custom("SM_PIPELINE").Input("pipeline"),
                io.Image.Input("image"),
                io.String.Input(
                    "prompt", default="", multiline=True, optional=True,
                    tooltip="Optional. Empty means albedo-only conditioning, which is the "
                            "usage the model card documents. Non-empty fuses the prompt "
                            "with the albedo along the sequence axis (see fuse_text).",
                ),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True),
                io.Int.Input("steps", default=4, min=1, max=50,
                             tooltip="The published LCM UNet is distilled for 4 steps."),
                io.Float.Input("guidance_scale", default=10.0, min=1.0, max=20.0, step=0.5,
                               tooltip="Feeds a guidance timestep embedding, not CFG -- this "
                                       "model has no active classifier-free guidance. The "
                                       "model card uses 10.0."),
                io.Int.Input("size", default=512, min=256, max=1024, step=64,
                             tooltip="Must equal the input's side length. The base model is "
                                     "trained at 512; the refiner that would allow more has "
                                     "not been released."),
                io.Boolean.Input("tileable", default=True,
                                 tooltip="Use the model's own feature rolling."),
                io.Boolean.Input("strict_torus", default=True,
                                 tooltip="Additionally force circular padding on all UNet "
                                         "convolutions for every step. The model's own "
                                         "tileability skips the first quarter of the steps."),
                io.Boolean.Input("fuse_text", default=True,
                                 tooltip="Only applies when prompt is non-empty."),
            ],
            outputs=[
                io.Image.Output(display_name="basecolor"),
                io.Image.Output(display_name="normal"),
                io.Image.Output(display_name="height"),
                io.Image.Output(display_name="roughness"),
                io.Image.Output(display_name="metallic"),
            ],
        )

    @classmethod
    def execute(cls, pipeline, image, prompt: str = "", seed: int = 0, steps: int = 4,
                guidance_scale: float = 10.0, size: int = 512, tileable: bool = True,
                strict_torus: bool = True, fuse_text: bool = True) -> io.NodeOutput:
        if image.ndim != 4:
            raise ValueError(f"expected an IMAGE tensor of shape (B, H, W, C), got {tuple(image.shape)}")
        batch, height, width = image.shape[0], image.shape[1], image.shape[2]
        if batch != 1:
            raise ValueError(
                f"this node generates one material per call, got a batch of {batch}. "
                "Split the batch or call it once per image."
            )
        if height != size or width != size:
            raise ValueError(
                f"input albedo is {width}x{height} but size={size}. Insert a Seamless "
                "Resize node, or set size to match -- resampling here would break the "
                "periodicity that makes the tile seamless."
            )

        pil_image = _comfy_to_pil(image[0])
        text = (prompt or "").strip()
        generator = torch.Generator(device=pipeline.device).manual_seed(int(seed))

        common = dict(
            height=size,
            width=size,
            tileable=bool(tileable),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )

        LOGGER.info("[seamless] StableMaterials: %dx%d, %d steps, guidance %.1f, tileable=%s, torus=%s, text=%r",
                    size, size, steps, guidance_scale, tileable, strict_torus, text)

        with torch.inference_mode():
            with circular_conv_padding(pipeline.unet, enabled=bool(strict_torus)):
                if text and fuse_text:
                    result = pipeline(prompt_embeds=_fuse_embeddings(pipeline, pil_image, text), **common)
                else:
                    # The documented image-conditioning entry point: the image *is* the prompt.
                    result = pipeline(prompt=pil_image, **common)

        material = result.images[0]
        maps = [_pil_to_comfy(getattr(material, name)) for name in MAPS]
        LOGGER.info("[seamless] StableMaterials produced %s maps of %s",
                    len(maps), tuple(maps[0].shape))
        return io.NodeOutput(*maps)
