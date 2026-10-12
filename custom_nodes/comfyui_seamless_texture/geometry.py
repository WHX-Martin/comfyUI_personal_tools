"""Geometry-driven PBR: recover relief from the *lit* render, then de-light it.

Why this module exists
----------------------
The v2 LoRA was trained on lit renders, not on flat albedo. Its captions all carry
`directional sunlight from upper left`, `raking light` and `visible bump relief with
soft cast shadows`, so what it paints is a *shaded surface*: the luminance already
contains the relief. That changes the problem from "invent the other channels" to
"invert the shading", and MoGe-2 (``moge_2_vitl_normal_fp16``) has a normal head
that recovers that relief well enough to be worth using.

What was measured before writing this
-------------------------------------
Four things were checked against the live server rather than assumed. They are
recorded here because each one changed the design:

1. **MoGe does recover real relief from these renders.** On 1024x1024 v2 renders
   the normal head returns per-grain perturbations -- mean slope 0.07-0.23, mean
   tilt 3.8-13.9 deg -- that visibly follow pebbles and ripple crests. It is not
   the near-planar sheet you get from feeding a *flat* albedo into a scene-depth
   network, which is what an earlier evaluation of this idea concluded.

2. **But the normals are a prior, not an inversion.** Fitting a single Lambertian
   light to the render and the recovered normals explains only R^2 = 0.10-0.72 of
   the luminance variance (0.01-0.47 if the fit is restricted to the high-pass
   band where shading actually lives). The fitted light direction disagrees with
   the caption's "upper left raking" in most cases. So the normals are
   spatially plausible and *partially* photometric, and must not be treated as
   ground truth. Two consequences are baked into the workflow: the de-lighting
   step is a robust retinex, not a division by ``n.L``; and the checkpoint with
   the stronger relief (ep10) is the one whose normals fit the shading best.

3. **MoGe breaks tiling.** Fed a provably periodic render, the returned depth and
   normal came back with a wrap discontinuity 4x-629x the median interior
   difference -- a hard seam, because a feed-forward net pads at the image border.
   Hence ``SeamlessMoGeInference``: extend the tile periodically, run the net,
   crop the centre back out. Same argument and same fix as ``SeamlessUpscale``.

4. **The depth head is much weaker than the normal head.** Its high-frequency
   content is 4-12% of the render's, against 60%+ for the normals. So height is
   *integrated from the normal field* rather than taken from the depth map.

The one place MoGe's resolution matters
---------------------------------------
``MoGeModelV2.forward`` builds a ``base_h x base_w`` patch grid from
``num_tokens`` (level 9 => 3600 tokens => a 60x60 grid, i.e. ~840 px of internal
resolution) and then bilinearly resizes the result to the input size. Padding a
1024 tile to 1536 with the *same* token count would therefore spread those 840
internal pixels over a wider field and the centre crop would come back softer
than the unpadded run. ``SeamlessMoGeInference`` scales ``num_tokens`` with the
padded area so the padding costs compute and nothing else.

Note while wiring this up: core's ``MoGeInference`` declares a ``refine_steps``
widget but never passes it to ``MoGeModel.infer``, whose signature has no such
parameter. It is inert there. This node does not offer it.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from comfy_api.latest import io

from .pbr import _circular_blur, _gray3, _luminance, _monotone_norm, _percentile_norm

__all__ = [
    "SeamlessMoGeInference",
    "SeamlessDSINENormal",
    "SeamlessFlattenNormal",
    "SeamlessHeightFromNormal",
    "SeamlessDelightAlbedo",
    "SeamlessRoughnessFromHeight",
]

MoGeModelType = io.Custom("MOGE_MODEL")
MoGeGeometry = io.Custom("MOGE_GEOMETRY")


def _decode_normal(normal_map):
    """Encoded normal IMAGE (BHWC, [0,1]) -> unit vectors in the map's own space.

    ``MoGeRender`` already applied the OpenGL/DirectX convention change, so the
    decode is convention-agnostic: whatever it wrote, the vector that reads as
    "straight out of the surface" is the one that comes back as (0, 0, 1).
    """
    n = (normal_map[..., :3] - 0.5) * 2.0
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _encode_normal(n):
    return (n * 0.5 + 0.5).clamp(0.0, 1.0)


def _crop(t, y0, x0, h, w):
    """Centre crop for the layouts MoGe returns: depth/mask are BHW, the rest BHWC."""
    if t is None:
        return None
    if t.ndim == 4:
        return t[..., y0:y0 + h, x0:x0 + w, :]
    return t[..., y0:y0 + h, x0:x0 + w]


class SeamlessMoGeInference(io.ComfyNode):
    """MoGe-2 with a periodic boundary condition, so the maps still tile.

    Extends the tile circularly by ``pad`` pixels, runs MoGe, crops the centre
    ``pad``-extension back off. The network then never sees an edge, and the crop
    is periodic by construction -- the same argument, and the same measurement,
    as ``SeamlessUpscale``. Set ``circular=false`` for a same-seed A/B: the
    difference between the two is the seam, and it is not subtle (measured 4x-629x
    the interior difference on the wrap row).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessMoGeInference",
            display_name="Seamless MoGe Inference",
            description=(
                "Run MoGe-2 on a tileable image while keeping the result tileable, by "
                "extending the tile periodically before inference and centre-cropping "
                "afterwards. Output is a MOGE_GEOMETRY packet, so the stock MoGe Render "
                "and Point Map to Mesh nodes work downstream."
            ),
            category="seamless/geometry",
            inputs=[
                MoGeModelType.Input("moge_model"),
                io.Image.Input("image", tooltip="A tileable (四方连续) render."),
                io.Int.Input(
                    "pad", default=256, min=0, max=1024, step=32,
                    tooltip="How far to extend the tile periodically. MoGe-2 attends "
                            "globally, so this converges more slowly than a local conv net "
                            "does; check by raising it and re-running. Costs "
                            "(size + 2*pad)^2, and token count is scaled to match so the "
                            "crop does not come back softer than the unpadded run.",
                ),
                io.Boolean.Input(
                    "circular", default=True,
                    tooltip="Off = plain inference, for an A/B. The wrap seam it leaves "
                            "is the reason this node exists.",
                ),
                io.Int.Input(
                    "resolution_level", default=9, min=0, max=9,
                    tooltip="Token budget, 0 = fastest, 9 = most detail. Level 9 is 3600 "
                            "tokens, about 840 px of internal resolution.",
                ),
                io.Int.Input(
                    "num_tokens", default=0, min=0, max=32768, step=64,
                    tooltip="0 = derive from resolution_level and scale with the padded "
                            "area (the recommended path). Any other value overrides both "
                            "resolution_level and the scaling.",
                ),
                io.Float.Input("fov_x_degrees", default=0.0, min=0.0, max=170.0, step=0.1,
                               tooltip="0 = recover the focal length from the predicted "
                                       "point map. Only affects the point map; the normal "
                                       "head ignores intrinsics."),
                io.Int.Input("batch_size", default=1, min=1, max=64),
                io.Boolean.Input("force_projection", default=True),
                io.Boolean.Input(
                    "apply_mask", default=False,
                    tooltip="Off is the safer default for a full-frame texture: MoGe's "
                            "mask is trained on sky/background and can punch holes in a "
                            "flat material.",
                ),
            ],
            outputs=[MoGeGeometry.Output(display_name="moge_geometry")],
        )

    @classmethod
    def execute(cls, moge_model, image, pad: int = 256, circular: bool = True,
                resolution_level: int = 9, num_tokens: int = 0,
                fov_x_degrees: float = 0.0, batch_size: int = 1,
                force_projection: bool = True, apply_mask: bool = False) -> io.NodeOutput:
        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise ValueError(f"expected BHWC image, got shape {tuple(image.shape)}")

        image = image[..., :3]
        b, h, w, _ = image.shape
        bchw = image.movedim(-1, -3).contiguous()

        if circular and pad > 0:
            bchw = F.pad(bchw, (pad, pad, pad, pad), mode="circular")
        _, _, ph, pw = bchw.shape

        if num_tokens > 0:
            tokens = int(num_tokens)
        else:
            lo, hi = moge_model.num_tokens_range
            base = int(lo + (resolution_level / 9) * (hi - lo))
            tokens = max(64, int(round(base * (ph * pw) / (h * w))))

        fov = None if fov_x_degrees <= 0 else float(fov_x_degrees)
        chunks = []
        for i in range(0, b, batch_size):
            chunks.append(moge_model.infer(
                bchw[i:i + batch_size], num_tokens=tokens, fov_x=fov,
                force_projection=force_projection, apply_mask=apply_mask))

        def stack(field):
            vals = [c[field] for c in chunks if field in c]
            return torch.cat(vals, dim=0) if vals else None

        y0 = (ph - h) // 2
        x0 = (pw - w) // 2

        geometry = {}
        for field in ("points", "depth", "normal", "mask"):
            geometry[field] = _crop(stack(field), y0, x0, h, w)

        # The crop moves the principal point, so the intrinsics have to follow or
        # MoGePointMapToMesh would unproject against the wrong centre.
        intrinsics = stack("intrinsics")
        if intrinsics is not None:
            intrinsics = intrinsics.clone()
            intrinsics[..., 0, 2] -= x0
            intrinsics[..., 1, 2] -= y0
            geometry["intrinsics"] = intrinsics

        geometry["image"] = image.cpu()
        return io.NodeOutput(geometry)


# ---------------------------------------------------------------------------
# DSINE -- the convolutional route to a *seamless* normal, with a smaller FOV
#
# MoGe is a ViT: global attention plus absolute positional encoding couple every
# output pixel to the whole image, so extending the tile only pushes the wrap
# discontinuity from 4.24x to a 1.62x plateau and never to 1.0. There is no
# converged interior to crop. DSINE (CVPR 2024) is the opposite case: timm
# EfficientNet-B5 encoder + conv decoder + ConvGRU, no attention and no learned
# positional embedding, so the influence of the padded border genuinely decays
# (measured 0.0280 -> 0.0048 going pad 0 -> 768).
#
# But "convolutional" alone was not enough, and the reason is worth recording
# because it is the whole trick of this node. DSINE consumes a per-pixel RAY
# DIRECTION, which is an absolute positional field and therefore not periodic.
# At the default fov=60 the padding made the wrap ratio *worse* (6.87 -> 5.45);
# at fov<=10 it collapses (fov=3, pad=256 gives wrap excess 1.11 with z=0.50, i.e.
# statistically indistinguishable from an ordinary interior neighbouring
# difference). A texture has no camera, so fov->0 is not a hack: it is the
# orthographic observation a flat material actually is. Relief amplitude does not
# suffer; mean tilt rose from 12.9 to 15.1 deg.
#
# Convention, measured rather than assumed. DSINE writes (n+1)/2 for normals in
# its camera frame (right, down, front). This project's maps are OpenGL-flavoured:
# `_normal_from_height` builds `normalize(-dh/dx, +dh/dy_down, 1)`, so the map's Y
# axis points UP while the image's rows run DOWN. Four independent measurements
# agreed that the conversion is **flip the RED channel only**:
#   * cross-correlation against MoGe on a real gravel render -> signs (-1, +1, +1);
#   * a synthetic render built from a known height field, 2-D light -> (-1,-1,+1)
#     in image coordinates, which is the same statement;
#   * a single-axis analytic test, which also reproduced the red flip;
#   * a control run of MoGe on that same synthetic render, proving the render was
#     not mirrored (MoGe needed (1,-1,1), exactly as its known convention implies).
# Note the asymmetry is real: green comes out already correct, red does not.
# Two lighting-based tests were tried first and had no power at all -- the
# high-pass R^2 of a single-Lambertian fit on a gravel render is 0.023-0.026,
# i.e. this render's high-frequency luminance is ~2.5% shading and ~97.5% albedo.
# ---------------------------------------------------------------------------

DSINE_REPO = "hr16/Diffusion-Edge"
DSINE_FILENAME = "dsine.pt"
DSINE_BYTES = 291298581          # for truncation checks; mirrors have served 29 MB
_DSINE_ALIGN = 32                # DSINE reads H//32, W//32 and upsamples from 1/8

_DSINE_CACHE: dict = {}


def _custom_node_roots():
    roots = []
    try:
        import folder_paths
        for p in folder_paths.get_folder_paths("custom_nodes"):
            roots.append(Path(p))
    except Exception:
        pass
    if not roots:
        roots.append(Path(__file__).resolve().parents[1])
    return roots


def _dsine_weight_path():
    """Locate dsine.pt without downloading anything behind the user's back."""
    cands = []
    env = os.environ.get("SEAMLESS_DSINE_WEIGHTS")
    if env:
        cands.append(Path(env))
    for root in _custom_node_roots():
        cands.append(root / "comfyui_controlnet_aux" / "ckpts" / DSINE_REPO / DSINE_FILENAME)
    try:
        import folder_paths
        for kind in ("dsine", "geometry_estimation", "checkpoints"):
            try:
                for d in folder_paths.get_folder_paths(kind):
                    cands.append(Path(d) / DSINE_FILENAME)
            except Exception:
                continue
    except Exception:
        pass
    for c in cands:
        if c.is_file():
            return c
    return None


def _load_dsine(device):
    """Load DSINE, caching it. The stock preprocessor node rebuilds the model on
    every call, which costs seconds per run for no reason."""
    path = _dsine_weight_path()
    if path is None:
        raise RuntimeError(
            "SeamlessDSINENormal needs the DSINE checkpoint, which was not found.\n"
            "Expected it at:\n"
            f"    <custom_nodes>/comfyui_controlnet_aux/ckpts/{DSINE_REPO}/{DSINE_FILENAME}\n"
            "Fetch it with (hf-mirror + curl, because this machine's Python does not "
            "trust the local proxy's CA):\n"
            f'    curl -skL -m 900 "https://hf-mirror.com/{DSINE_REPO}/resolve/main/'
            f'{DSINE_FILENAME}" -o "<that path>"\n'
            f"It should be {DSINE_BYTES} bytes. Or point SEAMLESS_DSINE_WEIGHTS at it."
        )

    key = (str(path), str(device))
    if key in _DSINE_CACHE:
        return _DSINE_CACHE[key]

    src = None
    for root in _custom_node_roots():
        cand = root / "comfyui_controlnet_aux" / "src"
        if (cand / "custom_controlnet_aux" / "dsine").is_dir():
            src = cand
            break
    if src is None:
        raise RuntimeError(
            "SeamlessDSINENormal reuses the DSINE architecture that ships inside the "
            "comfyui_controlnet_aux node pack, and that pack was not found. Install it, "
            "or point SEAMLESS_DSINE_WEIGHTS at a checkpoint and install the pack anyway."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    try:
        from custom_controlnet_aux.dsine import load_checkpoint
        from custom_controlnet_aux.dsine.models.dsine_arch import DSINE
    except Exception as exc:                       # pragma: no cover - env dependent
        raise RuntimeError(
            f"could not import the DSINE architecture from {src}: {exc}"
        ) from exc

    model = DSINE()
    model = load_checkpoint(str(path), model)
    model.eval().to(device)
    model.pixel_coords = model.pixel_coords.to(device)   # plain attr: .to() skips it
    _DSINE_CACHE[key] = model
    return model


def _grow_pixel_grid(model, size: int, device) -> None:
    """Keep DSINE's absolute pixel-coordinate table big enough and on `device`.

    DSINE ships a 2000x2000 coordinate table. It is pure geometry, not a weight, so
    rebuilding it larger is safe -- and it is the only thing that stops a pad beyond
    about 488 px. Note it is stored as a plain attribute, NOT a registered buffer, so
    ``Module.to()`` does not move it; that is why the stock controlnet_aux detector
    moves it by hand, and forgetting it here is exactly the "two devices" RuntimeError
    that this guard exists to prevent.
    """
    have = model.pixel_coords.shape[-1]
    if have >= size and str(model.pixel_coords.device) == str(device):
        return
    if have >= size:
        model.pixel_coords = model.pixel_coords.to(device)
        return
    n = max(size, have * 2)
    coords = torch.zeros((3, n, n), dtype=torch.float32)
    coords[0] = torch.arange(n, dtype=torch.float32).view(1, n) + 0.5
    coords[1] = torch.arange(n, dtype=torch.float32).view(n, 1) + 0.5
    coords[2] = 1.0
    model.pixel_coords = coords.unsqueeze(0).to(device)


class SeamlessDSINENormal(io.ComfyNode):
    """DSINE surface normals on a torus -- the convolutional answer to MoGe's seam.

    Extends the tile periodically, runs DSINE, crops the centre back out. Unlike the
    ViT case this actually converges, but only once the fov is small: DSINE's
    per-pixel ray direction is an absolute positional prior, and at the default 60
    degrees it is that prior -- not the image border -- that puts a step at the wrap.
    A flat material has no camera, so 3 degrees (near-orthographic) is both the
    physically right choice and the one that measures seamless.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessDSINENormal",
            display_name="Seamless DSINE Normal",
            description=(
                "Convolutional (EfficientNet-B5 + ConvGRU) normal estimation that stays "
                "tileable: periodic padding, centre crop, and a near-orthographic FOV so "
                "the ray-direction prior does not break periodicity. Output is an encoded "
                "normal map in this pack's convention, so Flatten Normal / Height From "
                "Normal / AO / Roughness all take it directly."
            ),
            category="seamless/geometry",
            inputs=[
                io.Image.Input("image", tooltip="A tileable (四方连续) lit render."),
                io.Float.Input(
                    "fov", default=3.0, min=0.5, max=120.0, step=0.5,
                    tooltip="Field of view assumed for the ray-direction prior, in degrees. "
                            "This is the main quality knob: at 60 (DSINE's default) the "
                            "padding cannot fix the wrap at all; at 3 the wrap excess is "
                            "1.11 with z=0.50. Keep it small for flat materials.",
                ),
                io.Int.Input(
                    "pad", default=256, min=0, max=1024, step=32,
                    tooltip="How far to extend the tile periodically. Unlike MoGe this "
                            "converges: the border influence decays 0.0280 -> 0.0048 from "
                            "pad 0 to 768. 256 already reaches the floor.",
                ),
                io.Int.Input(
                    "iterations", default=5, min=1, max=20,
                    tooltip="ConvGRU refinement passes. Measured to be nearly inert: "
                            "3 -> 20 moved the wrap excess only 1.12 -> 1.14.",
                ),
                io.Boolean.Input(
                    "flip_red", default=True,
                    tooltip="DSINE's camera frame has Y down; this pack's maps have Y up, "
                            "and measurement says the net conversion is a RED-channel flip "
                            "only. Turn off to see the difference.",
                ),
                io.Boolean.Input(
                    "use_fp16", default=False,
                    tooltip="Off keeps the fp32 behaviour the seam numbers were measured "
                            "with. On halves memory for large pads.",
                ),
            ],
            outputs=[io.Image.Output(display_name="normal")],
        )

    @classmethod
    def execute(cls, image, fov: float = 3.0, pad: int = 256, iterations: int = 5,
                flip_red: bool = True, use_fp16: bool = False) -> io.NodeOutput:
        import comfy.model_management as model_management

        if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
            raise ValueError(f"expected BHWC image, got shape {tuple(image.shape)}")

        image = image[..., :3]
        b, h, w, _ = image.shape

        # DSINE reads H//32 and W//32 and upsamples from 1/8, so keep the padded
        # size a multiple of 32; grow the requested pad if needed.
        p = int(pad)
        while p < 1024 and ((h + 2 * p) % _DSINE_ALIGN or (w + 2 * p) % _DSINE_ALIGN):
            p += _DSINE_ALIGN
        ph, pw = h + 2 * p, w + 2 * p
        y0, x0 = (ph - h) // 2, (pw - w) // 2

        device = model_management.get_torch_device()
        model = _load_dsine(device)
        model.num_iter = int(iterations)
        _grow_pixel_grid(model, max(ph, pw), device)

        bchw = image.movedim(-1, -3).contiguous()
        if p > 0:
            bchw = F.pad(bchw, (p, p, p, p), mode="circular")
        bchw = bchw.to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        bchw = (bchw - mean) / std

        # Intrinsics for the *crop*, with the principal point moved to the crop's
        # centre inside the padded frame. DSINE's forward() adds +0.5 to (cx, cy),
        # and padding rescales nothing, so the focal length in pixels is unchanged
        # and the ray field inside the crop is the same for every pad.
        f = (max(h, w) / 2.0) / math.tan(math.radians(float(fov)) / 2.0)
        intrins = torch.zeros((b, 3, 3), device=device)
        intrins[:, 0, 0] = f
        intrins[:, 1, 1] = f
        intrins[:, 0, 2] = x0 + w / 2.0 - 0.5
        intrins[:, 1, 2] = y0 + h / 2.0 - 0.5
        intrins[:, 2, 2] = 1.0

        with torch.no_grad():
            if use_fp16 and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    preds = model(bchw, intrins)
            else:
                preds = model(bchw, intrins)
        normal = preds[-1].float()[..., y0:y0 + h, x0:x0 + w]     # BCHW

        enc = (normal.permute(0, 2, 3, 1) * 0.5 + 0.5)
        if flip_red:
            enc = enc.clone()
            enc[..., 0] = 1.0 - enc[..., 0]
        return io.NodeOutput(enc.clamp(0.0, 1.0))


class SeamlessFlattenNormal(io.ComfyNode):
    """Level a normal map's base plane and rescale its relief.

    MoGe interprets a close-up material as a slightly tilted *surface in a scene*
    and returns roughly the right grain sitting on a plane tilted a few degrees
    off the image plane. A texture's base plane has to be exactly the image
    plane, so the mean direction is rotated onto (0, 0, 1) with a Rodrigues
    rotation -- a rotation rather than a per-channel subtraction, because the
    latter does not preserve unit length.

    Rotation and scaling are per-pixel and global respectively, so a tileable
    normal map stays tileable.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessFlattenNormal",
            display_name="Seamless Flatten Normal",
            description=(
                "Rotate a normal map's average direction onto straight-out-of-surface "
                "and scale its slope amplitude, so the result reads as a texture rather "
                "than as a tilted surface in a scene. Reported tilt is the angle being "
                "removed."
            ),
            category="seamless/geometry",
            inputs=[
                io.Image.Input("normal", tooltip="Encoded normal map, e.g. MoGe Render / normal_opengl."),
                io.Float.Input(
                    "relief_scale", default=2.0, min=0.0, max=12.0, step=0.1,
                    tooltip="Multiplier on the tangential component. MoGe returns gentle "
                            "slopes (measured 3.8-13.9 deg mean tilt); 2-3 puts a material "
                            "back in the range a real normal map lives in. Judge on a 2x2 "
                            "tile preview.",
                ),
                io.Boolean.Input("flatten", default=True,
                                 tooltip="Off keeps MoGe's global tilt -- useful to see how "
                                         "much of the map was tilt."),
            ],
            outputs=[io.Image.Output(display_name="normal")],
        )

    @classmethod
    def execute(cls, normal, relief_scale: float = 2.0, flatten: bool = True) -> io.NodeOutput:
        n = _decode_normal(normal)
        if flatten:
            mean = n.reshape(-1, 3).mean(dim=0)
            mean = mean / mean.norm().clamp_min(1e-8)
            target = torch.zeros_like(mean)
            target[2] = 1.0
            axis = torch.cross(mean, target, dim=0)
            sin_a = axis.norm()
            cos_a = (mean * target).sum().clamp(-1.0, 1.0)
            if float(sin_a) > 1e-6:
                axis = axis / sin_a
                angle = torch.atan2(sin_a, cos_a)
                k = axis.view(1, 1, 1, 3)
                n = (n * math.cos(float(angle))
                     + torch.cross(k.expand_as(n), n, dim=-1) * math.sin(float(angle))
                     + k * (k * n).sum(-1, keepdim=True) * (1.0 - math.cos(float(angle))))
        if relief_scale != 1.0:
            xy = n[..., :2] * relief_scale
            n = torch.cat([xy, n[..., 2:3]], dim=-1)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return io.NodeOutput(_encode_normal(n))


class SeamlessHeightFromNormal(io.ComfyNode):
    """Integrate a normal field into height, exactly on the torus.

    Uses the Frankot-Chellappa formulation: with ``p = dh/dx`` and ``q = dh/dy``,
    ``lap(h) = dp/dx + dq/dy``, which in the Fourier domain is a division by
    ``kx^2 + ky^2``. Two properties make this the right integrator here:

    * it is *exactly* periodic -- the FFT *is* the circle, so no boundary term is
      introduced and a tileable normal map yields a tileable height map;
    * a constant slope is invisible to it (a uniform gradient field has zero
      divergence), so MoGe's global plane tilt is removed automatically and the
      base plane comes out flat without a separate flattening pass.

    The k = 0 mode is undefined and is pinned to zero, which fixes the arbitrary
    offset.

    The low frequencies have to go, and that is what ``low_freq_radius`` does
    ------------------------------------------------------------------------
    Integration weights the spectrum by ``1/k^2``, so ``h`` is dominated by the
    sloppiest component of the input gradient field: a slow drift with zero
    divergence, which the solve cannot remove and which behaves like a huge
    bowl-shaped offset. Left in, that component

    * fills the [0, 1] value range, so percentile normalisation squeezes the
      per-pixel gradients (measured: a normal re-derived from the raw integral
      needed ``bump_strength`` ~35-50 instead of the usual 2-6);
    * tiles into a strongly visible repeating blob -- the low-frequency bowl
      repeats at every tile boundary and reads as banding, not as material.

    Subtracting a circular blur at ``low_freq_radius`` (default 96 px) removes
    exactly that, keeps everything at grain and dune scale, and restores a
    sensible gradient scale. It stays periodic, so it costs nothing in
    tileability. Set it to 0 to see the raw integral.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessHeightFromNormal",
            display_name="Seamless Height From Normal (Poisson)",
            description=(
                "Integrate a normal map into a [0,1] height field with a periodic Poisson "
                "solve, then high-pass it to drop the low-frequency runaway that "
                "integration amplifies. Exact on the torus, so the height tiles whenever "
                "the normal does, and a constant global tilt cancels out for free. Carries "
                "more grain than MoGe's depth head (4-12% of the render's high-frequency "
                "content) because it comes from the normals."
            ),
            category="seamless/geometry",
            inputs=[
                io.Image.Input("normal", tooltip="Encoded normal map, ideally flattened first."),
                io.Int.Input(
                    "low_freq_radius", default=96, min=0, max=512, step=8,
                    tooltip="High-pass radius, in pixels. Integration amplifies low "
                            "frequencies as 1/k^2, so an integrated field is dominated by "
                            "whatever slow drift the gradient field had -- left in, it "
                            "fills the value range with a blob that repeats visibly and "
                            "squeezes the grain's gradients hard (measured: a normal "
                            "re-derived from the raw integral needed bump_strength ~35-50). "
                            "Removing content slower than this radius is what makes the "
                            "map usable as displacement. 0 disables it.",
                ),
                io.Boolean.Input(
                    "invert", default=False,
                    tooltip="Flip the height polarity if the relief reads inside-out.",
                ),
                io.Float.Input(
                    "headroom", default=0.15, min=0.02, max=0.35, step=0.01,
                    tooltip="Share of the value range each tail gets before the map "
                            "reaches 0 / 1. A hard limit here is a bug, not a style: "
                            "the height is differentiated downstream, so a clamped "
                            "plateau becomes an exactly flat facet (measured: 30 blobs "
                            "with dh == 0 and n == (0,0,1), largest 1600 px). The middle "
                            "99 % stays exactly linear, so raising this only rescales "
                            "relief amplitude by (1 - 2*headroom) -- compensate with "
                            "bump_strength. 0.15 (the default) measured: float plateaus "
                            "8003 -> 0 px, largest stored-map flat blob 1695 -> 85 px, "
                            "amplitude x0.70 so bump_strength 22 -> 31.4.",
                ),
            ],
            outputs=[io.Image.Output(display_name="height")],
        )

    @classmethod
    def execute(cls, normal, low_freq_radius: int = 96, invert: bool = False,
                headroom: float = 0.15) -> io.NodeOutput:
        n = _decode_normal(normal)
        nz = n[..., 2].clamp_min(1e-4)
        # Mirror of _normal_from_height: nx = -dh/dx, ny = +dh/dy with rows downward.
        # Single-channel maps keep BHW, which is already the shape fft2 works on.
        p = -n[..., 0] / nz
        q = n[..., 1] / nz

        b, h, w = p.shape
        fy = torch.fft.fftfreq(h, device=p.device, dtype=p.dtype).view(1, h, 1)
        fx = torch.fft.fftfreq(w, device=p.device, dtype=p.dtype).view(1, 1, w)
        denom = fx ** 2 + fy ** 2
        pf = torch.fft.fft2(p.float())
        qf = torch.fft.fft2(q.float())
        numer = -1j * (fx * pf + fy * qf)
        keep = denom > 1e-12
        safe = torch.where(keep, denom, torch.ones_like(denom))
        hf = torch.where(keep, numer / (2 * math.pi * safe), torch.zeros_like(numer))
        height = torch.fft.ifft2(hf).real

        if low_freq_radius > 0:
            height = height - _circular_blur(height.unsqueeze(-1), low_freq_radius)[..., 0]

        out = _gray3(_monotone_norm(height, headroom=headroom).unsqueeze(-1))
        if invert:
            out = 1.0 - out
        return io.NodeOutput(out)


class SeamlessDelightAlbedo(io.ComfyNode):
    """Remove the baked-in illumination, keeping the grain.

    The v2 LoRA paints a lit surface, so its output is albedo *times* shading.
    Dividing by the shading needs a shading estimate, and the obvious one --
    ``n.L`` from MoGe's normals -- was measured not to be reliable enough here
    (a best-fit light explains 0.01-0.47 of the high-frequency luminance). A
    flat-field division is used instead: the shading is taken to be the
    luminance low-passed at ``radius``, which is exactly the large-scale part of
    the lighting -- the global gradient, the vignetting and the broad soft cast
    shadows -- while relief-scale shading is left in place, because that is
    indistinguishable from albedo detail once the geometry is no longer trusted.

    All three channels are divided by the *same* luminance-derived field, so no
    colour cast is introduced. The result is rescaled to the original mean
    brightness, which keeps the material's exposure unchanged.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessDelightAlbedo",
            display_name="Seamless De-light Albedo",
            description=(
                "Divide a lit render by its low-frequency illumination so the albedo is "
                "usable under any lighting. The blur is circular, so the albedo tiles with "
                "the rest of the set. Relief-scale shading is deliberately left in: "
                "separating it from albedo would require normals trustworthy enough to "
                "divide by, and MoGe's are not."
            ),
            category="seamless/geometry",
            inputs=[
                io.Image.Input("image", tooltip="The lit render."),
                io.Int.Input(
                    "radius", default=128, min=1, max=1024, step=1,
                    tooltip="Scale above which luminance counts as lighting rather than "
                            "material. Keep it well above the grain size.",
                ),
                io.Float.Input(
                    "amount", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="1.0 = fully de-lit, 0.0 = untouched render. Useful at ~0.6 if "
                            "the render's shading is part of the look you want.",
                ),
                io.Boolean.Input("preserve_mean", default=True),
            ],
            outputs=[io.Image.Output(display_name="albedo")],
        )

    @classmethod
    def execute(cls, image, radius: int = 128, amount: float = 1.0,
                preserve_mean: bool = True) -> io.NodeOutput:
        # _luminance reads three channels; a single-channel map (an already
        # extracted luminance, say) is broadcast so the node still works.
        gray = image if image.shape[-1] >= 3 else image.repeat(1, 1, 1, 3)
        lum = _luminance(gray)
        shade = _circular_blur(lum, radius).clamp_min(1e-3)
        flat = image[..., :3] / shade
        if preserve_mean:
            flat = flat * (lum.mean() / flat.mean().clamp_min(1e-6))
        out = (image[..., :3] * (1.0 - amount) + flat * amount).clamp(0.0, 1.0)
        return io.NodeOutput(out)


class SeamlessRoughnessFromHeight(io.ComfyNode):
    """Roughness from the height field's micro-structure.

    ``SeamlessRoughnessFromAlbedo`` drives roughness off albedo *brightness*,
    which has no physical basis -- a dark grain is not smoother than a light one.
    Height micro-structure is a better proxy: roughness is what makes a surface
    scatter, and a surface whose micro-relief is energetic at short scales
    scatters more than a smooth one. It is still a proxy rather than a
    measurement -- nothing in a render without a specular highlight carries
    roughness -- so the range is exposed and the value can simply be pinned by
    setting min == max.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessRoughnessFromHeight",
            display_name="Seamless Roughness From Height",
            description=(
                "Map short-scale height energy into a roughness range. A proxy, not a "
                "measurement: with no specular highlight in the render there is nothing "
                "that actually observes roughness. Set min == max to pin a measured or "
                "art-directed constant instead."
            ),
            category="seamless/geometry",
            inputs=[
                io.Image.Input("height", tooltip="Height field, e.g. from Seamless Height From Normal."),
                io.Int.Input("window", default=12, min=1, max=256, step=1,
                             tooltip="Averaging window for the local micro-relief energy."),
                io.Float.Input("min_roughness", default=0.55, min=0.0, max=1.0, step=0.01),
                io.Float.Input("max_roughness", default=0.95, min=0.0, max=1.0, step=0.01),
                io.Float.Input("gamma", default=1.0, min=0.1, max=4.0, step=0.01),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[io.Image.Output(display_name="roughness")],
        )

    @classmethod
    def execute(cls, height, window: int = 12, min_roughness: float = 0.55,
                max_roughness: float = 0.95, gamma: float = 1.0,
                invert: bool = False) -> io.NodeOutput:
        h = _luminance(height)
        detail = h - _circular_blur(h, window)
        energy = _circular_blur(detail * detail, window).clamp_min(0.0).sqrt()
        driver = _percentile_norm(energy)
        if invert:
            driver = 1.0 - driver
        lo, hi = min(min_roughness, max_roughness), max(min_roughness, max_roughness)
        rough = lo + (hi - lo) * driver.clamp(0.0, 1.0).pow(gamma)
        return io.NodeOutput(_gray3(rough))
