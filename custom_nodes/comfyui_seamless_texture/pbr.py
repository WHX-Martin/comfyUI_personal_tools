"""PBR map derivation that *preserves* the torus.

Why derivation and not diffusion
--------------------------------
The sand LoRA was trained on 30 Poly Haven base-colour maps and nothing else
(`10_sand/*_Color.jpg`, `*_diff_1k.jpg`). It has never seen a normal or a
roughness map, so it cannot generate one -- no prompt can ask for a channel the
model has no representation of. A second problem rules out per-map diffusion
anyway: independently sampled maps are not pixel-aligned, so the bumps in the
normal map would not correspond to the features in the albedo. PBR sets only
work when every channel reads the *same* surface.

So the other four maps are derived, which is also what Substance, Materialize
and Blender's own bake pipeline do. The chain is:

    albedo --(high-pass luminance)--> height
    height --(circular central difference)--> normal
    height --(multi-scale cavity)--> ambient occlusion
    albedo --(luminance + local contrast)--> roughness
    ao + roughness + constant --> channel-packed ORM

The one thing that matters here
-------------------------------
A seam is a discontinuity, and every operator above can *create* one. A Sobel
filter with zero padding reads 0 outside the tile, so it puts a hard 1-pixel
seam into the normal map even when the height map it came from is perfectly
tileable -- the seam just moves from the albedo into the normal, where it is
much harder to see and much more annoying to debug.

Every spatial operator in this module is therefore periodic by construction:

* gradients use ``torch.roll``, which *is* the circular shift -- reading the
  neighbour across the boundary is what it does, not a special case;
* blurs pad with ``mode="circular"`` before pooling.

Net effect: if the input tiles, so does the output, for every map, at no extra
cost and with no second sampler pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from comfy_api.latest import io

__all__ = [
    "SeamlessHeightFromAlbedo",
    "SeamlessNormalFromHeight",
    "SeamlessAOFromHeight",
    "SeamlessRoughnessFromAlbedo",
    "PackPBRChannels",
]


def _luminance(images):
    """Rec.709 luminance as a single-channel BHWC tensor."""
    r, g, b = images[..., 0:1], images[..., 1:2], images[..., 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _gray3(x):
    """Single-channel BHWC -> 3-channel, so the result is a normal IMAGE."""
    return x.repeat(1, 1, 1, 3)


def _circular_blur(x, radius):
    """Box blur whose window wraps. ``radius=0`` is a no-op.

    ``F.avg_pool2d`` has no ``padding_mode``, so the wrap has to be done by hand
    -- and with an odd kernel of exactly ``2*radius+1`` the output size lands
    back on the input size, which is why the kernel is derived that way instead
    of being passed in.
    """
    if radius <= 0:
        return x
    b, h, w, c = x.shape
    kernel = 2 * radius + 1
    t = x.permute(0, 3, 1, 2)
    t = F.pad(t, (radius, radius, radius, radius), mode="circular")
    t = F.avg_pool2d(t, kernel_size=kernel, stride=1)
    return t.permute(0, 2, 3, 1)


def _percentile_norm(x, low=0.005, high=0.995):
    """Rescale to [0, 1] using robust percentiles rather than min/max.

    A single stray pixel (a lone bright pebble) would otherwise set the whole
    scale and flatten everything else. Quantiles are taken on a subsample: the
    result is visually identical and it keeps the node interactive on 4K maps.
    """
    flat = x.reshape(-1).float()
    if flat.numel() > 262144:
        flat = flat[:: flat.numel() // 262144]
    lo = torch.quantile(flat, low)
    hi = torch.quantile(flat, high)
    if float(hi - lo) < 1e-6:
        return torch.zeros_like(x)
    return ((x - lo) / (hi - lo)).clamp(0.0, 1.0)


def _monotone_norm(x, pivot=0.005, outer_pivot=0.0005, headroom=0.15):
    """Rescale to exactly [0, 1] with a mapping that is strictly increasing.

    Why this exists
    ---------------
    A height map is differentiated to produce the normal, so *any* hard limit in
    its value range becomes a zero-gradient plateau -- and a plateau is an
    exactly flat facet. ``_percentile_norm(x, 0.005, 0.995)`` ends in
    ``.clamp(0.0, 1.0)``, and on the gravel map that clamp turned the top and
    bottom 0.5 % of pixels into 30 blobs whose interior had ``dh == 0`` and whose
    normal was exactly ``(0, 0, 1)``. The largest was 1600 px, and several read
    as plainly cut-flat stones.

    Those tails are real geometry, not stray samples: each side holds 13-26 % of
    the field's total value range while carrying only 0.5 % of the pixels. So
    the fix is not a wider pivot (that only shrinks the cut, it never removes
    it) -- it is to stop cutting.

    How it works
    ------------
    Quantile-anchored piecewise-linear, six knots::

        xmin -> 0     q(outer) -> t        q(pivot) -> headroom
        q(1-pivot) -> 1-headroom           q(1-outer) -> 1-t        xmax -> 1

    Every segment has a strictly positive slope, so no pixel can land on a
    plateau, however far out it sits. The bulk segment is exactly linear, so for
    the middle 99 % this is a pure amplitude change -- gradients scale by
    ``1 - 2*headroom``, and dividing the previous ``bump_strength`` by that
    factor restores the previous relief exactly. Robustness is preserved too: an
    extreme outlier can only shallow out the outermost segment (0.05 % of
    pixels), because the bulk slope is pinned by ``headroom`` alone, not by the
    data's extremes.

    ``headroom`` is the share of the output range each inner tail receives; 0 is
    the old hard clamp. Measured on the gravel map at 0.15: float plateaus
    8003 -> 0 px, largest flat blob in the stored PNG 1695 -> 85 px, bulk
    amplitude x0.70 (so ``bump_strength`` 22 -> 31.4 for identical relief).
    """
    flat = x.reshape(-1).float()
    if flat.numel() > 262144:
        flat = flat[:: flat.numel() // 262144]
    a = float(min(max(headroom, 0.01), 0.35))
    t = a / 3.0
    xs = torch.stack([
        x.min(), torch.quantile(flat, outer_pivot), torch.quantile(flat, pivot),
        torch.quantile(flat, 1.0 - pivot), torch.quantile(flat, 1.0 - outer_pivot),
        x.max(),
    ])
    if float(xs[-1] - xs[0]) < 1e-6:
        return torch.zeros_like(x)
    ys = torch.tensor([0.0, t, a, 1.0 - a, 1.0 - t, 1.0],
                      device=x.device, dtype=torch.float32)
    idx = torch.bucketize(x, xs[1:-1])
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0).clamp_min(1e-12)


def _d_height(height):
    """Circular central difference of a single-channel BHWC height field.

    ``torch.roll`` by -1 alongside a dim is the wrapped neighbour in that
    direction, so the boundary is handled by the same expression as the
    interior. Returns (dh/dcolumn, dh/drow); row increases downward.
    """
    dx = (torch.roll(height, -1, dims=2) - torch.roll(height, 1, dims=2)) * 0.5
    dy = (torch.roll(height, -1, dims=1) - torch.roll(height, 1, dims=1)) * 0.5
    return dx, dy


def _normal_from_height(height, bump_strength, directx):
    """Height -> tangent-space normal in OpenGL convention.

    The sign of the green channel is the usual trap. With image rows increasing
    *downward* but the normal's Y axis pointing *up*, the surface normal works
    out to ``normalize(-dh/dx, +dh/dy_down, 1)``. Flipping that second sign
    yields the DirectX convention, which is what UE-style importers expect --
    choosing wrong makes every bump read as a dent.
    """
    dx, dy = _d_height(height)
    nx = -dx * bump_strength
    ny = dy * bump_strength
    if directx:
        ny = -ny
    nz = torch.ones_like(nx)
    n = torch.cat([nx, ny, nz], dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return n * 0.5 + 0.5


class SeamlessHeightFromAlbedo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessHeightFromAlbedo",
            display_name="Seamless Height From Albedo",
            description=(
                "Derive a height (displacement) field from an albedo's luminance. The "
                "high-pass default matters: the fine grain is real relief, while the "
                "slow luminance drift is usually baked-in shading that must not be "
                "re-lit as geometry. Set detail_mix to 0 to keep that drift as dunes."
            ),
            category="seamless/pbr",
            inputs=[
                io.Image.Input("image", tooltip="The albedo to derive height from."),
                io.Int.Input(
                    "blur_radius", default=8, min=0, max=256, step=1,
                    tooltip="High-pass radius in pixels. Roughly the scale below which "
                            "luminance counts as relief instead of lighting.",
                ),
                io.Float.Input(
                    "detail_mix", default=0.85, min=0.0, max=1.0, step=0.01,
                    tooltip="1.0 = pure high-pass (only relief), 0.0 = raw luminance.",
                ),
                io.Boolean.Input(
                    "invert", default=False,
                    tooltip="Flip brightness/height polarity. Bright grains look like they "
                            "sit proud by default; enable if the result reads inside-out.",
                ),
            ],
            outputs=[io.Image.Output(display_name="height")],
        )

    @classmethod
    def execute(cls, image, blur_radius: int, detail_mix: float, invert: bool) -> io.NodeOutput:
        lum = _luminance(image)
        detail = lum - _circular_blur(lum, blur_radius)
        base = lum - lum.mean()
        height = detail_mix * detail + (1.0 - detail_mix) * base
        height = _monotone_norm(height)
        if invert:
            height = 1.0 - height
        return io.NodeOutput(_gray3(height))


class SeamlessNormalFromHeight(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessNormalFromHeight",
            display_name="Seamless Normal From Height",
            description=(
                "Tangent-space normal map from a height map, converted with a circular "
                "central difference so a tileable height map yields a tileable normal "
                "map. A zero-padded Sobel here would inject a 1-pixel seam that the "
                "albedo does not have."
            ),
            category="seamless/pbr",
            inputs=[
                io.Image.Input("height", tooltip="Height field, e.g. from Seamless Height From Albedo."),
                io.Float.Input(
                    "bump_strength", default=4.0, min=0.0, max=200.0, step=0.1,
                    tooltip="How much slope a unit of height buys. Measured on a real "
                            "fine-sand map: 1.0 -> 4.9 deg mean tilt, 2.5 -> 11.5 deg, "
                            "4.0 -> 17.1 deg, 8.0 -> 28.0 deg (spiky). Fine sand sits "
                            "near 2.5; use 4-6 for coarse gravel. Judge it on a 2x2 tile "
                            "preview, never on the flat map alone.",
                ),
                io.Boolean.Input(
                    "directx", default=False,
                    tooltip="Off = OpenGL (+Y up, Blender/glTF/Unity default). On = DirectX "
                            "(flip green), which many UE-style importers expect.",
                ),
            ],
            outputs=[io.Image.Output(display_name="normal")],
        )

    @classmethod
    def execute(cls, height, bump_strength: float, directx: bool) -> io.NodeOutput:
        if bump_strength <= 0:
            flat = torch.zeros_like(height[..., :1]).repeat(1, 1, 1, 3)
            flat[..., 2] = 1.0
            return io.NodeOutput(flat)
        return io.NodeOutput(_normal_from_height(_luminance(height), bump_strength, directx))


class SeamlessAOFromHeight(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessAOFromHeight",
            display_name="Seamless AO From Height (cavity)",
            description=(
                "Cavity-style ambient occlusion: a pixel is darkened by how far it sits "
                "below the average of its neighbourhood, measured at several scales so "
                "both fine grain and broad hollows contribute. Every blur is circular, so "
                "the AO tiles with everything else. This is a local approximation, not a "
                "horizon search -- excellent for grain, but it will not reproduce "
                "long-range occlusion from real geometry."
            ),
            category="seamless/pbr",
            inputs=[
                io.Image.Input("height", tooltip="Height field, e.g. from Seamless Height From Albedo."),
                io.Int.Input(
                    "radius", default=24, min=1, max=512, step=1,
                    tooltip="Radius of the coarsest cavity scale, in pixels.",
                ),
                io.Int.Input(
                    "scales", default=4, min=1, max=8, step=1,
                    tooltip="How many octaves, each half the previous radius, averaged together.",
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=4.0, step=0.01),
                io.Float.Input("gamma", default=1.0, min=0.1, max=4.0, step=0.01),
                io.Int.Input(
                    "smooth", default=2, min=0, max=64, step=1,
                    tooltip="Final circular blur on the AO, in pixels. 0 leaves the occlusion crisp.",
                ),
            ],
            outputs=[io.Image.Output(display_name="ao")],
        )

    @classmethod
    def execute(cls, height, radius: int, scales: int, strength: float,
                gamma: float, smooth: int) -> io.NodeOutput:
        h = _luminance(height)
        cavity = torch.zeros_like(h)
        for i in range(scales):
            r = max(1, round(radius / (2 ** i)))
            cavity = cavity + (_circular_blur(h, r) - h)
        cavity = cavity / scales
        occlusion = (_percentile_norm(cavity) * strength).clamp(0.0, 1.0)
        ao = (1.0 - occlusion).pow(gamma)
        return io.NodeOutput(_gray3(_circular_blur(ao, smooth)))


class SeamlessRoughnessFromAlbedo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SeamlessRoughnessFromAlbedo",
            display_name="Seamless Roughness From Albedo",
            description=(
                "Map the albedo into a roughness range, blending overall luminance with "
                "local contrast (graininess). Luminance alone makes roughness and albedo "
                "perfectly correlated, which reads as plastic; the contrast term adds the "
                "variation that sells loose material. Both measurements use circular blurs."
            ),
            category="seamless/pbr",
            inputs=[
                io.Image.Input("image", tooltip="The albedo."),
                io.Int.Input("window", default=6, min=1, max=128, step=1,
                             tooltip="Local-contrast measurement window, in pixels."),
                io.Float.Input("contrast_mix", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="0 = roughness follows brightness only, 1 = follows graininess only."),
                io.Float.Input("min_roughness", default=0.55, min=0.0, max=1.0, step=0.01),
                io.Float.Input("max_roughness", default=0.98, min=0.0, max=1.0, step=0.01),
                io.Float.Input("gamma", default=1.0, min=0.1, max=4.0, step=0.01),
                io.Boolean.Input("invert", default=False),
            ],
            outputs=[io.Image.Output(display_name="roughness")],
        )

    @classmethod
    def execute(cls, image, window: int, contrast_mix: float, min_roughness: float,
                max_roughness: float, gamma: float, invert: bool) -> io.NodeOutput:
        lum = _luminance(image)
        # local std dev, via E[x^2] - E[x]^2, every expectation circularly blurred
        mean = _circular_blur(lum, window)
        var = (_circular_blur(lum * lum, window) - mean * mean).clamp_min(0.0)
        contrast = var.sqrt()

        driver = ((1.0 - contrast_mix) * _percentile_norm(lum)
                  + contrast_mix * _percentile_norm(contrast))
        if invert:
            driver = 1.0 - driver

        lo, hi = min(min_roughness, max_roughness), max(min_roughness, max_roughness)
        rough = lo + (hi - lo) * driver.clamp(0.0, 1.0).pow(gamma)
        return io.NodeOutput(_gray3(rough))


class PackPBRChannels(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PackPBRChannels",
            display_name="Pack PBR Channels (ORM)",
            description=(
                "Pack occlusion / roughness / metalness into one RGB texture, the "
                "R=AO G=Roughness B=Metalness layout engines and glTF expect. Leaves are "
                "optional: anything not wired falls back to a sane constant, so an "
                "occlusion-only or roughness-only pack needs no extra nodes. Sand and "
                "gravel are dielectrics, so the metalness default of 0 is the correct "
                "value, not a placeholder -- do not invent metalness for a ground material."
            ),
            category="seamless/pbr",
            inputs=[
                io.Image.Input("ao", optional=True, tooltip="Occlusion -> red. Unwired: 1.0."),
                io.Image.Input("roughness", optional=True, tooltip="Roughness -> green. Unwired: 0.5."),
                io.Image.Input("metalness", optional=True, tooltip="Metalness -> blue, overrides metalness_value."),
                io.Float.Input("metalness_value", default=0.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output(display_name="orm")],
        )

    @classmethod
    def execute(cls, ao=None, roughness=None, metalness=None,
                metalness_value: float = 0.0) -> io.NodeOutput:
        shape = None
        for candidate in (ao, roughness, metalness):
            if candidate is not None:
                shape = candidate.shape
                break
        if shape is None:
            raise ValueError(
                "Pack PBR Channels needs at least one wired input to take its size from."
            )
        b, h, w, _ = shape

        def channel(source, fallback, like):
            if source is None:
                return torch.full((b, h, w, 1), float(fallback),
                                  dtype=like.dtype, device=like.device)
            return _luminance(source)

        # ``like`` only needs to supply dtype/device, so any wired map will do.
        like = next(c for c in (ao, roughness, metalness) if c is not None)
        orm = torch.cat([
            channel(ao, 1.0, like),
            channel(roughness, 0.5, like),
            channel(metalness, metalness_value, like),
        ], dim=-1)
        return io.NodeOutput(orm.clamp(0.0, 1.0))
