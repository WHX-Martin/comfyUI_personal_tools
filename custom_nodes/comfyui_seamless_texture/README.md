# comfyui-seamless-texture

Seamless (四方连续) texture generation for ComfyUI, in four parts that share one
idea: **a seam is a discontinuity, so every operator that can introduce one has
to wrap.**

* **Padding half** — generate a tileable albedo by changing the sampling padding
  to circular instead of blending edges afterwards.
* **Geometry half** — recover the surface relief from the *lit* render with a
  convolutional normal estimator, then integrate it into a height map. This is
  what replaced the StableMaterials route: it is per-pixel, so the height lands
  under the albedo's own grain instead of being re-synthesised.
* **Derivation half** — turn that height / albedo into a full aligned PBR set,
  with every blur and gradient also circular so the seamlessness survives the
  derivation.
* **Super-resolution** — upscale the set without undoing either of the above.

ComfyUI 0.38.2 has no circular/seamless node left in core (`CircularVAEDecode`
was removed; the string `circular` does not appear once in `/object_info`).
This pack puts the capability back.

## Nodes

### Generation (padding)

| Node | Direction | What it does |
|---|---|---|
| `Seamless Model Padding (Circular)` | `MODEL → MODEL` | Runs the UNet on a **torus**: every `Conv2d` gets `padding_mode="circular"` so column 0 wraps to column W-1. No artificial border layer gets painted, so the result is periodic by construction. |
| `Seamless VAE Decode (Circular Padding)` | `LATENT, VAE → IMAGE` | Circular padding in the decoder, plus a circular margin that is cropped back off so the decoder's unresolvable edge rows never enter the tile. |
| `Seamless VAE Encode (Circular Padding)` | `IMAGE, VAE → LATENT` | Same for the encode direction (img2img / hires-fix / repaint). Needed because the VAE encoder's downsampler hardcodes an asymmetric zero pad. |
| `Tile Preview (2x2 check)` | `IMAGE → IMAGE` | Repeats the tile in a grid. A seam shows up here as a visible cross while staying invisible in the single tile. |

### Derivation (albedo → PBR)

| Node | Direction | What it does |
|---|---|---|
| `Seamless Height From Albedo` | `IMAGE → IMAGE` | High-pass luminance. The high-pass default matters: fine grain is real relief, slow luminance drift is usually baked shading that must not be re-lit as geometry. |
| `Seamless Normal From Height` | `IMAGE → IMAGE` | Tangent-space normal via a **circular central difference** (`torch.roll`, which *is* the wrapped neighbour). OpenGL by default, `directx` flips green. |
| `Seamless AO From Height (cavity)` | `IMAGE → IMAGE` | Multi-scale cavity occlusion. A local approximation, not a horizon search — excellent for grain, will not reproduce long-range occlusion. |
| `Seamless Roughness From Albedo` | `IMAGE → IMAGE` | Blends overall luminance with local contrast (graininess) into a roughness range. Luminance alone makes roughness perfectly correlated with albedo, which reads as plastic. |
| `Pack PBR Channels (ORM)` | `IMAGE×3 → IMAGE` | `R=AO G=Roughness B=Metalness`. All leaves optional, unwired ones fall back to constants. `metalness_value=0` is the *correct* value for sand and gravel, not a placeholder. |

### Geometry (normal estimator → height)

| Node | Direction | What it does |
|---|---|---|
| `Seamless DSINE Normal` | `IMAGE → IMAGE` | Normal map of a lit render, using **DSINE** — a purely convolutional estimator (EfficientNet-B5 + UpSampleBN + ConvGRU: zero attention blocks, no learned positional embedding) run on a periodically padded image. `fov` defaults to **3**, not DSINE's 60; `pad` sets the margin, `flip_red` the channel convention. |
| `Seamless MoGe Inference` | `IMAGE → MOGE_GEOMETRY` | MoGe-2 (ViT) through the same periodic padding. Kept as the **comparison branch**: a ViT's wrap discontinuity cannot be padded away (it stalls at a ~1.6× plateau), which is exactly why DSINE is the master geometry source. |
| `Seamless Flatten Normal` | `IMAGE → IMAGE` | Levels the best-fit base plane and rescales relief, so the only slope knob downstream is `bump_strength`. |
| `Seamless Height From Normal (Poisson)` | `IMAGE → IMAGE` | Periodic Poisson integration via FFT, then a circular high-pass at `low_freq_radius` to remove the low-frequency runaway that integration amplifies as `1/k²`. `headroom` controls normalisation — see "A height map must never be clipped". |
| `Seamless De-light Albedo` | `IMAGE → IMAGE` | Divides out a circular low-pass of luminance, removing baked large-scale lighting while leaving relief-scale shading in place. |
| `Seamless Roughness From Height` | `IMAGE → IMAGE` | Maps short-scale height energy into a roughness band. A proxy, not a measurement — nothing in a lit render observes roughness. |

The DSINE node needs `comfyui_controlnet_aux` importable (it supplies the
architecture) plus the `dsine.pt` checkpoint. Weights resolve from
`SEAMLESS_DSINE_WEIGHTS`, then
`<custom_nodes>/comfyui_controlnet_aux/ckpts/hr16/Diffusion-Edge/dsine.pt`, then
ComfyUI's usual `dsine` / `geometry_estimation` / `checkpoints` folders; if none
is found the node raises and prints the `curl` command that fetches it. The
architecture is imported rather than vendored, and the model is cached
in-process, because the upstream preprocessor rebuilds it on every call.

#### Why `fov` is the seam control

DSINE consumes **per-pixel ray directions**, and that is an absolute position
field, so it is not periodic. Padding alone does nothing at the default
`fov = 60`; narrowing the fov is what lets padding converge at all. Wrap excess
(horizontal), pad 0 → 512:

| fov | pad 0 | 128 | 256 | 512 |
|---|---|---|---|---|
| 60 (DSINE default) | 6.87 | 4.74 | 4.97 | 5.45 |
| 20 | 4.44 | 2.24 | 2.13 | 2.29 |
| 10 | 4.21 | 1.64 | 1.47 | 1.57 |
| 5 | 4.20 | 1.36 | 1.19 | 1.25 |
| **3 (default here)** | 4.24 | 1.29 | **1.11** | 1.14 |

At `fov = 3 / pad = 256` the wrap excess is 1.11 — statistically
indistinguishable from an ordinary interior step — and `iterations` (the ConvGRU
passes, 1–20) turns out to be a near-empty knob. A material map has no camera,
so `fov → 0` (orthographic) is also the more physical prior. Full A/B against
the same map through MoGe, and the Blender render used to confirm it:

| | wrap excess H / V | mean tilt | `corr(\|∇n\|, \|∇I\|)` |
|---|---|---|---|
| MoGe ViT, pad 1024 | 1.77 / 2.68 | 7.64° | 0.106 |
| **DSINE, fov 3, pad 256** | **1.11 / 1.64** | 15.08° | **0.267** |

DSINE's relief is 3.5× stronger and holds 2.5× more edge correlation; the honest
caveat is that `corr` is a weak proxy (it shows slope *magnitude* tracks image
edges, not that the direction is right) and that a scene-trained model at
`fov = 3` is out of distribution.

### Super resolution

| Node | Direction | What it does |
|---|---|---|
| `Seamless Upscale` | `UPSCALE_MODEL, IMAGE → IMAGE` | Extends the tile periodically, runs a super-resolution model, crops the centre back out. Fixes the one place an SR network actually breaks tiling — the image border. `circular=False` gives the plain upscale as a control. |
| `Seamless Resize` | `IMAGE → IMAGE` | Periodic `common_upscale` (area / bicubic / …) with the same extend-and-crop margin. Measured: `area` at ×0.5 is bit-exact against a 2×2 mean, and both directions are translation-equivariant to 0.000/255. |

### Model-generated (StableMaterials) — superseded

> **Superseded**, kept for reference. The image condition is one pooled global
> CLIP-Vision token, so this route re-synthesises a material of similar
> *character* rather than transferring your albedo's own pattern: the height map
> does not sit under the albedo's grain. The per-pixel geometry chain above is
> what replaced it. Numbers behind that conclusion are in the bullets below.

| Node | Direction | What it does |
|---|---|---|
| `StableMaterials: Albedo -> PBR (seamless)` | `SM_PIPELINE, IMAGE → IMAGE×5` | Runs StableMaterials (LCM UNet) on a tileable albedo and returns basecolor / normal / height / roughness / metallic **in one pass, so they are aligned by construction**. Optional prompt fused along the sequence axis. `strict_torus` forces every UNet conv to circular padding for the whole run. |

Requires `P5ina/comfyui-stable-materials` for `Load StableMaterials`, plus the
gvecchio/StableMaterials weights in the HF cache. The upstream pack's own image node
is unusable: it passes `image=` / `image_guidance_scale=`, neither of which exists in
the pipeline's `__call__`, so both are swallowed by `**kwargs` and the conditioning is
silently dropped — and with the default empty prompt it then forwards `prompt=None`,
which `check_inputs` rejects.

Four measured facts that shaped this node (numbers and method in the workflow's own
note, `workflows/sdxl_sand_pbr_stablematerials.json`):

* **The image condition is global, not spatial.** `__encode_image` takes
  `.image_embeds` from `CLIPVisionModelWithProjection` — one pooled 768-d token.
  Block-shuffling the input changes the output as much as replacing it with flat grey
  (mean |Δ| 0.084 vs 0.083), so the node re-synthesises a material of similar *character*
  rather than transferring the input's pattern. Do not expect the height map to sit
  under the albedo's own grain.
* **The text prompt is what keeps the physics right.** With the albedo alone, metallic
  came out at 0.099 for sand; fused with the prompt, 0.001. CFG is inactive on this
  model (`time_cond_proj_dim=256` disables it), so fusing along the *sequence* axis is
  unconstrained — and it is the reason this node exists rather than reusing upstream.
* **4 steps is not enough.** The LCM distillation default leaves the normal essentially
  flat (`mean |xy| = 0.025`); 16 steps gives 0.057 at ~1 s per generation.
* **`strict_torus` is not a free win.** It measured *no* improvement in the seam metric
  (worst wrap percentile 82 vs 68 at 16 steps), so it defaults off. `tileable` must stay
  on regardless — it also gates the decoder's circular margin.

## Why derivation, not diffusion

If you trained an LoRA on base-colour maps only, it cannot generate a normal or
roughness map — it has no representation of those channels to ask for. And even
with a model that could, sampling each map independently gives maps that are not
pixel-aligned, so the normal's bumps would not correspond to the albedo's
features. PBR sets only work when every channel reads the *same* surface. That
is why Substance, Materialize and Blender's bake pipeline all derive rather than
generate the non-albedo channels.

The derivation half of this pack is one answer to that, and it has an honest
ceiling: it infers roughness from albedo *luminance*, which is not a physical
relationship at all. Where a model has actually learned the joint distribution —
StableMaterials is trained on 6,198 real PBR materials — use it instead, and note
that it satisfies the alignment requirement *by construction*: one denoising pass
emits all five maps. The two routes are not competitors:

| | derivation (`pbr.py`) | StableMaterials |
|---|---|---|
| spatial fidelity to your albedo | **exact** (height *is* derived from it) | none — the condition is a single global token |
| roughness / metallic | guessed from luminance | learned, physically plausible |
| normal | derived from the height you use | predicted, i.e. independent information |
| requires | nothing extra | ~5.5 GB of weights + diffusers |

## A height map must never be clipped

A height map gets differentiated to produce the normal, so **any hard limit in
its value range becomes a zero-gradient plateau — and a plateau is an exactly
flat facet.** The original normalisation ended in `.clamp(0.0, 1.0)` at the
0.005 / 0.995 percentile pivots. On a real gravel map that turned the top and
bottom 0.5 % of pixels into 30 blobs whose interior had `dh == 0` and whose
normal was exactly `(0, 0, 1)`; the largest was 1600 px, i.e. several stones
visibly cut flat. (The estimator was not at fault: DSINE's own output had 6 such
pixels, not 5000.)

Widening the pivot does not fix it, only shrink the cut. The two tails are
**real geometry, not stray samples**: on that map the middle 99 % of pixels
(`q0.005 .. q0.995`) spans just **60 %** of the value range, while the 0.5 %
tail at each end holds **13 %** (low) and **26 %** (high).

`_monotone_norm` replaces the clamp with a quantile-anchored
**piecewise-linear** map: six knots, every segment with a strictly positive
slope, so no pixel can land on a plateau however far out it sits.

| | float plateaus | largest flat blob in the saved map | bulk amplitude |
|---|---|---|---|
| `.clamp(0, 1)` | 8003 px (largest 1690) | 1695 px | ×1.00 |
| **piecewise, `headroom = 0.15`** | **2 px** | **85 px** | **×0.70** |

The middle 99 % is still exactly linear between the pivots, so the change is a
pure amplitude rescale: gradients move by `1 - 2*headroom`, and dividing the old
`bump_strength` by that factor (**22 → 31.4**) restores the previous relief
exactly. Robustness survives too — the bulk segment's slope is pinned by
`headroom` alone, so an extreme outlier can only shallow out the outermost
segment (0.05 % of pixels); it cannot compress the main relief the way a min/max
normalisation would.

Two things worth knowing before auditing or "fixing" this again:

- **Judge on the float plateaus, not on the bytes.** The normal is computed
  in-pipeline from the float image, so `dh == 0` *there* is the artefact. A
  genuinely gentle slope can quantise to equal bytes without the normal being
  flat, so the 8-bit zero-gradient count is only a storage-fidelity proxy and
  the two numbers must not be compared.
- **Two tempting tail formulas are broken.** A mirrored tail written as
  `sign(s) * tail` drives the lower half **negative**, and clipping that to 0
  re-creates the very plateau being removed; a hyperbolic tail
  `a / (1 + |s| k)` keeps a continuous slope at the knee but its denominator
  **changes sign** past `|s| = a / (1 - 2a)`. Only a piecewise-linear map is
  positive-slope by construction over the whole domain, which is why the
  regression test asserts it directly (`dy < 0` must be 0, and no pair of inputs
  differing by more than 1e-3 may map to the same output).

## Why circular padding is exact

With `zeros` padding the feature map is a rectangle and the model compensates by
painting a border — that border is the seam. With circular padding the feature
map is topologically a torus. Every remaining operation in the pipeline
(pointwise sampler updates, nearest-neighbour upsampling, attention, group norm)
commutes with that periodicity, and the random noise itself is just an arbitrary
field *on* the torus — it does not need to look periodic. So the final latent is
a function on the torus and the decoded tile repeats with no seam, not merely a
faint one.

The same argument applies one level down: a zero-padded Sobel on a tileable
height map injects a 1-pixel seam into the normal map. Hence `torch.roll` for
gradients and `mode="circular"` for every blur in the derivation nodes.

Four properties of the host install were verified before relying on this:

- `comfy.ops.Conv2d.forward` reaches `torch.nn.Conv2d._conv_forward`, the path
  that honours `padding_mode`. A subclass calling `F.conv2d` directly would
  ignore it.
- The SDXL UNet contains **no hand-rolled `F.pad`**; its downsampler is
  `conv 3x3 / stride 2 / pad 1`, which wraps correctly.
- The SDXL VAE **decoder** is nearest-upsample + `conv 3x3 / pad 1`, i.e. clean.
  Only the **encoder's** `Downsample` hardcodes `F.pad(x, (0, 1, 0, 1))`, which
  is why the encode node pads and crops.
- `vae.decode` returns **BHWC** (it ends in `.movedim(1, -1)`), `vae.encode`
  takes BHWC and returns BCHW. Getting this backwards silently crops the channel
  axis instead of the spatial margin.

## Upscaling is where a seamless tile usually stops being seamless

A feed-forward SR network pads with zeros (or reflect) at the **image** border, so
the first and last few output pixels are computed from neighbours that do not
exist on a torus. Everything else it does is fine. Measured on a 512 tile through
RealESRGAN_x2, comparing a plain upscale against the periodic answer:

| where | mean abs difference |
|---|---|
| border band (first/last 32 px) | **0.00422** |
| interior (256 px in from the border) | **0.00010** |

So the interior is already exactly right and only the rim is wrong — the failure
is a boundary condition, not a filter, and the fix is to extend the tile
periodically before running the network and crop the centre back out. The network
never sees an edge, and the crop is periodic by construction.

How much padding is enough is measurable instead of guessable. Running the same
upscale at pad = 64/128/256/512 and comparing each against pad = 512:

| pad | mean | max | verdict |
|---|---|---|---|
| 64 | 0.000243 | 0.0187 | not converged |
| 128 | 0.000077 | 0.0031 | converged |
| 256 | 0.000055 | **0.00064** | converged past 8-bit precision |

Once more padding stops changing the result the network has stopped seeing the
outer boundary, so the crop equals the operator applied to the infinite periodic
extension of the tile. That is the actual proof of tileability.

### Two things not to super-resolve

- **Normal maps.** A normal map's channels are vector components, not colours. A
  network trained on photographs denoises and sharpens them as if they were,
  which breaks unit length and rounds off the places where the direction flips.
  Upscale **height**, then re-derive the normal with `Seamless Normal From
  Height`.
- **Roughness / metallic.** Low-frequency and structureless. The network's job is
  to invent detail, so it will fabricate structure that does not match the height
  map and the set stops being physically consistent. Resample these with a plain
  interpolation.

Splitting a set this way costs nothing and keeps it consistent: measured
`albedo high-pass` vs `height high-pass` correlates at r = +0.989, and
`normal slope` vs `saved height gradient` at r = 1.000.

## Why it does not poison ComfyUI's model cache

Mutating a cached model's convs in place would leak: ComfyUI keeps the patcher
alive across runs, so an unrelated later workflow would silently keep generating
on a torus. Here the UNet patch is installed as a `model_function_wrapper` on a
**clone** of the patcher and applied only for the duration of each UNet call,
then reverted. The VAE nodes use the same scoped-context approach.

## Usage

The current chain starts from the **lit** render, because that is what the sand
LoRA was trained on (see `workflows/sdxl_sand_pbr_v2_geom.json`):

```
CheckpointLoaderSimple → LoraLoader → Seamless Model Padding → KSampler → Seamless VAE Decode
                                                                      ↓        ↘ Seamless De-light Albedo → albedo
                                        Seamless DSINE Normal  ←───────┘
                                                 ↓
                                        Seamless Flatten Normal
                                                 ↓
                              Seamless Height From Normal (Poisson) ─┬─→ Normal From Height → normal
                                                                      ├─→ AO From Height ─────┐
                                                                      └─→ Roughness From Height ─┴─→ Pack PBR Channels → orm
```

`Seamless Model Padding` goes **after** the LoRA loader and **before** the
sampler. The knobs that matter most, in order:

* `bump_strength` on the normal node — measured on a real fine-sand map,
  1.0 → 4.9° mean tilt, 2.5 → 11.5°, 4.0 → 17.1°, 8.0 → 28.0° (spiky). Judge it
  on the 2×2 preview, never on the flat map.
* `fov` on `Seamless DSINE Normal` — the seam control; 3 by default.
* `headroom` on the Poisson node — normalisation contrast. It rescales relief by
  `1 - 2 * headroom`, so scale `bump_strength` by the reciprocal.

### Variant: height from albedo (no estimator weights)

```
CheckpointLoaderSimple → LoraLoader → Seamless Model Padding
                       ↘ CLIPTextEncode ×2 ↘
                              EmptyLatentImage → KSampler → Seamless VAE Decode
                                                                    ↓            ↘ albedo → SaveImage
                          Seamless Height From Albedo ──┬─→ Normal From Height → normal
                                                       ├─→ AO From Height ──┐
                          Seamless Roughness From Albedo ─────────────────────┴→ Pack PBR Channels → orm
```

Derives relief from the albedo's own high-pass rather than from an estimator, so
it needs no extra weights — but it can only recover what the render's lighting
actually reveals, and it cannot tell relief from baked shading. Use it when the
render is flat-lit, or as a cross-check against the geometry chain.

## Known measurement caveats

ComfyUI's `SaveImage` writes `np.clip(...).astype(np.uint8)`, which **truncates
rather than rounds**. Every float map therefore lands up to 1/255 low per
channel. For a normal map that is a systematic ~2.3° steeper reading (measured:
tensor `n_z` 0.96252 → saved 0.95772). Harmless in practice, since renderers
renormalise, but do not read a saved normal's tilt as the node's output.

**The wrap-difference metric is relative, so it only works within one operator.**
`verify_seamless.py` divides the wrap difference by the distribution of
inside-image differences. That is sound when a single flag is flipped and the
operator is otherwise identical (it is what made the circular-vs-zeros padding
A/B decisive). It is **not** sound across operators, because the operator changes
the distribution it divides by — super resolution sharpens everything, so the
ratio moves even when nothing is wrong. A 1024 seam-free upscaled tile reads
"ratio 1.39, 96th percentile → SEAM" while being provably periodic. Use
distribution-free tests across operators:

- **Convergence** — if more padding stops changing the result, the boundary
  influence is gone.
- **Shift equivariance** — `upscale(roll(src, d)) == roll(upscale(src), d*scale)`.
  Measured max deviation 0.16–0.17/255 for d = 64/128/256, i.e. equivariant, which
  makes the output a function on the torus.

**Super resolution amplifies input quantisation.** A 1-LSB change in the input
(0.50/255) came out of RealESRGAN_x2 as up to **7.18/255**. So do not push a map
through an 8-bit save/load round trip in the middle of an SR chain, and expect
slowly-growing discrepancies when comparing a float reference against a saved
tile — match the input quantisation before concluding anything.

