"""ComfyUI nodes for pulling *several objects out of a single scene*.

A VLM looks at the image and names what it sees; these nodes turn that reply into
something the rest of the graph can act on, one pass per object:

* ``ObjectExtract_VLMQuery`` does the asking itself: it sends the image to any
  OpenAI-compatible vision endpoint and hands back both the raw reply and the
  parsed result. Reach for it when the VLM should run *inside* the graph instead
  of you pasting a reply in by hand.
* ``ObjectExtract_PromptsFromVLM`` turns prose/JSON into a list of located
  objects -- a comma-separated ``name:max_detections`` prompt plus pixel-space
  bounding boxes. That is exactly what a grounding segmenter (SAM3) consumes, so
  it can drive precise masks. An unparseable reply yields empty values, which
  downstream ignores instead of failing.
* ``ObjectExtract_BBoxesToMasks`` skips the segmenter entirely: it paints one
  filled rectangle per box, giving a *coarse* mask batch. Crop to each rectangle
  and a generative editor (Qwen-Image 2.1 and friends) re-draws that one object
  cleanly -- the box only has to say *where*, the editor decides *what*. Useful
  when a precise segmenter is unavailable or its edges are not trustworthy.

Both degrade safely rather than raising, so a bad reply never kills a queue run.
``ObjectExtract_VLMQuery`` is the one exception: a *transport* failure (bad key,
unreachable host) raises by default, because that is a configuration bug and
silently continuing would burn a sampler run on a nonsense prompt. Set
``on_error`` to ``fallback`` to opt out of that.
"""

from __future__ import annotations

import asyncio
from io import BytesIO

import numpy as np
import torch
from PIL import Image as PILImage

import comfy.model_management
from comfy_api.latest import io

from . import parse, vlm_api

__all__ = [
    "ObjectExtractPromptsFromVLM",
    "ObjectExtractBBoxesToMasks",
    "ObjectExtractVLMQuery",
]

# Default caption for the coarse-mask route. Wording matters, but not the way you
# would expect: a rectangle mask keeps *everything* inside the box, background
# clutter included, so the prompt has to do the "which object?" work.
#
# Measured on Qwen-Image 2.1 (cfg=1, euler/simple, 25 steps, same crop, 3 seeds):
#   - plain "isolate the X, remove the background and everything else" -> the
#     model's magenta extraction backdrop, subject cleanly separated, 3/3.
#   - demanding "pure white studio background, seamless white backdrop, no
#     background clutter at all" -> white background sometimes, but the model
#     then *keeps* the stray object that was also inside the box, 2/3. Worse.
# So do not fight the background here; fight for the separation. Also note that
# cfg=1 makes negative_prompt a literal no-op (verified: identical pixels with
# and without it), so all steering has to live in this positive prompt.
ISOLATE_TEMPLATE = (
    "isolate the single {name}, remove the background and everything else, "
    "keep its original shape, colour and texture"
)


def _image_size(image):
    """Return ``(height, width, came_from_image)`` for a BHWC/HWC tensor."""
    shape = getattr(image, "shape", None)
    if shape is None:
        return parse.DEFAULT_SIZE, parse.DEFAULT_SIZE, False
    dims = tuple(int(v) for v in shape)
    if len(dims) == 4:  # BHWC
        return dims[1], dims[2], True
    if len(dims) == 3:  # HWC
        return dims[0], dims[1], True
    return parse.DEFAULT_SIZE, parse.DEFAULT_SIZE, False


def _format_box(index: int, box: dict, name: str) -> str:
    label = f"  {name}" if name else ""
    return "  #%-2d [%5d,%5d,%5d,%5d]%s" % (
        index, box["x"], box["y"], box["width"], box["height"], label
    )


def _report(text, prompt, boxes, names, space, width, height, from_image, dropped) -> str:
    """Human-readable summary; also doubles as the node's only debugging window."""
    lines = []

    if prompt:
        lines.append("prompt : " + prompt)
    elif text and str(text).strip():
        lines.append("prompt : (empty - nothing matched name:count)")
    else:
        lines.append("prompt : (empty - no input text)")

    if boxes:
        size_note = " " if from_image else " (assumed, no image wired)"
        dedupe_note = f"  (dropped {dropped} duplicate)" if dropped else ""
        lines.append(
            f"boxes  : {len(boxes)}  [image {width}x{height}, {space}]"
            f"{size_note}{dedupe_note}"
        )
        shown = boxes if len(boxes) <= 12 else boxes[:6]
        for i, (box, name) in enumerate(zip(shown, names), start=1):
            lines.append(_format_box(i, box, name))
        if len(boxes) > len(shown):
            lines.append(f"  ... {len(boxes) - len(shown)} more")
    elif text and str(text).strip():
        lines.append("boxes  : none found (SAM3 will ignore the bboxes port)")

    if boxes and prompt:
        lines.append("note   : prompt folded from the box labels -- usable as a SAM3 "
                     "text prompt, or ignored when the boxes go to BBoxes to Masks")
    elif boxes:
        lines.append("note   : boxes only, no name:count built -- SAM3 segments inside "
                     "each box, one mask per box")

    return "\n".join(lines)


class ObjectExtractPromptsFromVLM(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ObjectExtract_PromptsFromVLM",
            display_name="Object Prompts from VLM",
            category="object_extract",
            description=(
                "Locate every object a vision model named in its free-form reply: outputs a "
                "comma-separated 'name:max_detections' prompt and pixel-space bounding boxes, "
                "one box per object. Accepts JSON (fenced, bare-key or nested) or plain prose, "
                "and rescues markdown fences, trailing commas and 0-1000 normalized "
                "coordinates. Wire 'bboxes' into SAM3 Detect for precise masks, or into "
                "'BBoxes to Masks' when a rectangle is all you need; an unparseable reply "
                "returns empty values, which downstream ignores instead of failing."
            ),
            search_aliases=[
                "vlm", "sam3", "grounding", "bounding box", "bbox",
                "json to bbox", "prompt from image", "qwen vl", "detection prompt",
                "object extract", "multi object", "scene objects",
            ],
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Raw VLM reply. Also usable as a manual text box when nothing is wired in.",
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Image the reply describes. Only used for its width/height, to "
                            "convert normalized coordinates into pixels (1024x1024 assumed if unwired).",
                ),
                io.Combo.Input(
                    "coord_mode",
                    options=list(parse.COORD_MODES),
                    default="auto",
                    tooltip="How to read the numbers. auto guesses from the image size and value range.",
                ),
                io.Combo.Input(
                    "box_format",
                    options=list(parse.BOX_FORMATS),
                    default="auto",
                    tooltip="auto treats a bare 4-number list as x1,y1,x2,y2 and a dict with "
                            "width/height as x,y,w,h.",
                ),
                io.Int.Input(
                    "pad",
                    default=0, min=0, max=512,
                    tooltip="Grow every box by this many pixels on each side before clamping.",
                ),
                io.Int.Input(
                    "min_size",
                    default=4, min=1, max=4096,
                    tooltip="Drop boxes narrower or shorter than this. Guards against stray numbers.",
                ),
                io.Boolean.Input(
                    "dedupe",
                    default=True,
                    tooltip="Drop boxes that nearly duplicate one already kept (IoU >= 0.95). "
                            "VLMs repeat themselves on long replies, which would inflate every ':N'.",
                ),
                io.Int.Input(
                    "count_cap",
                    default=20, min=0, max=200,
                    tooltip="Upper bound for the ':N' max-detections value. 0 leaves counts uncapped.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.BoundingBox.Output(display_name="bboxes"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, text, image=None, coord_mode="auto", box_format="auto",
                pad=0, min_size=4, dedupe=True, count_cap=20) -> io.NodeOutput:
        height, width, from_image = _image_size(image)

        # Same reasoning-model problem as in the query node, reached by the other
        # door: a reply pasted in by hand. Thinking is fluent prose about sizes
        # and positions, so every number in it parses as a coordinate -- and those
        # boxes are pure invention.
        text, reasoning = vlm_api.split_thinking(text)

        boxes, names, space = parse.parse_boxes(
            text, width, height, coord_mode, box_format, pad, min_size
        )
        raw_count = len(boxes)
        if dedupe:
            boxes, names = parse.dedupe_boxes(boxes, names)
        dropped = raw_count - len(boxes)

        pairs = parse.count_names(names) if boxes else []
        if not pairs:
            # No usable boxes: fall back to whatever "name:count" text exists.
            pairs = parse.parse_prompts(text)
        prompt = parse.build_prompt(pairs, count_cap)

        report = _report(text, prompt, boxes, names, space, width, height,
                         from_image, dropped)
        if reasoning:
            report += (f"\nthink  : {len(reasoning)} chars of reasoning ignored, "
                       f"final answer only")
        # The label rides along inside each box dict (consumers only read the four
        # coordinate fields), which lets a downstream mask node caption every crop
        # without needing a second, index-aligned wire for the names.
        labelled = [dict(box, label=name) for box, name in zip(boxes, names)]
        return io.NodeOutput(prompt, labelled, report)


def _canvas_size(image, width: int, height: int):
    """Resolve the mask canvas from the wired image, else the widget values."""
    shape = getattr(image, "shape", None)
    if shape is None:
        return width, height, "widgets"
    dims = tuple(int(v) for v in shape)
    if len(dims) == 4:  # BHWC
        return dims[2], dims[1], "image"
    if len(dims) == 3:  # HWC
        return dims[1], dims[0], "image"
    return width, height, "widgets"


def _paint_masks(boxes, height: int, width: int):
    """Rasterise one hard-edged rectangle per box into a ``[N, H, W]`` mask batch."""
    device = comfy.model_management.intermediate_device()
    canvas = torch.zeros((len(boxes), height, width), dtype=torch.float32, device=device)
    for i, box in enumerate(boxes):
        y0, x0 = box["y"], box["x"]
        canvas[i, y0:y0 + box["height"], x0:x0 + box["width"]] = 1.0
    return canvas


def _mask_report(boxes, shape_note, width, height, size_src, grow, min_size,
                 limit, raw_count, fell_back, prompts=None) -> str:
    lines = [f"canvas : {width}x{height} ({size_src})",
             f"input  : {raw_count} box(es) read as {shape_note}"]

    tunables = []
    if grow:
        tunables.append(f"grow {grow:+d}px")
    if min_size > 1:
        tunables.append(f"min_size {min_size}px")
    if limit:
        tunables.append(f"limit {limit}")
    if tunables:
        lines.append("filters: " + ", ".join(tunables))

    if fell_back:
        lines.append("masks  : 0 boxes in -> 1 full-image mask (if_empty = full_image)")
    elif boxes:
        total = width * height
        lines.append(f"masks  : {len(boxes)} rectangle(s)")
        shown = boxes if len(boxes) <= 12 else boxes[:6]
        for i, box in enumerate(shown, start=1):
            area = box["width"] * box["height"] / total * 100 if total else 0.0
            label = ("  " + box["label"]) if box.get("label") else ""
            lines.append("  #%-2d [%5d,%5d,%5d,%5d]  %4dx%-4d %5.1f%%%s" % (
                i, box["x"], box["y"], box["width"], box["height"],
                box["width"], box["height"], area, label,
            ))
        if len(boxes) > len(shown):
            lines.append(f"  ... {len(boxes) - len(shown)} more")
    else:
        lines.append("masks  : none (if_empty = none -> nothing downstream runs)")

    dropped = raw_count - (0 if fell_back else len(boxes))
    if dropped > 0 and not fell_back:
        lines.append(f"dropped: {dropped} box(es) (degenerate, under min_size, or past limit)")

    if prompts:
        lines.append(f"prompts: {len(prompts)} rendered from the template")
        lines.append("  #1  " + (prompts[0][:120] or "(empty)"))

    return "\n".join(lines)


class ObjectExtractBBoxesToMasks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ObjectExtract_BBoxesToMasks",
            display_name="BBoxes to Masks",
            category="object_extract",
            description=(
                "Paint one filled rectangle per bounding box, producing a *coarse* mask "
                "batch ([N, H, W]) instead of running a segmenter. Feed the batch into "
                "'Masks To Mask List' and each rectangle drives one downstream pass -- "
                "the intended use is cropping to the box and handing that rough region to "
                "a generative editor (Qwen-Image 2.1) so it isolates the single subject. "
                "The 'prompts' output renders one caption per box from the template below, "
                "using each box's label, so the editor is told *which* object to pull out. "
                "Accepts a bare box dict, a flat box list, or batched frames. No boxes in "
                "means either one full-image mask (so the graph still runs) or nothing at "
                "all, depending on if_empty."
            ),
            search_aliases=[
                "bbox to mask", "bounding box to mask", "rect mask", "coarse mask",
                "box mask", "no sam", "vlm mask", "qwen image", "crop box",
                "object extract", "multi object", "scene objects",
            ],
            inputs=[
                io.BoundingBox.Input(
                    "bboxes",
                    force_input=True,
                    tooltip="Boxes to rasterise. A bare dict, a flat list, or image-batch "
                            "frames are all accepted; for frames only the first non-empty "
                            "one is used.",
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Sets the mask canvas to this image's size. Without it the "
                            "width/height widgets below are used instead.",
                ),
                io.Int.Input(
                    "width",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Canvas width, used only when no image is wired in.",
                ),
                io.Int.Input(
                    "height",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Canvas height, used only when no image is wired in.",
                ),
                io.Int.Input(
                    "grow",
                    default=0, min=-256, max=512,
                    tooltip="Pad each box by this many pixels per side before clamping. "
                            "Negative values trim it -- handy when the VLM box is generous.",
                ),
                io.Int.Input(
                    "min_size",
                    default=8, min=1, max=4096,
                    tooltip="Drop boxes narrower or shorter than this after growing.",
                ),
                io.Int.Input(
                    "limit",
                    default=0, min=0, max=200,
                    tooltip="Keep at most this many boxes, so a runaway reply cannot queue "
                            "hundreds of sampler runs. 0 keeps them all.",
                ),
                io.Combo.Input(
                    "if_empty",
                    options=["full_image", "none"],
                    default="full_image",
                    tooltip="What to emit when no box survives: one mask covering the whole "
                            "canvas (graph still produces output), or an empty batch "
                            "(nothing downstream runs).",
                ),
                io.String.Input(
                    "prompt_template",
                    multiline=True,
                    default=ISOLATE_TEMPLATE,
                    tooltip="Caption rendered for each box, with {name} replaced by that "
                            "box's label. Drop the placeholder to send the same prompt for "
                            "every box. A label-less box reads as 'main object'.",
                ),
            ],
            outputs=[
                io.Mask.Output(display_name="masks"),
                io.String.Output(display_name="prompts", is_output_list=True),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, bboxes, image=None, width=1024, height=1024, grow=0,
                min_size=8, limit=0, if_empty="full_image",
                prompt_template=ISOLATE_TEMPLATE) -> io.NodeOutput:
        raw_boxes, shape_note = parse.normalize_boxes(bboxes)
        raw_count = len(raw_boxes)

        width, height, size_src = _canvas_size(image, width, height)
        boxes = parse.fit_boxes(raw_boxes, width, height, grow, min_size, limit)

        fell_back = False
        if not boxes and if_empty == "full_image":
            boxes = [{"x": 0, "y": 0, "width": width, "height": height, "label": ""}]
            fell_back = True

        masks = _paint_masks(boxes, height, width)
        # One caption per mask, in the same order -- the executor slices every list
        # input at the same index, so caption i pairs with crop i without any
        # explicit synchronisation.
        prompts = [parse.render_prompt(prompt_template, b.get("label", "")) for b in boxes]
        report = _mask_report(boxes, shape_note, width, height, size_src, grow,
                              min_size, limit, raw_count, fell_back, prompts)
        return io.NodeOutput(masks, prompts, report)


# --------------------------------------------------------------------------- #
# image -> API payload
# --------------------------------------------------------------------------- #

def _tensor_to_rgb_bytes(image, frame_index: int, max_side: int,
                         image_format: str, jpeg_quality: int):
    """Encode one frame of an IMAGE tensor as PNG/JPEG bytes.

    Returns ``(raw, fmt, sent_w, sent_h, orig_w, orig_h, note)``. The sent size is
    reported separately from the original because the caller has to scale the
    model's coordinates back when they differ.

    Alpha is composited onto white rather than dropped. ``Image.convert("RGB")``
    simply throws the alpha channel away, and in a WebP/PNG cut-out the fully
    transparent pixels usually hold arbitrary colour -- which would show up as
    coloured fringe behind the subject and confuse the model.
    """
    if image is None:
        raise vlm_api.VLMError("nothing wired into the 'image' input")

    shape = tuple(int(v) for v in image.shape)
    if len(shape) == 4:
        batch, height, width, channels = shape
    elif len(shape) == 3:
        batch, (height, width, channels) = 1, shape
    else:
        raise vlm_api.VLMError(f"unsupported IMAGE tensor shape {shape}")

    index = max(0, min(int(frame_index), batch - 1))
    frame = image[index].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    if channels == 4:
        alpha = frame[..., 3:4].astype(np.float32) / 255.0
        blended = frame[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        frame = blended.round().astype(np.uint8)
    elif channels == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif channels > 4:
        frame = frame[..., :3]

    pil = PILImage.fromarray(frame, mode="RGB")
    orig_w, orig_h = pil.size

    notes = []
    if batch > 1:
        notes.append(f"sent frame {index} of {batch}")

    sent_w, sent_h = orig_w, orig_h
    limit = int(max_side)
    if limit >= 16 and max(orig_w, orig_h) > limit:
        ratio = limit / float(max(orig_w, orig_h))
        sent_w = max(1, int(round(orig_w * ratio)))
        sent_h = max(1, int(round(orig_h * ratio)))
        pil = pil.resize((sent_w, sent_h), PILImage.LANCZOS)
        notes.append(f"downscaled {orig_w}x{orig_h} -> {sent_w}x{sent_h}")

    fmt = (image_format or "png").strip().lower()
    buffer = BytesIO()
    if fmt in ("jpg", "jpeg"):
        pil.save(buffer, format="JPEG", quality=int(jpeg_quality), optimize=True)
        fmt = "jpeg"
    else:
        pil.save(buffer, format="PNG", optimize=True)
        fmt = "png"

    return buffer.getvalue(), fmt, sent_w, sent_h, orig_w, orig_h, "; ".join(notes)


class ObjectExtractVLMQuery(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ObjectExtract_VLMQuery",
            display_name="VLM Query (OpenAI compatible)",
            category="object_extract",
            description=(
                "Ask a vision model what is in the image and get the answer back already "
                "parsed. Sends the picture to any OpenAI-compatible /chat/completions "
                "endpoint (OpenAI, DashScope, Zhipu, Ark, SiliconFlow, local vLLM), reads "
                "the reply, and outputs both the raw text and what 'Object Prompts from "
                "VLM' would have produced from it -- a comma-separated "
                "'name:max_detections' prompt plus one pixel-space bounding box per "
                "object. Wire 'bboxes' into SAM3 Detect for precise masks, or into "
                "'BBoxes to Masks' for coarse ones. The API key is read from an "
                "environment variable, never baked into the graph. An empty or "
                "unparseable reply falls back to 'fallback_prompt' instead of failing."
            ),
            search_aliases=[
                "vlm", "vision api", "gpt-4o", "qwen vl", "openai", "chat completions",
                "bounding box", "bbox", "grounding", "detection prompt", "sam3",
                "image to json", "object extract", "multi object", "scene objects",
                "dashscope", "zhipu", "glm-4v", "ollama",
            ],
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Image handed to the vision model. A batch sends one frame "
                            "(see frame_index); alpha is composited onto white.",
                ),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default=vlm_api.DEFAULT_PROMPT,
                    tooltip="Instruction sent alongside the image. The default asks for a "
                            "JSON array of {name, bbox} -- keep that shape, since the "
                            "reply is parsed for exactly it.",
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="",
                    advanced=True,
                    tooltip="Optional system message. Some models follow the format rule "
                            "more reliably when it lives here instead of in the user turn.",
                ),
                io.String.Input(
                    "model",
                    default=vlm_api.DEFAULT_MODEL,
                    tooltip="Model id as the endpoint names it, e.g. qwen-vl-max, "
                            "qwen3-vl-plus, gpt-4o, glm-4v-plus, doubao-1.5-vision-pro.",
                ),
                io.String.Input(
                    "base_url",
                    default=vlm_api.DEFAULT_BASE_URL,
                    tooltip="Endpoint root; '/chat/completions' is appended if absent. "
                            "DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1 "
                            "| OpenAI: https://api.openai.com/v1 "
                            "| Zhipu: https://open.bigmodel.cn/api/paas/v4 "
                            "| local: http://127.0.0.1:11434/v1",
                ),
                io.String.Input(
                    "api_key_env",
                    default=vlm_api.DEFAULT_KEY_ENV,
                    advanced=True,
                    tooltip="Environment variable(s) holding the key. Space- or "
                            "comma-separated names are tried in order. Checked before the "
                            "api_key widget, so the graph never has to carry a secret.",
                ),
                io.String.Input(
                    "api_key",
                    default="",
                    advanced=True,
                    tooltip="Fallback only, used when no environment variable is set. "
                            "Careful: a key typed here is written into the workflow JSON.",
                ),
                io.Float.Input(
                    "temperature",
                    default=0.2, min=0.0, max=2.0, step=0.05,
                    tooltip="0 is the most repeatable reply; a little warmth helps some "
                            "models stop looping on the same object.",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=vlm_api.DEFAULT_MAX_TOKENS, min=16, max=32768,
                    tooltip="Reply budget. The JSON reply needs under 1000 tokens, but a "
                            "reasoning model spends this budget on its thinking first -- "
                            "at 1024 the thinking eats everything and no answer is ever "
                            "produced. A big ceiling is free: only generated tokens are "
                            "billed. If a reply still runs out mid-answer the node "
                            "automatically re-sends with more.",
                ),
                io.Combo.Input(
                    "on_error",
                    options=["raise", "fallback"],
                    default="raise",
                    tooltip="raise stops the queue with the HTTP status and vendor "
                            "message -- right for a wrong key or endpoint. fallback "
                            "carries on with fallback_prompt, which suits unattended runs.",
                ),
                io.String.Input(
                    "fallback_prompt",
                    default="",
                    advanced=True,
                    tooltip="Used as 'prompt' whenever nothing parseable came back, e.g. "
                            "'pineapple:2'. Leave empty to pass an empty prompt through "
                            "and let SAM3 run on the boxes alone.",
                ),
                io.Int.Input(
                    "timeout",
                    default=60, min=5, max=600,
                    tooltip="Seconds to wait per attempt. Reasoning models on big images "
                            "can take well over a minute.",
                ),
                io.Int.Input(
                    "retries",
                    default=2, min=0, max=5,
                    advanced=True,
                    tooltip="Extra attempts after a network fault or a 429/5xx. Client "
                            "errors (400/401/403/404) are never retried.",
                ),
                io.Int.Input(
                    "frame_index",
                    default=0, min=0, max=255,
                    advanced=True,
                    tooltip="Which frame of an image batch to send.",
                ),
                io.Int.Input(
                    "max_side",
                    default=0, min=0, max=4096, step=8,
                    advanced=True,
                    tooltip="Shrink the longest side to this many pixels before sending, "
                            "to cut tokens on large photos. 0 sends the original. Boxes "
                            "are scaled back to the original size automatically.",
                ),
                io.Combo.Input(
                    "image_format",
                    options=list(vlm_api.IMAGE_FORMATS),
                    default="png",
                    advanced=True,
                    tooltip="png is lossless and better for reading fine detail; jpeg is "
                            "much smaller on photographs.",
                ),
                io.Int.Input(
                    "jpeg_quality",
                    default=90, min=40, max=100,
                    advanced=True,
                    tooltip="Only used when image_format is jpeg.",
                ),
                io.Combo.Input(
                    "detail",
                    options=["auto", "low", "high"],
                    default="auto",
                    advanced=True,
                    tooltip="OpenAI image-detail hint. Sent only when not 'auto', since "
                            "some compatible servers reject the field.",
                ),
                io.String.Input(
                    "extra_body",
                    multiline=True,
                    default="",
                    advanced=True,
                    tooltip="JSON object merged into the request, for vendor-specific "
                            "switches. Overrides the fields above. Worth knowing:\n"
                            '  {"thinking": {"type": "disabled"}}   '
                            "DeepSeek V4 / Zhipu -- thinking is ON by default and is "
                            "pure overhead for this task\n"
                            '  {"enable_thinking": false}          Qwen via DashScope\n'
                            '  {"response_format": {"type": "json_object"}}   '
                            "force a strictly valid JSON reply",
                ),
                io.Combo.Input(
                    "coord_mode",
                    options=list(parse.COORD_MODES),
                    default="auto",
                    advanced=True,
                    tooltip="How to read the numbers. auto guesses from the sent image size "
                            "and the value range.",
                ),
                io.Combo.Input(
                    "box_format",
                    options=list(parse.BOX_FORMATS),
                    default="auto",
                    advanced=True,
                    tooltip="auto treats a bare 4-number list as x1,y1,x2,y2 and a dict "
                            "with width/height as x,y,w,h.",
                ),
                io.Int.Input(
                    "pad",
                    default=0, min=0, max=512,
                    advanced=True,
                    tooltip="Grow every box by this many pixels per side before clamping.",
                ),
                io.Int.Input(
                    "min_size",
                    default=4, min=1, max=4096,
                    advanced=True,
                    tooltip="Drop boxes narrower or shorter than this.",
                ),
                io.Boolean.Input(
                    "dedupe",
                    default=True,
                    advanced=True,
                    tooltip="Drop boxes nearly duplicating one already kept (IoU >= 0.95) "
                            "-- models repeat themselves on long replies.",
                ),
                io.Int.Input(
                    "count_cap",
                    default=20, min=0, max=200,
                    advanced=True,
                    tooltip="Upper bound for the ':N' max-detections value. 0 leaves "
                            "counts uncapped.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.BoundingBox.Output(display_name="bboxes"),
                io.String.Output(display_name="text"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    async def execute(cls, image, prompt=vlm_api.DEFAULT_PROMPT, system_prompt="",
                      model=vlm_api.DEFAULT_MODEL, base_url=vlm_api.DEFAULT_BASE_URL,
                      api_key_env=vlm_api.DEFAULT_KEY_ENV, api_key="",
                      temperature=0.2, max_tokens=vlm_api.DEFAULT_MAX_TOKENS,
                      on_error="raise",
                      fallback_prompt="", timeout=60, retries=2, frame_index=0,
                      max_side=0, image_format="png", jpeg_quality=90,
                      detail="auto", extra_body="", coord_mode="auto",
                      box_format="auto", pad=0, min_size=4, dedupe=True,
                      count_cap=20) -> io.NodeOutput:
        # Declared async on purpose: ComfyUI runs a synchronous node body on the
        # event-loop thread, so a blocking HTTP call would freeze the whole server
        # (no progress bar, no cancel) for as long as the model takes. The request
        # runs in a worker thread via to_thread instead.
        raw, fmt, sent_w, sent_h, orig_w, orig_h, image_note = _tensor_to_rgb_bytes(
            image, frame_index, max_side, image_format, jpeg_quality
        )

        transport: list[str] = []
        extra, extra_warning = vlm_api.parse_extra_body(extra_body)
        if extra_warning:
            transport.append(f"extra  : {extra_warning}")

        text = ""
        result = None
        key_note = "not needed"
        try:
            key_value, key_note = vlm_api.resolve_api_key(api_key_env, api_key)
            payload = vlm_api.build_payload(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                images=[vlm_api.encode_data_url(raw, fmt)],
                temperature=temperature,
                max_tokens=max_tokens,
                detail=detail,
                extra=extra,
            )
            result = await asyncio.to_thread(
                vlm_api.chat, payload, key_value, base_url, timeout, retries
            )
            text = result.text
        except vlm_api.VLMError as exc:
            # Raised before the request could even be built: no key, bad base_url.
            if on_error == "raise":
                raise RuntimeError(str(exc)) from exc
            transport.append(f"error  : {exc}")
        else:
            if not result.ok and on_error == "raise":
                # The request went out and came back wrong -- a wrong key, a wrong
                # host, or the service being down. That is a configuration bug, and
                # carrying on would silently repaint with fallback_prompt, leaving
                # the user to wonder why every image came out wrong.
                raise RuntimeError(f"{result.error} (POST {result.endpoint})")

        if result is not None and result.ok:
            tokens = ""
            if result.total_tokens is not None:
                tokens = (f"   tokens {result.prompt_tokens}+{result.completion_tokens}"
                          f"={result.total_tokens}")
            finish = f"   finish={result.finish_reason}" if result.finish_reason else ""
            transport.append(f"request: POST {result.endpoint}")
            transport.append(f"model  : {result.model}   key: {key_note}   "
                             f"image: {sent_w}x{sent_h} {fmt} "
                             f"{vlm_api.human_bytes(len(raw))}")
            if image_note:
                transport.append(f"image  : {image_note}")
            transport.append(f"reply  : HTTP {result.status} in {result.elapsed:.2f}s "
                             f"({result.attempts} attempt(s)){tokens}{finish}")
            if result.escalated:
                # Said out loud because it is a paid round-trip the user can make
                # unnecessary: turning the model's reasoning off removes it.
                ladder = " -> ".join(str(b) for b in result.budgets)
                transport.append(f"retry  : max_tokens {ladder} -- the first budget went "
                                 f"into reasoning and never reached the answer")
            if result.reasoning:
                # Visible on purpose: a model that thinks for 900 tokens and then
                # answers in 20 is worth knowing about (it costs, and it is slow),
                # and a model that thinks and then says *nothing* is the whole
                # reason the answer came back empty.
                if text.strip():
                    transport.append(f"think  : {len(result.reasoning)} chars of reasoning "
                                     f"ignored, final answer only")
                else:
                    last = result.budgets[-1] if result.budgets else max_tokens
                    transport.append(f"think  : {len(result.reasoning)} chars of reasoning, "
                                     f"but no answer even at max_tokens={last} -- turn this "
                                     f"model's thinking off (see extra_body)")
            if result.truncated and text.strip():
                transport.append("note   : the reply hit max_tokens and was cut off; some "
                                 "objects may be missing")
        elif result is not None:
            transport.append(f"request: POST {result.endpoint}")
            transport.append(f"reply  : FAILED after {result.attempts} attempt(s) in "
                             f"{result.elapsed:.2f}s")
            transport.append(f"error  : {result.error}")

        boxes, names, space = parse.parse_boxes(
            text, sent_w, sent_h, coord_mode, box_format, pad, min_size
        )

        if sent_w != orig_w or sent_h != orig_h:
            sx, sy = orig_w / float(sent_w), orig_h / float(sent_h)
            boxes, names = parse.rescale_boxes(boxes, names, sx, sy, orig_w, orig_h)
            if boxes:
                transport.append(f"coords : scaled by {sx:.4f}x{sy:.4f} back to the "
                                 f"original {orig_w}x{orig_h}")

        raw_count = len(boxes)
        if dedupe:
            boxes, names = parse.dedupe_boxes(boxes, names)
        dropped = raw_count - len(boxes)

        pairs = parse.count_names(names) if boxes else []
        if not pairs:
            pairs = parse.parse_prompts(text)
        parsed_prompt = parse.build_prompt(pairs, count_cap)

        prompt_out = parsed_prompt
        if not prompt_out and fallback_prompt.strip():
            prompt_out = fallback_prompt.strip()
            transport.append("fallback: nothing parseable -> fallback_prompt used")

        parse_report = _report(text, prompt_out, boxes, names, space, orig_w, orig_h,
                               True, dropped)
        if text.strip() and not boxes:
            # Nothing located: the raw reply is the only thing that explains why.
            transport.append(f"raw    : {vlm_api.short(text, 400)}")

        labelled = [dict(box, label=name) for box, name in zip(boxes, names)]
        return io.NodeOutput(prompt_out, labelled, text, "\n".join(transport) + "\n" + parse_report)
