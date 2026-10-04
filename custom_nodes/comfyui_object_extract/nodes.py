"""ComfyUI nodes for pulling *several objects out of a single scene*.

A VLM looks at the image and names what it sees; these two nodes turn that reply
into something the rest of the graph can act on, one pass per object:

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
"""

from __future__ import annotations

import torch

import comfy.model_management
from comfy_api.latest import io

from . import parse

__all__ = ["ObjectExtractPromptsFromVLM", "ObjectExtractBBoxesToMasks"]

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
