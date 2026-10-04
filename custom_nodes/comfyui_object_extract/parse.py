"""Pure-Python helpers turning free-form VLM output into SAM3 prompts and boxes.

No ComfyUI imports live here on purpose. Parsing is the fragile part of this node
(a different model, prompt or image changes the reply format), so it stays
importable and unit-testable without a running ComfyUI.
"""

from __future__ import annotations

import json
import re

__all__ = [
    "COORD_MODES",
    "BOX_FORMATS",
    "normalize_name",
    "build_prompt",
    "parse_prompts",
    "parse_boxes",
    "dedupe_boxes",
    "normalize_boxes",
    "fit_boxes",
    "render_prompt",
    "try_load_json",
    "extract_json_blobs",
    "guess_space",
    "count_names",
    "DEFAULT_SIZE",
]

DEFAULT_SIZE = 1024

COORD_MODES = ["auto", "pixels", "normalized_1000", "normalized_1"]
BOX_FORMATS = ["auto", "xyxy", "xywh"]

# --------------------------------------------------------------------------- #
# text prompts:  "apple:2, banana:1"
# --------------------------------------------------------------------------- #

# SAM3 splits its text prompt on plain commas and then requires *every* part to
# match ^(.+?)\s*:\s*([\d.]+)$  (comfy/text_encoders/sam3_clip.py::_parse_prompts).
# A single bad part silently collapses the whole prompt into one max_det=1 entry,
# so names must not contain commas -- and must stay short, since each name is
# capped at 32 CLIP tokens. Sentence punctuation is excluded too, otherwise
# prose like "There are 2 apples. apple: 2" produces a garbage name.
_PAIR_RE = re.compile(
    r"([^,:;|，：；、.。!?！？\r\n\[\]{}()\"'`]{1,64}?)\s*[:：]\s*(\d+(?:\.\d+)?)"
)

# Characters that break the comma split, cost token budget, or read as noise.
_BAD_NAME_RE = re.compile(r"[,:;|，：；、\[\]{}()\"'`\\\r\n\t]+")
_WS_RE = re.compile(r"\s+")

_NAME_KEYS = ("name", "label", "class", "category", "object", "obj", "text", "tag")
_COUNT_KEYS = ("count", "quantity", "num", "number", "amount", "instances")
_COORD_LIST_KEYS = (
    "bbox", "box", "bbox_2d", "bbox2d", "bbox_2d_list", "coordinates",
    "coords", "rect", "rectangle", "region", "location",
)

# Structural key names that must never be mistaken for an object label when a
# truncated payload forces the regex fallback. Without this, the label of
# {"name": "flower", "bbox": [...]} comes back as "bbox".
_JSON_KEY_WORDS = frozenset(
    key.lower() for key in _NAME_KEYS + _COUNT_KEYS + _COORD_LIST_KEYS
) | {"x", "y", "w", "h", "x1", "y1", "x2", "y2", "width", "height",
     "cx", "cy", "id", "index", "type", "desc", "description", "score",
     "confidence", "objects", "object", "items", "results", "detections"}


def _looks_like_key(word: str) -> bool:
    return word.strip().strip("\"'").lower() in _JSON_KEY_WORDS


def normalize_name(name: str | None) -> str:
    """Make ``name`` safe to use as one SAM3 prompt entry."""
    if name is None:
        return ""
    text = _BAD_NAME_RE.sub(" ", str(name))
    text = _WS_RE.sub(" ", text).strip(" -_.")
    # Keep inside the 32-token budget with room to spare for tokenisation noise.
    return text[:48].strip(" -_.")


def count_names(names) -> list[tuple[str, int]]:
    """Fold an iterable of names into ordered ``(name, occurrences)`` pairs."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for raw in names:
        key = normalize_name(raw)
        if not key:
            continue
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1
    return [(k, counts[k]) for k in order]


def build_prompt(pairs, count_cap: int = 20) -> str:
    """Render ordered ``(name, count)`` pairs into SAM3's prompt grammar."""
    parts = []
    for name, n in pairs:
        key = normalize_name(name)
        if not key:
            continue
        k = max(1, int(round(float(n))))
        if count_cap and count_cap > 0:
            k = min(k, count_cap)
        parts.append(f"{key}:{k}")
    return ", ".join(parts)


def _merge_pairs(pairs) -> list[tuple[str, int]]:
    """Fold duplicate names together, preserving first-seen order."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for name, n in pairs:
        key = normalize_name(name)
        if not key:
            continue
        if key not in counts:
            counts[key] = 0
            order.append(key)
        try:
            counts[key] += max(1, int(round(float(n))))
        except (TypeError, ValueError):
            counts[key] += 1
    return [(k, counts[k]) for k in order]


def parse_prompts(text: str) -> list[tuple[str, int]]:
    """Extract ordered ``(name, count)`` pairs from a free-form reply.

    Counts for the same name are summed: an object listed twice is still one
    SAM3 entry, just with a larger max-detection budget.
    """
    if not text or not str(text).strip():
        return []
    text = str(text)

    pairs = []
    data = try_load_json(text)
    if data is not None:
        pairs = _prompts_from_json(data)
    if not pairs:
        pairs = _prompts_from_regex(text)
    return _merge_pairs(pairs)


def _prompts_from_json(data) -> list[tuple[str, int]]:
    """Pull ``(name, count)`` pairs out of a parsed JSON payload."""
    collected: list[tuple[str, int]] = []

    def walk(node):
        if isinstance(node, dict):
            name = None
            for key in _NAME_KEYS:
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    name = value.strip()
                    break
            if name is not None:
                n = 1
                for key in _COUNT_KEYS:
                    value = _as_number(node.get(key))
                    if value is not None and value > 0:
                        n = int(round(value))
                        break
                collected.append((name, n))
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(data)
    return collected


def _prompts_from_regex(text: str) -> list[tuple[str, int]]:
    """Fallback for prose replies such as ``apple: 2, banana: 1``."""
    pairs: list[tuple[str, int]] = []
    for raw_name, raw_n in _PAIR_RE.findall(text):
        name = normalize_name(raw_name)
        if not name:
            continue
        try:
            n = float(raw_n)
        except (TypeError, ValueError):
            continue
        if n > 0:
            pairs.append((name, int(round(n))))
    return pairs


# --------------------------------------------------------------------------- #
# boxes
# --------------------------------------------------------------------------- #

# yanked from a reply that is not valid JSON at all, e.g. "apple: [238, 35, 767, 945]"
_QUAD_RE = re.compile(
    r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]"
)
_QUOTED_RE = re.compile(r"[\"']([^\"'\n]{1,40})[\"']")
_WORD_RE = re.compile(r"([A-Za-z\u4e00-\u9fa5][A-Za-z\u4e00-\u9fa5 \-_]{1,40})")

_OPEN_BRACKETS = {"[": "]", "{": "}"}


def extract_json_blobs(text: str) -> list[str]:
    """Return every balanced ``[...]`` / ``{...}`` run of text, outermost first."""
    blobs: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] not in _OPEN_BRACKETS:
            i += 1
            continue
        depth = 0
        in_str = False
        escaped = False
        end = None
        for j in range(i, n):
            ch = text[j]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in _OPEN_BRACKETS:
                depth += 1
            elif ch in "]}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            break
        blobs.append(text[i:end + 1])
        i = end + 1
    return blobs


_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")
_BARE_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:")


def _strip_trailing_commas(blob: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", blob)


def _quote_bare_keys(blob: str) -> str:
    """Turn ``{name: ...}`` -- legal JS, illegal JSON -- into valid JSON."""
    return _BARE_KEY_RE.sub(r'\1"\2":', blob)


# Tried in order; the untouched blob comes first so a well-formed reply wins.
_JSON_REPAIRS = (
    lambda s: s,
    _strip_trailing_commas,
    _quote_bare_keys,
    lambda s: _quote_bare_keys(_strip_trailing_commas(s)),
)


def try_load_json(text: str):
    """Best-effort JSON recovery from a model reply; ``None`` when hopeless.

    Handles markdown fences, prose wrapped around the payload, trailing commas
    and unquoted keys -- the sloppiness small VLMs actually fall into.
    """
    if not text:
        return None
    candidates = [m.group(1) for m in re.finditer(r"```[a-zA-Z0-9_+-]*\s*(.*?)```", text, re.S)]
    candidates.append(text)
    for chunk in candidates:
        for blob in extract_json_blobs(chunk):
            for repair in _JSON_REPAIRS:
                try:
                    parsed = json.loads(repair(blob))
                except (ValueError, TypeError):
                    continue
                if isinstance(parsed, (list, dict)):
                    return parsed
    return None


def _as_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_quad(value) -> tuple[float, float, float, float] | None:
    """Coerce a 4-element sequence of numbers (or numeric strings) into a tuple."""
    if isinstance(value, (list, tuple)) and len(value) == 4:
        nums = [_as_number(v) for v in value]
        if all(v is not None for v in nums):
            return (nums[0], nums[1], nums[2], nums[3])
    if isinstance(value, dict):
        flat = [_as_number(value.get(k)) for k in ("x", "y", "width", "height")]
        if all(v is not None for v in flat):
            return (flat[0], flat[1], flat[2], flat[3])
        flat = [_as_number(value.get(k)) for k in ("x1", "y1", "x2", "y2")]
        if all(v is not None for v in flat):
            return (flat[0], flat[1], flat[2], flat[3])
    return None


def _name_from_dict(node: dict) -> str:
    for key in _NAME_KEYS:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _entry_from_dict(node: dict):
    """Return ``(name, quad, kind)`` when ``node`` *is* one box entry."""
    name = _name_from_dict(node)
    for key in _COORD_LIST_KEYS:
        quad = _as_quad(node.get(key))
        if quad is not None:
            return name, quad, "xyxy"
    if all(k in node for k in ("x", "y", "width", "height")):
        quad = _as_quad({k: node[k] for k in ("x", "y", "width", "height")})
        if quad is not None:
            return name, quad, "xywh"
    if all(k in node for k in ("x1", "y1", "x2", "y2")):
        quad = _as_quad({k: node[k] for k in ("x1", "y1", "x2", "y2")})
        if quad is not None:
            return name, quad, "xyxy"
    return None


def _iter_box_entries(node):
    """Recursively yield ``(name, quad, kind)`` from an arbitrary JSON payload."""
    if isinstance(node, dict):
        entry = _entry_from_dict(node)
        if entry is not None:
            yield entry
            return
        for value in node.values():
            yield from _iter_box_entries(value)
    elif isinstance(node, (list, tuple)):
        quad = _as_quad(node)
        if quad is not None:
            yield "", quad, "xyxy"
            return
        for value in node:
            yield from _iter_box_entries(value)


def _label_from_prefix(prefix: str) -> str:
    """Pick the object label sitting just before a coordinate quad.

    Structural JSON key names are skipped so that a truncated payload such as
    ``{"name": "flower", "bbox": [550, 438, 624, 546]}`` yields ``flower`` and
    not ``bbox``.
    """
    quoted = [q for q in _QUOTED_RE.findall(prefix) if not _looks_like_key(q)]
    if quoted:
        return quoted[-1].strip()
    words = [w for w in _WORD_RE.findall(prefix) if not _looks_like_key(w)]
    return words[-1].strip() if words else ""


def _regex_boxes(text: str) -> list[tuple[str, tuple[float, float, float, float], str]]:
    """Last-resort scan: pair every ``[a, b, c, d]`` with the nearest label.

    This is the main path for replies truncated mid-JSON -- a VLM that overruns
    its token budget still yields usable boxes for everything it managed to emit.
    """
    entries = []
    for match in _QUAD_RE.finditer(text):
        quad = tuple(float(g) for g in match.groups())
        prefix = text[max(0, match.start() - 80): match.start()]
        entries.append((_label_from_prefix(prefix), quad, "xyxy"))
    return entries


def _collect_raw_boxes(text: str):
    if not text or not str(text).strip():
        return []
    text = str(text)
    data = try_load_json(text)
    if data is not None:
        found = list(_iter_box_entries(data))
        if found:
            # A bare [[0,0,10,10]] payload parses as JSON but carries no labels,
            # so fall back to the text scan when *it* can recover names.
            if any(name for name, _, _ in found):
                return found
            labelled = _regex_boxes(text)
            if any(name for name, _, _ in labelled):
                return labelled
            return found
    return _regex_boxes(text)


def guess_space(quads, width: int, height: int) -> str:
    """Infer whether coordinates are pixels, 0-1000 or 0-1 normalized.

    Genuinely ambiguous cases exist (a 1024x1024 image is also a valid 0-1000
    space), so the tie is broken towards pixels: VLMs reporting on a small image
    almost always report real pixel values, while the 0-1000 fallback shows up
    on large inputs where pixel coordinates would blow past 1000.
    """
    values = [v for quad in quads for v in quad]
    if not values:
        return "pixels"
    hi = max(values)
    if hi <= 1.5:
        return "normalized_1"
    longest = max(width, height)
    if hi <= 1000 and longest > 1250:
        # Every value sits inside 0-1000 but the image is far bigger: normalized.
        return "normalized_1000"
    if hi > longest * 1.5:
        # Way past the image edge -- cannot be a pixel value.
        return "normalized_1000"
    return "pixels"


def _to_pixels(quad, kind, space, width, height, pad):
    a, b, c, d = (float(v) for v in quad)

    if space == "normalized_1000":
        sx, sy = width / 1000.0, height / 1000.0
        a, c = a * sx, c * sx
        b, d = b * sy, d * sy
    elif space == "normalized_1":
        a, c = a * width, c * width
        b, d = b * height, d * height

    if kind == "xywh":
        x, y, w, h = a, b, c, d
    else:
        x, y, w, h = a, b, c - a, d - b

    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h

    x0 = max(0, min(int(round(x)) - pad, width))
    y0 = max(0, min(int(round(y)) - pad, height))
    x1 = max(0, min(int(round(x + w)) + pad, width))
    y1 = max(0, min(int(round(y + h)) + pad, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _iou(a: dict, b: dict) -> float:
    """Intersection-over-union of two pixel-space boxes."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def dedupe_boxes(boxes, names, iou_threshold: float = 0.95):
    """Drop boxes near-identical to one already kept.

    VLMs fall into repetition loops on long replies -- a 47-flower image comes
    back as the same six flower boxes emitted over and over. Left alone, that
    inflates every ``:N`` count and makes the prompt meaningless. The threshold
    is deliberately high so genuinely overlapping objects survive.
    """
    kept_boxes: list[dict] = []
    kept_names: list[str] = []
    for box, name in zip(boxes, names):
        if any(_iou(box, kept) >= iou_threshold for kept in kept_boxes):
            continue
        kept_boxes.append(box)
        kept_names.append(name)
    return kept_boxes, kept_names


def parse_boxes(
    text: str,
    width: int = DEFAULT_SIZE,
    height: int = DEFAULT_SIZE,
    coord_mode: str = "auto",
    box_format: str = "auto",
    pad: int = 0,
    min_size: int = 4,
) -> tuple[list[dict], list[str], str]:
    """Convert a reply into ComfyUI ``BOUNDING_BOX`` dicts.

    Returns ``(boxes, names, space)`` where ``boxes`` are pixel-space
    ``{"x", "y", "width", "height"}`` dicts ready for ``SAM3_Detect.bboxes``.
    ``names`` mirrors ``boxes`` so the caller can build a matching prompt.
    An empty list is a *valid* result: SAM3 then ignores the port entirely.
    """
    raw = _collect_raw_boxes(text)
    if not raw:
        return [], [], "pixels"

    quads = [quad for _, quad, _ in raw]
    space = coord_mode if coord_mode in ("pixels", "normalized_1000", "normalized_1") \
        else guess_space(quads, width, height)

    boxes: list[dict] = []
    names: list[str] = []
    for name, quad, kind in raw:
        if box_format != "auto":
            kind = box_format
        box = _to_pixels(quad, kind, space, width, height, pad)
        if box is None:
            continue
        if box["width"] < min_size or box["height"] < min_size:
            continue
        boxes.append(box)
        names.append(name)
    return boxes, names, space


# --------------------------------------------------------------------------- #
# boxes -> masks
# --------------------------------------------------------------------------- #

_SE_KEYS = ("x1", "y1", "x2", "y2")
_LABEL_KEYS = ("label", "name", "text", "class", "category")


def _label_of(node: dict) -> str:
    for key in _LABEL_KEYS:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _as_box(node) -> dict | None:
    """Read one box out of a dict, accepting either xywh or x1y1x2y2 keys.

    A ``label`` (if the payload carries one under any of the usual key names) is
    copied through untouched, so it can still be used to caption the crop two
    nodes later. Extra keys are ignored -- SAM3 and friends only read the four
    coordinate fields.
    """
    if not isinstance(node, dict):
        return None

    label = _label_of(node)

    nums = [_as_number(node.get(k)) for k in ("x", "y", "width", "height")]
    if all(v is not None for v in nums):
        return {"x": nums[0], "y": nums[1], "width": nums[2], "height": nums[3],
                "label": label}

    nums = [_as_number(node.get(k)) for k in _SE_KEYS]
    if all(v is not None for v in nums):
        return {"x": nums[0], "y": nums[1],
                "width": nums[2] - nums[0], "height": nums[3] - nums[1],
                "label": label}
    return None


def normalize_boxes(bboxes) -> tuple[list[dict], str]:
    """Flatten any ``BOUNDING_BOX`` payload into pixel-space box dicts.

    ComfyUI nodes disagree on the container shape -- ``PrimitiveBoundingBox``
    returns a bare dict, ``DrawBBoxes`` reads a flat ``[box, box]``, and boxes
    riding an image batch arrive as frames ``[[box, box], [box, box]]``. All
    three are accepted here so the caller never has to care where they came
    from. For a multi-frame payload only the first non-empty frame is used,
    because the masks are painted onto a single still image.

    Returns ``(boxes, note)`` where ``note`` names the shape that was seen --
    useful in the node report when a wiring mistake yields zero boxes.
    """
    if bboxes is None:
        return [], "none"
    if isinstance(bboxes, dict):
        box = _as_box(bboxes)
        return ([box] if box else []), "single dict"

    if not isinstance(bboxes, (list, tuple)) or len(bboxes) == 0:
        return [], "empty"

    if isinstance(bboxes[0], dict):
        boxes = [b for b in (_as_box(n) for n in bboxes) if b is not None]
        return boxes, f"flat list ({len(boxes)})"

    # Frames: take the first frame that actually carries a box.
    for idx, frame in enumerate(bboxes):
        if not isinstance(frame, (list, tuple)) or not frame:
            continue
        boxes = [b for b in (_as_box(n) for n in frame) if b is not None]
        if boxes:
            note = f"frame {idx}/{len(bboxes)} ({len(boxes)})"
            return boxes, note
    return [], f"{len(bboxes)} empty frame(s)"


def fit_boxes(boxes, width: int, height: int, grow: int = 0, min_size: int = 1,
              limit: int = 0) -> list[dict]:
    """Clamp boxes to the canvas and optionally grow (or shrink) each one.

    ``grow`` is applied symmetrically and may be negative, which trims a few
    pixels off every side -- handy when the VLM's box is slightly generous and
    you do not want its border baked into the coarse mask. Boxes that collapse
    to nothing, or fall under ``min_size``, are dropped.
    """
    out: list[dict] = []
    for box in boxes:
        x0 = int(round(box["x"])) - grow
        y0 = int(round(box["y"])) - grow
        x1 = int(round(box["x"] + box["width"])) + grow
        y1 = int(round(box["y"] + box["height"])) + grow
        x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
        y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
        if x1 - x0 < min_size or y1 - y0 < min_size:
            continue
        out.append({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
                    "label": box.get("label", "")})
        if limit and len(out) >= limit:
            break
    return out


def render_prompt(template: str, label: str = "") -> str:
    """Substitute a box's label into the user's prompt template.

    ``{name}`` is the placeholder; a box that arrived without a label gets the
    neutral phrase ``main object`` instead of a hole in the sentence, so the
    template still reads as valid English. A template with no placeholder is
    passed through unchanged, which is how you get one shared prompt for every
    box without rewiring anything.
    """
    tpl = (template or "").strip()
    if not tpl:
        return ""
    name = (label or "").strip() or "main object"
    return tpl.replace("{name}", name)
