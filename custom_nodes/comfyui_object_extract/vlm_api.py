"""Minimal OpenAI-compatible vision client, deliberately free of ComfyUI imports.

Everything here is standard library only (``urllib`` + ``base64`` + ``json``), so
the request/response handling can be exercised against a mock server without a
running ComfyUI, and the node package gains no new dependency.

The wire format targeted is the one every vendor converged on::

    POST {base_url}/chat/completions
    {"model": ..., "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "..."}]}],
     "temperature": 0.2, "max_tokens": 1024}

That covers OpenAI, Alibaba DashScope (``/compatible-mode/v1``), Zhipu, Moonshot,
Volcengine Ark, DeepSeek, SiliconFlow, and local vLLM / LM Studio / Ollama's
compatibility endpoint -- only ``base_url`` and ``model`` change.

``chat()`` never raises for transport reasons: it returns a :class:`ChatResult`
with ``error`` set, because the caller has to build a report either way and a
raise would throw away the status code and body that explain *what* went wrong.

Reasoning models are handled too: :attr:`ChatResult.text` carries the final
answer only -- an inline ``think`` block or a separate ``reasoning_content``
field is set aside into :attr:`ChatResult.reasoning` rather than fed to the
parser. See :func:`strip_thinking`.

They also need a second guard, because the scratchpad is paid for out of the
same ``max_tokens`` budget as the answer: a reasoning model with too small a
ceiling thinks until it runs out and *never gets to answer*. The call comes
back HTTP 200, fully billed, with an empty reply. :func:`chat` notices that
shape and sends the request again with a larger budget (see :func:`_starved`).

Note the deliberate absence of ``from __future__ import annotations``: with it,
the string annotations on :class:`ChatResult` make ``@dataclass`` resolve types
through ``sys.modules[cls.__module__]``, which fails (``NoneType has no
__dict__``) whenever this file is loaded standalone via ``spec_from_file_location``
-- exactly how the offline test bench loads it. The PEP 604 unions used below need
Python 3.10, which is already this package's floor.
"""

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

__all__ = [
    "ChatResult",
    "DEFAULT_BASE_URL",
    "DEFAULT_KEY_ENV",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT",
    "IMAGE_FORMATS",
    "build_payload",
    "chat",
    "encode_data_url",
    "extract_text",
    "extract_thinking",
    "human_bytes",
    "normalize_endpoint",
    "parse_extra_body",
    "resolve_api_key",
    "short",
    "split_thinking",
    "strip_thinking",
]

# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #

# The instruction handed to the vision model. Wording is load-bearing: the reply
# goes through parse.py, which wants a JSON array of {name, bbox}, and SAM3 wants
# each name to survive its 32-CLIP-token budget -- hence "1-3 common words".
DEFAULT_PROMPT = (
    "List every distinct object in this image that could be cut out and repainted.\n"
    "Reply with JSON only, no prose, no markdown:\n"
    '[{"name": "object", "bbox": [x1, y1, x2, y2]}]\n'
    "- name: singular lowercase English noun, 1-3 common words\n"
    "- bbox: [left, top, right, bottom] in pixels of this image\n"
    "- list each object once; never repeat a box\n"
    "- cover every object you can see; stop when you are done\n"
    "Reply:"
)

# DashScope's OpenAI-compatible endpoint is the default because the rest of this
# workflow already runs on Qwen weights; swap both values for another vendor.
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-max"
DEFAULT_KEY_ENV = "DASHSCOPE_API_KEY"

# Reply budget. The JSON this node asks for needs well under 1000 tokens -- but
# a reasoning model spends the budget on its scratchpad *first*, and v4-style
# models have reasoning on by default. At 1024 the scratchpad eats everything
# and the model never gets to answer: HTTP 200, tokens billed, empty reply.
# A generous ceiling costs nothing, because only generated tokens are billed.
DEFAULT_MAX_TOKENS = 8192

IMAGE_FORMATS = ["png", "jpeg"]

_MIME = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg",
         "webp": "image/webp"}

# Transient by nature -- worth another attempt. Everything else (400 bad request,
# 401 bad key, 403, 404 wrong path) is a configuration bug that retrying cannot
# fix and that the user wants to see immediately.
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_MAX_BACKOFF = 8.0

# Ladder for a reply that was starved of budget (see :func:`_starved`). The
# first jump is deliberately larger than a doubling: the budgets that cause the
# problem are small, and doubling them just lands in the same trap one round
# later. The ceiling is 8192 because that is the largest output limit the
# mainstream vendors all document -- asking for more risks a 400 on the retry,
# which would throw away the diagnosis the user needs.
_ESCALATE_FLOOR = 4096
_TOKEN_CAP = 8192


@dataclass
class ChatResult:
    """Everything worth reporting about one API round-trip."""

    text: str = ""
    reasoning: str = ""
    status: int = 0
    elapsed: float = 0.0
    attempts: int = 0
    model: str = ""
    endpoint: str = ""
    request_bytes: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str = ""
    error: str = ""

    # The max_tokens value used on each attempt, in order. One entry means the
    # first try was good enough; more than one means the budget had to grow
    # because the model was spending it all on thinking.
    budgets: tuple = ()

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def truncated(self) -> bool:
        """The vendor cut generation short: ``finish_reason == "length"``."""
        return self.finish_reason == "length"

    @property
    def escalated(self) -> bool:
        """The request was re-sent with a larger budget. See :func:`chat`."""
        return len(self.budgets) > 1


class VLMError(Exception):
    """Raised only for problems the caller can fix in the node itself."""


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #

def short(value, limit: int = 300) -> str:
    """Collapse whitespace and clip, for stuffing into a report line."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def human_bytes(count: int) -> str:
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / (1024 * 1024):.2f} MB"


# --------------------------------------------------------------------------- #
# reasoning models
# --------------------------------------------------------------------------- #

# Tags a thinking model wraps its scratchpad in. Qwen3 emits the plain "think"
# tag; several gateways relay "thinking" or "reasoning"; a few use "thought" or
# "analysis". Matched case-insensitively and across newlines: the block is prose.
_THINK_NAMES = ("think", "thinking", "reasoning", "thought", "analysis")
_THINK_ALT = "|".join(_THINK_NAMES)
_THINK_BLOCK = re.compile(
    rf"<\s*(?:{_THINK_ALT})\s*>.*?<\s*/\s*(?:{_THINK_ALT})\s*>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_OPEN = re.compile(rf"<\s*(?:{_THINK_ALT})\s*>", re.IGNORECASE)
_THINK_CLOSE = re.compile(rf"<\s*/\s*(?:{_THINK_ALT})\s*>", re.IGNORECASE)


def split_thinking(text: str) -> tuple[str, str]:
    """Split a reply into ``(final answer, scratchpad)``.

    Two shapes have to be handled, and a lone tag at either end means the reply
    was truncated mid-thought -- everything on that side of the tag is
    scratchpad too. Missing that case is how half a sentence of musing ends up
    in front of the parser.
    """
    if not text:
        return "", ""

    kept: list[str] = []
    removed: list[str] = []
    cursor = 0
    for match in _THINK_BLOCK.finditer(text):
        kept.append(text[cursor:match.start()])
        removed.append(match.group(0))
        cursor = match.end()
    kept.append(text[cursor:])
    body = "".join(kept)

    closers = list(_THINK_CLOSE.finditer(body))  # a lone closing tag
    if closers:
        removed.append(body[: closers[-1].end()])
        body = body[closers[-1].end():]

    opener = _THINK_OPEN.search(body)  # a lone opening tag: ran out of max_tokens
    if opener:
        removed.append(body[opener.start():])
        body = body[: opener.start()]

    return body.strip(), "".join(removed).strip()


def strip_thinking(text: str) -> str:
    """Keep only what the model actually answered.

    A reasoning model's scratchpad is worse than noise in front of the box
    parser: it is fluent prose full of numbers ("the pineapple spans roughly 100
    to 400 pixels"), and every one of those numbers parses as a coordinate.
    """
    return split_thinking(text)[0]


# --------------------------------------------------------------------------- #
# request
# --------------------------------------------------------------------------- #

def normalize_endpoint(base_url: str) -> str:
    """Turn a base URL into a full ``chat/completions`` URL.

    Accepts either style, because vendors document both::
        https://host/v1                     -> https://host/v1/chat/completions
        https://host/v1/chat/completions    -> unchanged
    """
    url = (base_url or "").strip()
    if not url:
        raise VLMError("base_url is empty")
    if not url.lower().startswith(("http://", "https://")):
        raise VLMError(f"base_url must start with http:// or https:// (got {url!r})")
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def resolve_api_key(env_names: str, inline: str = "") -> tuple[str, str]:
    """Find the API key, preferring the environment over the widget.

    The order is deliberate. A key typed into a widget is serialised into the
    workflow JSON, so it leaks the moment the graph is shared or committed; an
    environment variable never does. The widget therefore only acts as a
    fallback, which keeps the safe path the default one.

    Returns ``(key, source)`` where ``source`` names where it came from, so the
    node report can say ``env OPENAI_API_KEY`` or ``widget``.
    """
    for name in (env_names or "").replace(",", " ").split():
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip(), f"env {name}"

    if inline and inline.strip():
        return inline.strip(), "widget"

    names = " ".join((env_names or "").split()) or "(none configured)"
    raise VLMError(f"no API key: environment variable(s) {names} are unset and the "
                   f"api_key widget is empty")


def encode_data_url(raw: bytes, image_format: str = "png") -> str:
    """Wrap image bytes as a ``data:`` URL -- the only image form the API needs."""
    mime = _MIME.get((image_format or "png").strip().lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def parse_extra_body(raw: str) -> tuple[dict, str]:
    """Read an ``extra_body`` widget value: a JSON object merged into the request.

    Returns ``(fields, warning)``. A malformed value is *not* fatal: the request
    is still worth sending without it, and the warning rides back to the node
    report so the mistake is visible instead of silent.
    """
    text = (raw or "").strip()
    if not text:
        return {}, ""
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError) as exc:
        return {}, f"extra_body ignored, not valid JSON ({exc})"
    if not isinstance(parsed, dict):
        return {}, f"extra_body ignored, must be a JSON object (got {type(parsed).__name__})"
    return parsed, ""


def build_payload(model: str, prompt: str, images=(), system_prompt: str = "",
                  temperature: float = 0.2, max_tokens: int = DEFAULT_MAX_TOKENS,
                  detail: str = "auto", extra: dict | None = None) -> dict:
    """Assemble the chat-completions body.

    Images come before the text: that is the ordering Qwen-VL and most grounding
    models are trained on, and putting the instruction last reads as "about that
    image, do this".

    ``detail`` is emitted only when explicitly set, because the field is an
    OpenAI extension that stricter compatible servers reject outright.
    """
    content: list[dict] = []
    for url in images:
        part: dict = {"type": "image_url", "image_url": {"url": url}}
        if detail and detail != "auto":
            part["image_url"]["detail"] = detail
        content.append(part)
    content.append({"type": "text", "text": prompt})

    messages: list[dict] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": content})

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra:
        payload.update(extra)
    return payload


def _post(url: str, data: bytes, api_key: str, timeout: float):
    """One POST. Returns ``(status, body_bytes, headers)``; network faults raise."""
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "comfyui-object-extract/1.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        # An error status is still a complete response -- keep the body, it is
        # the only thing that explains *why*.
        try:
            body = exc.read()
        except Exception:
            body = b""
        return int(exc.code), body, dict(exc.headers or {})


def _error_message(status: int, body: bytes) -> str:
    """Pull the vendor's own message out of an error body when there is one."""
    text = body.decode("utf-8", "replace") if body else ""
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return short(text) or "(empty body)"
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            return short(error.get("message") or error)
        if isinstance(error, str):
            return short(error)
        for key in ("message", "detail", "msg", "error_msg"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return short(value)
        if isinstance(parsed.get("code"), (str, int)):
            return short(f"code={parsed.get('code')} {parsed.get('message', '')}")
    return short(text) or f"HTTP {status}"


def _flatten_content(content) -> str:
    """Reduce a message ``content`` field to plain text.

    OpenAI-compatible servers return a string, but multimodal and reasoning
    models sometimes return a list of parts (``[{"type": "text", "text": ...}]``)
    -- and a few wrap the whole thing one level deeper.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                for key in ("text", "content", "value"):
                    value = part.get(key)
                    if isinstance(value, str):
                        chunks.append(value)
                        break
        return "".join(chunks)
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return ""
    return str(content)


def extract_text(obj: dict) -> str:
    """Read the *final answer* out of a chat-completions response body.

    Only the answer. A reasoning model leaks its scratchpad in two shapes and
    both are dropped here: inline (``thinking...`` inside ``content``) and
    side-channel (a separate ``reasoning_content`` field).

    The side channel used to be a last-resort stand-in when ``content`` came
    back empty, on the theory that a truncated reply still parses. It does --
    it parses into objects the model only *imagined* while thinking. An empty
    answer is the honest result, and the node reports it as such.
    """
    choices = obj.get("choices") or []
    if not choices:
        raise VLMError("response has no 'choices' array")
    choice = choices[0] or {}
    message = choice.get("message") or choice.get("delta") or {}
    text = strip_thinking(_flatten_content(message.get("content")))
    if not text:
        # Legacy completions shape, still emitted by a few gateways.
        text = strip_thinking(_flatten_content(choice.get("text")))
    return text


def extract_thinking(obj: dict) -> str:
    """Collect the scratchpad :func:`extract_text` dropped, for the report.

    Kept out of the answer but not thrown away: the node prints its size, so
    "the model burned 900 tokens thinking and then said nothing" is visible,
    instead of looking like a model that simply had nothing to say.
    """
    choices = obj.get("choices") or [{}]
    choice = choices[0] or {}
    message = choice.get("message") or choice.get("delta") or {}

    parts: list[str] = []
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    inline = split_thinking(_flatten_content(message.get("content")))[1]
    if inline:
        parts.append(inline)
    return "\n".join(parts)


def _usage(obj: dict) -> tuple[int | None, int | None, int | None]:
    usage = obj.get("usage") or {}
    if not isinstance(usage, dict):
        return None, None, None

    def num(*keys):
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return None

    return (num("prompt_tokens", "input_tokens"),
            num("completion_tokens", "output_tokens"),
            num("total_tokens"))


# --------------------------------------------------------------------------- #
# the call
# --------------------------------------------------------------------------- #

def _send(body: dict, api_key: str, url: str, timeout: float,
          retries: int) -> ChatResult:
    """One request, plus retries for transient transport trouble.

    Split out of :func:`chat` so the budget-escalation loop can re-send a
    *different* body without duplicating the retry logic.
    """
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    model = str(body.get("model", ""))
    result = ChatResult(model=model, endpoint=url, request_bytes=len(data))
    tries = max(0, int(retries)) + 1

    for attempt in range(tries):
        result.attempts = attempt + 1
        try:
            status, response_body, _headers = _post(url, data, api_key, timeout)
        except VLMError:
            raise
        except Exception as exc:  # URLError, socket.timeout, ssl errors, OSError
            reason = getattr(exc, "reason", None) or exc
            result.status = 0
            result.error = f"network error: {short(reason)}"
            if attempt + 1 < tries:
                time.sleep(min(2.0 ** attempt, _MAX_BACKOFF))
                continue
            break

        result.status = status

        if 200 <= status < 300:
            try:
                obj = json.loads(response_body.decode("utf-8", "replace"))
            except (ValueError, TypeError):
                # A proxy or captive portal answering 200 with HTML.
                result.error = (f"HTTP {status} but the body is not JSON: "
                                f"{short(response_body, 160)}")
                break
            if not isinstance(obj, dict):
                result.error = f"HTTP {status} but the body is a JSON {type(obj).__name__}"
                break
            try:
                result.text = extract_text(obj)
            except VLMError as exc:
                result.error = str(exc)
                break
            result.reasoning = extract_thinking(obj)
            (result.prompt_tokens, result.completion_tokens,
             result.total_tokens) = _usage(obj)
            choices = obj.get("choices") or [{}]
            result.finish_reason = str((choices[0] or {}).get("finish_reason") or "")
            result.model = str(obj.get("model") or model)
            result.error = ""
            break

        result.error = f"HTTP {status}: {_error_message(status, response_body)}"
        if status in _RETRY_STATUS and attempt + 1 < tries:
            time.sleep(min(2.0 ** attempt, _MAX_BACKOFF))
            continue
        break

    return result


def _starved(result: ChatResult) -> bool:
    """True when the completion budget went entirely into the scratchpad.

    HTTP 200, tokens on the invoice, empty answer -- because the model was still
    thinking when it ran out of room. The last clause matters: a model that had
    nothing to say is a different story, and re-sending that would just buy the
    same silence.
    """
    return (not result.error
            and result.truncated
            and not result.text.strip()
            and bool(result.reasoning.strip()))


def chat(payload: dict, api_key: str, base_url: str, timeout: float = 60.0,
         retries: int = 2, escalate: int = 2) -> ChatResult:
    """POST ``payload`` and normalise the outcome into a :class:`ChatResult`.

    Retries only transient failures (network faults, 408/409/425/429/5xx) with
    exponential backoff; a 400/401/403/404 comes straight back because it is a
    configuration bug and the user needs the message, not another attempt.

    ``escalate`` covers the other way a call comes back empty: a reasoning model
    that spends the whole ``max_tokens`` budget thinking and never reaches the
    answer. Re-sending pays for the image twice, but the alternative is a
    fallback prompt and a wrong picture -- and unlike an ordinary retry there is
    no answer being thrown away, so it cannot make a good run worse.
    """
    url = normalize_endpoint(base_url)
    model = str(payload.get("model", ""))
    started = time.monotonic()

    body = dict(payload)
    result = ChatResult(model=model, endpoint=url)
    budgets: list[int] = []
    attempts = 0

    for _round in range(max(0, int(escalate)) + 1):
        result = _send(body, api_key, url, timeout, retries)
        attempts += result.attempts
        current = int(body.get("max_tokens") or 0)
        budgets.append(current)
        if not _starved(result):
            break
        bumped = min(max(current * 2, _ESCALATE_FLOOR), _TOKEN_CAP)
        if bumped <= current:
            break  # already at the ceiling: another round buys nothing
        body["max_tokens"] = bumped

    result.attempts = attempts
    result.budgets = tuple(budgets)
    result.elapsed = time.monotonic() - started
    return result
