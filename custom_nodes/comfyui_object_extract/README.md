# comfyui-object-extract

**从一张场景图里把多个物体各自提取出来。**

识图模型看完图，说出「场景里有哪些物体、分别在哪」；这三个节点把「问模型」
和「把它的自由文本回复变成下游真正能用的东西」都包圆了 —— 一个物体一次：

| 节点 | 做什么 | 何时用 |
|---|---|---|
| `VLM Query (OpenAI compatible)` | 图 → 调外部多模态 API → **直接**得到 `名:个数` 提示词 + 边界框 | 识图要在 ComfyUI 里自动跑 |
| `Object Prompts from VLM` | 你粘进来的模型回复 → 同样的提示词 + 边界框 | 识图在别处做、或要离线 |
| `BBoxes to Masks` | 每个框画成实心矩形 → **粗遮罩**批 | 不要分割器，裁到框交给生成式编辑器逐个分离/重绘 |

---

## 1. `VLM Query (OpenAI compatible)`

把手上的图发给任意 **OpenAI 兼容**的视觉接口，读回回复，**当场解析**。
输出 = 「`Object Prompts from VLM` 吃这个回复会吐出来的东西」，所以它可以
直接替掉那个节点，也能和它串起来用。

| 输出 | 类型 | 接到 |
|---|---|---|
| `prompt` | STRING | `CLIPTextEncode.text` → `SAM3_Detect.conditioning` |
| `bboxes` | BOUNDING_BOX | `SAM3_Detect.bboxes`（每个 dict 额外带 `label`） |
| `text` | STRING | `PreviewTextNode` —— **只含最终回复**，思考过程已剔除 |
| `report` | STRING | `PreviewTextNode` —— 请求详情 + 解析结果 |

### 为什么自己实现传输层而不用 `requests`

`vlm_api.py` **只用标准库**（`urllib` + `base64` + `json`），包不引入任何新依赖；
而且没有 ComfyUI import，所以可以脱离 ComfyUI 对着本地 mock 服务器单测。
见下面的「测试」。

### 接口按 OpenAI 那一套来

```jsonc
POST {base_url}/chat/completions
{"model": "...", "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "..."}]}],
 "temperature": 0.2, "max_tokens": 1024, "stream": false}
```

OpenAI、阿里云百炼（`/compatible-mode/v1`）、智谱、火山方舟、硅基流动、
DeepSeek、以及本地 vLLM / LM Studio / Ollama 的兼容口都走这套 ——
**只改 `base_url` 和 `model`**。`base_url` 结尾写不写 `/chat/completions`
都行，写了就不重复追加。

### 密钥只从环境变量读

| 参数 | 默认 | 说明 |
|---|---|---|
| `api_key_env` | `DASHSCOPE_API_KEY` | 环境变量名。空格或逗号分隔可写**多个**，从左往右取第一个有值的 |
| `api_key` | `""` | **仅作兜底**：环境变量全都没有时才用它。填了会明文写进工作流 JSON |

顺序刻意是「环境变量优先」。填在 widget 里的密钥会随工作流 JSON 被序列化，
一分享/提交就泄露；环境变量不会。所以安全的路径是默认路径，不安全的那条
得等到环境变量缺失才生效。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `prompt` | 见下 | 给模型的指令，要求它回 JSON 数组 |
| `system_prompt` | `""` | 可选 system 轮。有些模型把格式要求放这里更听话 |
| `model` | `qwen-vl-max` | 接口那边的模型 id |
| `base_url` | DashScope 兼容口 | 见上表 |
| `temperature` | `0.2` | 0 最稳定；给一点温度能让部分模型别复读同一个物体 |
| `max_tokens` | `8192` | 回复预算**上限，不是配额** —— 只按真实生成的 token 计费，别省。推理模型的思考也从这里扣（见下） |
| `on_error` | `raise` | API 报错时停下（见下）；`fallback` 则用 `fallback_prompt` 继续 |
| `fallback_prompt` | `""` | 解析不出提示词时顶上的 `name:count` |
| `timeout` | `60` | 每次尝试的秒数。带推理的模型要调大 |
| `retries` | `2` | 网络故障 / 429 / 5xx 的重试次数。**4xx 客户端错误从不重试** |
| `frame_index` | `0` | 图批（batch）里发第几帧 |
| `max_side` | `0` | 长边缩到这个像素再发，省 token。`0` = 原图；坐标会自动换算回去 |
| `image_format` | `png` | `png` 无损、细节更好；`jpeg` 体积小得多 |
| `jpeg_quality` | `90` | 只在 `image_format = jpeg` 时用 |
| `detail` | `auto` | OpenAI 的 image detail 提示。**只有非 `auto` 才发这个字段**，因为严格些的兼容服务会拒绝它 |
| `extra_body` | `""` | JSON 对象，**整份合并进请求体**，用于厂商私有开关，可覆盖上面的字段。关思考：DeepSeek / Zhipu 用 `{"thinking": {"type": "disabled"}}`，Qwen（百炼）用 `{"enable_thinking": false}`；强制合法 JSON 用 `{"response_format": {"type": "json_object"}}` |
| `coord_mode` / `box_format` / `pad` / `min_size` / `dedupe` / `count_cap` | | 与 `Object Prompts from VLM` 含义完全一致 |

默认提示词（要求 JSON 数组，名字限制 1–3 个单词，因为 SAM3 每个名字只有 32 个 CLIP token 的预算）：

```
List every distinct object in this image that could be cut out and repainted.
Reply with JSON only, no prose, no markdown:
[{"name": "object", "bbox": [x1, y1, x2, y2]}]
- name: singular lowercase English noun, 1-3 common words
- bbox: [left, top, right, bottom] in pixels of this image
- list each object once; never repeat a box
- cover every object you can see; stop when you are done
Reply:
```

### `on_error`：为什么默认是 `raise`

内容问题（模型没按格式回）**一律安全降级**，和另一个节点一样；
但**传输**问题（密钥错、地址错、服务没起）默认**抛异常停下**。

理由是静默失败最坑：会拿一条兜底提示词把整条重绘链路跑完，
你拿到一堆莫名其妙的图，才发现密钥早就填错了。无人值守跑批时再改 `fallback`。

### 图片编码的几个决定

- **alpha 合成到白底**，不是丢弃。`Image.convert("RGB")` 直接把 alpha 扔掉，
  而抠图类 WebP/PNG 里全透明的像素往往存着乱七八糟的颜色 —— 会在主体边缘
  留下一圈彩色毛边，干扰模型。
- **`max_side` 缩图后坐标自动换算回原图**。模型老实汇报它**看到的那张图**的
  坐标；不换算的话框会错位，而且**不报错**，要等到三个节点之后遮罩盖错地方
  才看得出来。`report` 里会写一行 `coords : scaled by ...` 供核对。
- **图批只发一帧**（`frame_index`），并在 `report` 里写明发的是第几帧。

### 同步还是异步

节点是 `async def execute`，网络请求跑在 `asyncio.to_thread` 里。
**这是必须的**：ComfyUI 在事件循环线程上直接执行同步节点函数，
一个阻塞的 HTTP 调用会把整个服务冻住（进度条不动、取消不了），时长等于模型耗时。

### 思考过程不会被当成回复

推理模型（Qwen3 系列、DeepSeek V4 的 `deepseek-flash` / `deepseek-v4-pro`、o 系列…）
的思考过程会以两种形态回来，**两种都在这里被剔掉**，只留最终回复：

| 形态 | 例子 | 处理 |
|---|---|---|
| 独立字段 | `{"content": "[...]", "reasoning_content": "让我先看看…"}` | 只读 `content`，`reasoning_content` 不再当兜底 |
| 内联在正文里 | 正文里用 think 标签对把思考包起来 | 把标签块删掉 |

**为什么必须剔**：思考过程是通顺的散文，而且**满是数字**
（"菠萝大概在 [80,120,500,900] 这一带…"）。框解析器看到四数字组就当真，
于是凭空造出一堆不存在的框 —— 而且是那种能跑通、只是结果不对的失败。

三点实现细节：

- 标签名认 `think` / `thinking` / `reasoning` / `thought` / `analysis`，
  大小写不敏感、跨行匹配。
- **落单的标签**也算：只有开标签 = 回复被 `max_tokens` 截断在思考中途，
  从那儿往后全是思考，一并删掉；只有闭标签则相反，它之前的是思考。
  两个方向都处理，是为了不让半句思考漏到解析器前面。
- 标签**之外**的散文无法与正文区分，会保留（已在此说明，别指望它是万能的）。

模型只思考、没给最终回复时，**不会拿思考顶包**：`text` 输出空串，
`report` 里写明原因，再由 `fallback_prompt` 接管。思考**丢弃但留档**：
`report` 的 `think  :` 行会写出丢了多少字符，所以「它是不是白想了」看得见。

### ★ 思考会把 `max_tokens` 吃光（2026-10-04 实测踩到）

思考是从**同一个** `max_tokens` 预算里扣的。预算不够时它想一半就用光了，
**永远轮不到写答案** —— 而 HTTP 状态码是 200，钱照付，回复是空的：

```
reply  : HTTP 200 in 4.95s (1 attempt(s))   tokens 1035+1024=2059   finish=length
think  : 3455 chars of reasoning, but no answer even at max_tokens=8192
```

两道防线：

1. **`max_tokens` 默认 8192。** 它是上限不是配额，开大不花钱，
   而推理模型的思考**默认就是开的**（DeepSeek V4 即如此，且思考模式
   官方默认预算高达 64K —— 手动填 1024 等于把预算全让给思考）。
2. **自动扩容重发。** 只要命中「`finish_reason = length` + 最终回复为空 +
   思考非空」，节点就用更大的预算重发：`1024 → 4096 → 8192`
   （首个档位直接跳到 4096，因为会踩坑的预算都很小，翻倍只是多踩一轮）。
   `report` 里留一行 `retry  : max_tokens ...`。
   重发会再付一次图片的输入 token，但那是**本来就被浪费掉的调用**，
   不存在「把好结果重试坏」的风险；而「服务端没截断、模型确实没话说」
   不会触发重发。上限 8192 是因为那是主流厂商都文档化的最大输出长度，
   要更多容易被回 400，反而把诊断信息盖掉。

> **最省的做法还是直接关掉思考。** 这活是「看图报框」，一步到位的感知任务：
> 思考没有收益，只带来 7 倍的费用和延迟（实测同一模型关思考后
> 输出 44 token，开着思考是 328 token）。填 `extra_body`：
> DeepSeek / Zhipu 用 `{"thinking": {"type": "disabled"}}`，
> Qwen（百炼）用 `{"enable_thinking": false}`。
> 这类字段是厂商私有的，所以没写进默认值 —— 换厂商时记得改或清空。

---

## 2. `Object Prompts from VLM`

| 输出 | 类型 | 接到 |
|---|---|---|
| `prompt` | STRING | `CLIPTextEncode.text` → `SAM3_Detect.conditioning` |
| `bboxes` | BOUNDING_BOX | `SAM3_Detect.bboxes`（每个 dict 额外带一个 `label`，消费者只读四个坐标键） |
| `report` | STRING | `PreviewTextNode`（调试用） |

### 为什么需要它

SAM3 的文本提示有硬语法（`comfy/text_encoders/sam3_clip.py::_parse_prompts`）：
按**英文逗号**切分，每段必须整段匹配 `^(.+?)\s*:\s*([\d.]+)$`，否则整条提示词
会退化成一条 `max_det=1` 的怪提示词。而 VLM 只会吐出散文或 JSON。
`CLIPTextEncode` 做不了这个翻译，`CreateBoundingBoxes` 也做不了——它的
`editor_state` 是必填 socket，类型 `BOUNDING_BOXES` 没有任何内置节点能产出。

### 接法

```
TextGenerate ──┬──> Object Prompts from VLM ──┬─ prompt ─> CLIPTextEncode ─> SAM3_Detect.conditioning
               │                            └─ bboxes ─────────────────────> SAM3_Detect.bboxes
               └──> 直接接也只会得到原始散文，SAM3 读不懂
```

`image` 可选接进 `LoadImage` 的图，只用来拿宽高做归一化坐标换算。

手动粘贴的回复也走同一道思考清洗（见上一节）：粘进来的如果是一整段思考加答案，
标签块会被丢掉，`report` 里留一行说明。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `coord_mode` | `auto` | `auto` 按图尺寸与数值范围猜；歧义时偏向像素 |
| `box_format` | `auto` | 裸四元组按 `xyxy`，带 `width/height` 的字典按 `xywh` |
| `pad` | `0` | 每个框向外扩 N 像素（再裁剪到图内） |
| `min_size` | `4` | 小于此尺寸的框丢弃，挡掉散文里的散落数字 |
| `dedupe` | `true` | IoU ≥ 0.95 的重复框丢掉（VLM 长回复会复读） |
| `count_cap` | `20` | `:N` 的上限，`0` 不限制 |

---

## 3. `BBoxes to Masks`

把每个框画成一个**实心矩形**，输出 `[N, H, W]` 的 MASK 批 —— 也就是「粗遮罩」。
不跑任何分割模型。

| 输出 | 类型 | 接到 |
|---|---|---|
| `masks` | MASK（批） | `Masks To Mask List` → 之后每个下游节点按框逐个跑 |
| `prompts` | STRING（**列表**，长度 = 框数） | `Text Encode Qwen Image 2.1.prompt` |
| `report` | STRING | `PreviewTextNode` |

### 为什么要用它

精确分割器不一定比一个框好用。矩形遮罩保留了框内**全部内容**（背景杂物也在），
裁到框之后交给生成式编辑器，由提示词告诉它「只分离出那个 pineapple」，
比先抠图再重绘更稳、也不需要额外的分割模型。

`prompts` 是这条链路的关键：它把每个框的 `label`（来自解析节点的物体名）
填进 `prompt_template` 的 `{name}`，得到 N 条提示词。
执行器按**同一索引**切所有列表输入，所以第 i 条提示词正好配第 i 个裁剪结果 ——
不需要额外对齐。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `width` / `height` | 1024 | 只在没接 `image` 时用作画布尺寸 |
| `grow` | `0` | 每边外扩 N 像素；**负数=内缩**，框太松时用它修 |
| `min_size` | `8` | 外扩后仍小于此边长的框丢弃 |
| `limit` | `0` | 最多保留几个框，挡住跑飞的回复；`0` = 不限 |
| `if_empty` | `full_image` | 一个框都没有时：整图一张遮罩（图还能出）或 `none`（下游不跑） |
| `prompt_template` | 见下 | 每框一条提示词，`{name}` 替换成物体名；没名字时用 `main object` |

`bboxes` 接受三种形状：裸字典（`PrimitiveBoundingBox`）、扁平列表（本包解析节点）、
帧列表 `[[box, box], ...]`（`SAM3_Detect` 输出）。多帧只取第一个非空帧。

---

## 容错设计

前两个节点**解析失败 = 安全降级**，不会报错：

- 没解析出 `name:count` → `prompt` 为空串 → SAM3 走它自己的单提示路径
- 没解析出框 → `bboxes` 为空列表 → SAM3 的 `len(bboxes) > 0` 判断为假，
  直接跳过 `bboxes` 端口（**不是** `ImpactMakeAnyList` 那种空列表崩溃）
- `BBoxes to Masks` 收到空框 → 按 `if_empty` 决定是整图遮罩还是空批，都不报错

能处理：markdown 围栏、前后散文、裸键 JSON（`{name: "apple"}`）、尾逗号、
字符串数字、嵌套 `{"objects": [...]}`、`x1y1x2y2` / `x/y/width/height` 字典、
0–1000 与 0–1 归一化坐标、反向坐标、越界裁剪、中文物体名。

`VLM Query` 只在**传输**层会抛异常，且默认就是抛（见 `on_error`）；
把模型回复交给的仍是同一套解析，所以格式问题一样不会崩。

## 与 `ImpactMakeAnyList` 接法的区别

`ImpactMakeAnyList` 声明了 `OUTPUT_IS_LIST`，执行器会把下游 SAM3 **逐框执行 N 次**，
于是「多框 + 文本」会产出 N×max_det 张重复遮罩。
本包的解析节点把框列表作为**单个值**输出，SAM3 只跑一次：有文本时框作为
几何提示融合，无文本时逐个框解码 —— 每框一张，干净。
`VLM Query` 的 `bboxes` 同样不声明 `OUTPUT_IS_LIST`。

## 测试

`parse.py` 和 `vlm_api.py` 都不依赖 ComfyUI，可离线单测：

| 脚本 | 测什么 |
|---|---|
| `.workbuddy/test_object_extract_parse.py` | 回复解析、坐标换算、去重、框规整 |
| `.workbuddy/test_vlm_api.py` | 请求体构造、回复抽取、思考剥离、data URL；对着本地 mock 服务器真发 HTTP，覆盖 200 / 401 不重试 / 5xx 重试 / 非 JSON 响应 / 连不上的端口 |
| `.workbuddy/e2e_vlmquery.py` | 真机端到端：mock 视觉服务替换厂商，跑通完整 graph（正常路径、兜底、降采样回算、`raise`、`fallback`、思考在独立字段、思考内联、只有思考没答案） |
| `.workbuddy/test_object_extract_node.py`、`.workbuddy/e2e_test_bboxmask.py` | 节点冒烟与粗遮罩链路 |

> `vlm_api.py` 刻意**不写** `from __future__ import annotations`：
> 那样 `@dataclass` 会走 `sys.modules[cls.__module__]` 解析字符串注解，
> 而用 `spec_from_file_location` 独立加载这个文件时它必然是 `None`，
> 直接 `AttributeError`。离线测试就是这么加载的。
