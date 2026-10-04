# comfyui-object-extract

**从一张场景图里把多个物体各自提取出来。**

识图模型看完图，说出「场景里有哪些物体、分别在哪」；这两个节点负责把它的
自由文本回复变成下游真正能用的东西 —— 一个物体一次：

| 节点 | 做什么 | 何时用 |
|---|---|---|
| `Object Prompts from VLM` | VLM 回复 → 物体定位信息（`名:个数` 提示词 + 边界框列表） | 下游接 SAM3 出精确遮罩，或接下面的节点 |
| `BBoxes to Masks` | 每个框画成实心矩形 → **粗遮罩**批 | 不要分割器，裁到框交给生成式编辑器逐个分离/重绘 |

---

## 1. `Object Prompts from VLM`

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

## 2. `BBoxes to Masks`

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

**解析失败 = 安全降级**，不会报错：

- 没解析出 `name:count` → `prompt` 为空串 → SAM3 走它自己的单提示路径
- 没解析出框 → `bboxes` 为空列表 → SAM3 的 `len(bboxes) > 0` 判断为假，
  直接跳过 `bboxes` 端口（**不是** `ImpactMakeAnyList` 那种空列表崩溃）
- `BBoxes to Masks` 收到空框 → 按 `if_empty` 决定是整图遮罩还是空批，都不报错

能处理：markdown 围栏、前后散文、裸键 JSON（`{name: "apple"}`）、尾逗号、
字符串数字、嵌套 `{"objects": [...]}`、`x1y1x2y2` / `x/y/width/height` 字典、
0–1000 与 0–1 归一化坐标、反向坐标、越界裁剪、中文物体名。

## 与 `ImpactMakeAnyList` 接法的区别

`ImpactMakeAnyList` 声明了 `OUTPUT_IS_LIST`，执行器会把下游 SAM3 **逐框执行 N 次**，
于是「多框 + 文本」会产出 N×max_det 张重复遮罩。
`ObjectExtract_PromptsFromVLM` 把框列表作为**单个值**输出，SAM3 只跑一次：有文本时框作为
几何提示融合，无文本时逐个框解码 —— 每框一张，干净。

## 测试

纯解析逻辑（`parse.py`）不依赖 ComfyUI，可离线单测：
`.workbuddy/test_object_extract_parse.py`。节点冒烟测试
`.workbuddy/test_object_extract_node.py`，真机端到端 `.workbuddy/e2e_test_bboxmask.py`。
