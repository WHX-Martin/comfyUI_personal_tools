# comfyUI_tools

个人 ComfyUI 工具集：**三个自定义节点包 + 一批可复用工作流**，围绕两条主线：

1. **无缝（四方连续）PBR 材质** —— 用「把卷积 padding 改成 circular」而不是「事后羽化接缝」的方式，从 SDXL 直接生成可平铺贴图，再派生出像素对齐的 normal / height / AO / roughness / ORM。
2. **场景图 → 多物体 3D 模型** —— VLM 识图报框 → SAM3 分割 → 逐个重绘 → Pixal3D + Trellis2 出 GLB。

外加一个小工具包做原生 `MESH` ↔ `trimesh` 的互转。

针对 **ComfyUI Desktop 0.38.2**（节点 V3 API `comfy_api.latest`）开发验证。

---

## 目录结构

```
comfyUI_tools/
├─ custom_nodes/
│  ├─ comfyui_seamless_texture/   18 节点 · 无缝 PBR 材质（主力包）
│  ├─ comfyui_object_extract/      3 节点 · VLM 识图 → 物体提示词 / 边界框 / 粗遮罩
│  └─ comfyui_mesh_trimesh/        2 节点 · MESH ↔ trimesh
└─ workflows/                     17 个 UI 格式 .json，直接拖进 ComfyUI
```

## 自定义节点包

| 包 | 节点数 | 版本 | 许可 | 额外依赖 | 说明 |
|---|---|---|---|---|---|
| [`comfyui_seamless_texture`](custom_nodes/comfyui_seamless_texture/) | 18 | 1.3.0 | MIT | `comfyui_controlnet_aux`（DSINE 架构）+ `dsine.pt` | 四方连续材质端到端：环面采样、卷积法线估计、泊松积分、PBR 派生、周期超分 |
| [`comfyui_object_extract`](custom_nodes/comfyui_object_extract/) | 3 | 1.1.0 | MIT | 无（纯标准库） | 一张场景图里提出多个物体：外部多模态 API → 解析 → `name:count` 提示词 + 边界框 + 粗遮罩 |
| [`comfyui_mesh_trimesh`](custom_nodes/comfyui_mesh_trimesh/) | 2 | 0.1.0 | MIT | `trimesh>=4.0.0` | 原生 `MESH` 与 trimesh 几何体互转，带批量语义与变长批次处理 |

各包的完整节点表、参数、实测数据与踩坑记录都在各自的 `README.md` 里，下面是索引：

### `comfyui_seamless_texture`（18 节点）

| 分组 | 节点 | 作用 |
|---|---|---|
| 生成（padding） | `Seamless Model Padding` / `Seamless VAE Decode` / `Seamless VAE Encode` / `Tile Preview` | 把 UNet 与 VAE 跑到**环面**上：每个 `Conv2d` 改 `padding_mode="circular"`，输出按构造严格周期，而不是「补一层边再想办法糊掉」 |
| 派生 | `Seamless Height From Albedo` / `Seamless Normal From Height` / `Seamless AO From Height` / `Seamless Roughness From Albedo` / `Pack PBR Channels` | 从 albedo 派生 PBR，每个模糊与梯度都走 `torch.roll` / circular，无缝性不会在派生中丢失 |
| 几何反演 | `Seamless DSINE Normal` / `Seamless Flatten Normal` / `Seamless Height From Normal` / `Seamless Delight Albedo` / `Seamless Roughness From Height` / `Seamless MoGe Inference` | 从**受光渲染图**反推表面起伏：纯卷积估计器 DSINE → 周期泊松积分成 height |
| 超分 | `Seamless Upscale` / `Seamless Resize` | 周期延拓 → 超分 → 裁回中心，修掉 SR 网络唯一会破坏平铺的地方（图像边缘） |
| 已弃用 | `Seamless Materials From Albedo` | StableMaterials 路线，留档对照 |

### `comfyui_object_extract`（3 节点）

| 节点 | 作用 |
|---|---|
| `VLM Query (OpenAI compatible)` | 图 → 调外部多模态 API → **当场解析**成 `name:count` 提示词 + 边界框 |
| `Object Prompts from VLM` | 粘贴进来的模型回复 → 同样的提示词 + 边界框（离线 / 识图在别处做） |
| `BBoxes to Masks` | 每框画成实心矩形 → 粗遮罩批，交给生成式编辑器逐个分离重绘 |

### `comfyui_mesh_trimesh`（2 节点）

`Mesh to Trimesh` / `Trimesh to Mesh`，通过自定义类型 `TRIMESH` 连接，分类在 `3d/trimesh`。

---

## 工作流

### 材质 / PBR

| 文件 | 节点 | 说明 |
|---|---|---|
| `sdxl_sand_pbr_v2_geom.json` | 43 | **当前主链路**。受光渲染 → DSINE 法线 → 泊松积分 height → 重导 normal / AO / roughness / ORM；MoGe 保留为对照支路。端到端约 16 s |
| `sdxl_sand_pbr_seamless.json` | 22 | 派生版：albedo 高通出 height，**不需要任何额外权重**；只能还原打光实际照出来的起伏 |
| `sdxl_sand_pbr_stablematerials.json` | 42 | StableMaterials 模型生成版（已弃用，留档） |
| `sdxl_sand_albedo_seamless.json` | 12 | 最小可跑：只出四方连续 albedo |
| `sand_upscale_ab.json` | 10 | 超分 A/B：普通 upscale 对照 `SeamlessUpscale` |

### 场景 → 3D

| 文件 | 节点 | 说明 |
|---|---|---|
| `scene_to_3d_multimodel_repaint.json` | 73 | **最新**。场景图 → 多物体逐个重绘 → 3D 模型 |
| `scene_to_3d_multimodel.json` | 49 | 上者的简化版 |
| `3d_pixal3d_trellis2_image_to_model.json` | 62 | 单图 → 3D 模型（Pixal3D + Trellis2） |

### SAM3 分割 / 逐个重绘

| 文件 | 节点 | 说明 |
|---|---|---|
| `sam3_text_to_mask_template.json` | 7 | 官方 subgraph 模板，最小起点 |
| `sam3_text_mask_inpaint.json` | 19 | 基础：文字 → 多物体遮罩 → Qwen-Image 2.1 逐个重绘 |
| `sam3_text_mask_inpaint_auto.json` | 28 | 自动版 |
| `sam3_text_mask_inpaint_bbox.json` | 33 | 带框提示 |
| `sam3_text_mask_inpaint_vlm.json` | 26 | 全自动：本机 Qwen3-VL 识图 |
| `sam3_text_mask_inpaint_api.json` | 24 | 内嵌 VLM 查询节点（OpenAI 兼容 API） |
| `sam3_text_mask_inpaint_ext.json` | 26 | 外部模型版：手动粘贴回复 |
| `qwen_vlm_bbox_inpaint.json` | 22 | 无 SAM3 版，本机 Qwen3-VL |
| `qwen_ext_bbox_inpaint.json` | 22 | 无 SAM3 + 外部模型 |

> UI 格式的 workflow 直接拖进 ComfyUI 即可打开。注意 ComfyUI 实际加载的是它自己数据目录下的副本，改仓库里的 json 不会影响正在跑的实例。

---

## 安装

把需要的包目录放进 ComfyUI 的 `custom_nodes/`（复制或 junction 均可），装依赖后重启：

```bash
# comfyui_seamless_texture 需要
pip install comfyui_controlnet_aux   # 或放进 custom_nodes
# comfyui_mesh_trimesh 需要
pip install trimesh>=4.0.0
# comfyui_object_extract 无额外依赖
```

`Seamless DSINE Normal` 需要 `dsine.pt`（约 278 MB）；找不到时节点会报错并打印可直接执行的 `curl` 下载命令。

## 环境

| 项 | 值 |
|---|---|
| ComfyUI | Desktop 0.38.2（HTTP 服务默认 `http://127.0.0.1:8000`，**不是** 8188） |
| 节点 API | V3（`comfy_api.latest`，`ComfyExtension` + `comfy_entrypoint`） |
| 数据目录 | `.../ComfyUI-data/`（`custom_nodes/`、`models/`、`user/default/workflows/`） |
| 运行 Python | ComfyUI 自带 venv（3.12 + torch），而非系统 Python |

---

## 几个决定设计走向的结论

这些是本仓库与「常见做法」分歧的地方，完整推导在各包 README：

- **无缝是 padding 的属性，不是后处理能补的。** 零填充下特征图是矩形，模型会自己画一圈边界 —— 那圈边界就是接缝。改成 circular 后特征图拓扑上是环面，输出严格周期。
- **「ViT 有缝、UNet 无缝」是错的。** 无缝需要三个条件同时成立：有界感受野 + 无绝对位置输入 + 无全局归一化。所以几何估计选了纯卷积的 DSINE（EfficientNet-B5 + ConvGRU，零 attention），而不是 ViT 系的 MoGe / DepthAnything。
- **DSINE 的接缝是 FOV 造成的，补边治不了。** 它吃逐像素光线方向（绝对位置场）。默认 `fov=60` 时补边无效；`fov=3 + pad=256` 下环绕差值降到 1.11，与图内普通相邻差无法区分。
- **主法线要从 height 重导，而不是直接用估计器直出。** 法线场一般不可积；而 AO / roughness / height 全部由 height 导出，主法线必须同源，否则渲染时明暗与遮蔽不在同一几何上。
- **height 图绝不能被截断。** 任何硬限幅都会变成零梯度平台，而零梯度平台就是一块精确的平面。用分位数锚定的**分段线性**映射（每段斜率严格为正）替代 `clamp`。
- **超分只在四条边破坏无缝。** 实测 RealESRGAN_x2 边缘带差异 0.00422、内部仅 0.00010 —— 是边界条件问题不是滤波器问题，所以先周期延拓再裁中心。
- **normal / roughness / metallic 不能直接超分。** 法线通道是向量分量不是颜色；低频无结构的图会被 SR 编造出与 height 不符的结构。只超分 height 再重导。

## 许可

MIT。三个包均为 MIT。
