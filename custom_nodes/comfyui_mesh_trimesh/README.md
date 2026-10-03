# comfyui-mesh-trimesh

在 ComfyUI 原生 `MESH` 类型与 [`trimesh`](https://trimesh.org/) 几何体之间互相转换,方便把 ComfyUI 的 3D 流程接到 Python 的 trimesh 生态里做处理、分析或导出。

## 节点

| 节点 | node_id | 作用 |
|---|---|---|
| **Mesh to Trimesh** | `MeshTrimesh_MeshToTrimesh` | `MESH` → `trimesh.Trimesh` / `trimesh.Scene` |
| **Trimesh to Mesh** | `MeshTrimesh_TrimeshToMesh` | `trimesh.Trimesh` / `trimesh.Scene` / 列表 → `MESH` |

分类位于 `3d/trimesh`,两者通过自定义类型 `TRIMESH` 连接。

### Mesh to Trimesh

| | 名称 | 类型 | 说明 |
|---|---|---|---|
| 输入 | `mesh` | `MESH` | 待转换的网格 |
| 输入 | `process` | `BOOLEAN` | 默认关闭。开启后交给 trimesh 合并重复顶点、移除退化面;关闭则原样转换 |

输出 `trimesh`(`TRIMESH`)。

### Trimesh to Mesh

| | 名称 | 类型 | 说明 |
|---|---|---|---|
| 输入 | `trimesh` | `TRIMESH` | `trimesh.Trimesh`、`trimesh.Scene`,或它们的列表 |

输出 `mesh`(`MESH`)。

## 批量语义

ComfyUI 的 `MESH` 是**批处理**结构,而 trimesh 没有批次概念,所以映射关系是:

- `MESH` 批次只有 1 件 → 输出单个 `trimesh.Trimesh`;
- `MESH` 批次多于 1 件 → 输出 `trimesh.Scene`,每件一个 geometry(按顺序命名 `mesh_000`、`mesh_001`……);
- 反过来,`Trimesh` → 1 个批次项;`Scene` → 每个 geometry 一个批次项;**列表** → 逐项展开。

因此 `MESH(B=3) → Scene → MESH(B=3)` 能精确往返,不会把多个部件压成一个。传入的 `Scene` 会先 `dump()`,即各 geometry 的变换矩阵会被烘焙进顶点坐标,导入的场景图不会丢失摆放位置。

变长批次(各件顶点数不同)按 ComfyUI 惯例处理:零填充到统一长度,并用 `vertex_counts` / `face_counts` 记录真实长度;转换时按该长度裁掉填充,不会产生幽灵顶点或幽灵面。

## 数据映射

| ComfyUI `MESH` | trimesh |
|---|---|
| `vertices (B,N,3)` float32 | `Trimesh.vertices` |
| `faces (B,M,3)` int64 | `Trimesh.faces` |
| `uvs (B,N,2)` | `visual.uv` |
| `vertex_colors (B,N,3或4)` 0–1 float | `visual.vertex_colors`(uint8 RGBA) |
| `normals (B,N,3)` | `vertex_normals` |
| `vertex_counts` / `face_counts` | 批次拆分与重组 |

顶点用 `float32`、面索引用 `int64`,与 ComfyUI 自带的 3D 加载器保持一致。反向转换时,若整条 alpha 通道都是 255 会去掉它,以 RGB 三通道输出。

## 已知取舍

- **纹理与材质不随转换。** `MESH` 上的 `texture`、`metallic_roughness`、`normal_map`、`emissive`、`material`、`tangents` 在 trimesh 几何体上没有对应位置,转换时会**明确告警**并丢弃,而不是静默丢失。
- **`process=True` 时顶点色会被丢弃。** trimesh 合并顶点时按「位置 + UV + 法线」分组,顶点色不参与分组,因此合并后无法可靠地重新对应。此时顶点色会被丢掉并给出告警(UV 和法线能正确跟随)。**想要保留顶点色,保持 `process` 关闭。** 默认关闭时转换是无损的。
- 未做贴图/UV 的展开或重算,trimesh 侧按需自行计算。

## 安装

把 `comfyui_mesh_trimesh` 整个目录放进 ComfyUI 的 `custom_nodes/` 下,然后安装依赖:

```bash
pip install -r requirements.txt
```

或在 ComfyUI-Manager 中安装本目录并让它读取 `requirements.txt`。唯一的额外依赖是 `trimesh`(`numpy`、`torch` 由 ComfyUI 自带)。

## 结构

```
comfyui_mesh_trimesh/
  __init__.py         # ComfyExtension 与 comfy_entrypoint
  nodes.py            # 两个节点类(V3 API)
  convert.py          # 纯转换逻辑,不依赖 ComfyUI,可单独测试
  requirements.txt
  pyproject.toml
  README.md
```

`convert.py` 刻意不导入任何 ComfyUI 模块:输入输出都是普通的张量/数组,因此转换逻辑可以脱离 ComfyUI 单独运行和验证。

## 许可

MIT
