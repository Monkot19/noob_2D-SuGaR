# hot_cell/39：2DGS 与 Metric3D 深度/法线先验实验记录

## 1. 实验目标

评估热室场景中，面对铁皮高光和强反射时，Metric3D 单目深度/法线先验是否能改善 2D Gaussian Splatting 的几何与渲染质量。

本记录区分两类问题：

1. **训练期法线先验损失是否有效**：必须使用同一代码、同一 10 万点初始化和同一随机种子，只改变 `lambda_normal_prior`。
2. **完整深度/法线初始化方案是否有效**：允许 Metric3D 改变点的初始旋转、增加深度生成点，并使用训练期法线先验。该实验代表完整方法，但不能单独归因到某一个因素。

## 2. 数据集概况

| 项目 | 内容 |
|---|---|
| 原始数据路径 | `/root/autodl-tmp/datasets/hot_cell/39` |
| 图像数量 | 8 |
| 相机模型 | `PINHOLE` |
| COLMAP tracks | 无；`images.txt` 中每张图的 observations 均为 0 |
| `points3D.txt/bin` | 均不存在 |
| 原始 PLY 点数 | 1,572,864 |
| PLY 属性 | `x,y,z,nx,ny,nz,red,green,blue` |
| `points3D_all.npy` | `(8,384,512,3)` |
| `pointsColor_all.npy` | `(8,384,512,3)` |
| `confidence.npy` | `(8,196608)` |
| 数据性质 | 稠密逐像素世界坐标导出，不是传统稀疏 COLMAP 三角化点云 |

`1,572,864 = 8 × 384 × 512`，且稠密 XYZ 与 PLY 抽样比较的最大误差为 0。图像与稠密颜色图的归一化 MAE 为 0.0046–0.0054，说明两者顺序和内容对齐。

## 3. 代码与随机性

| 项目 | 内容 |
|---|---|
| 仓库 | `Monkot19/noob_2D-SuGaR` |
| 正式实验代码提交 | `be9d131 Align prior sampling resolutions` |
| Trackless 支持提交 | `1873f51 Support trackless dense priors` |
| 训练随机种子 | Python、NumPy、PyTorch 均固定为 0（`safe_state`） |
| 10 万点抽样种子 | NumPy `default_rng(39)` |
| 分辨率 | `-r 2` |
| 迭代数 | 30,000 |
| cluster prune | 关闭：`--cluster_prune_iterations -1` |

说明：固定随机种子提高可复现性，但 CUDA 算子仍不保证跨硬件、驱动和库版本逐位一致。

## 4. 已有旧基线：普通 2DGS（参考实验 L0）

此实验在另一台服务器的 `/root/autodl-tmp/noob_2dgs` 中运行。

### 4.1 初始化数据

- 数据集：`/root/autodl-tmp/datasets/hot_cell_39_2dgs_baseline`
- 从 1,572,864 个 PLY 顶点中，用种子 39 无放回抽样 100,000 点。
- 原上传目录保持不变。
- 旧抽样脚本没有保存 indices；若要精确复用，应重新生成并保存 `sample_indices_seed39.npy`。

### 4.2 训练参数

```bash
python train.py \
  -s /root/autodl-tmp/datasets/hot_cell_39_2dgs_baseline \
  -m /root/autodl-tmp/outputs/hot_cell_39_rgb_only_30k_v1 \
  -r 2 \
  --iterations 30000 \
  --depth_ratio 0 \
  --lambda_dist 0 \
  --lambda_normal 0.05 \
  --opacity_cull 0.05 \
  --densify_from_iter 500 \
  --densify_until_iter 15000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.0002 \
  --test_iterations 7000 15000 30000 \
  --save_iterations 7000 15000 30000 \
  --checkpoint_iterations 7000 15000 30000
```

### 4.3 解释边界

L0 只能作为历史参考，不能与完整先验实验 C1 构成严格 A/B，原因包括：

- L0 初始化为 100,000 点，C1 初始化为 1,578,311 点；
- L0 与 C1 使用不同代码仓库；
- L0 没有 Metric3D 法线损失，C1 同时改变了初始旋转、新增点和训练损失；
- L0 的训练指标和产物统计尚未补录。

## 5. 已完成实验：完整深度/法线先验（C1）

### 5.1 路径与状态

| 项目 | 内容 |
|---|---|
| 数据集 | `/root/autodl-tmp/datasets/hot_cell/39` |
| 输出目录 | `/root/autodl-tmp/outputs/hot_cell_39_depth_normal_30k_v1` |
| 状态 | 完成，`TRAIN_RC=0` |
| 训练耗时 | 6 分 35 秒 |
| 平均速度 | 75.92 iter/s |

### 5.2 方法组成

- 使用 `points3D_all.npy` 和置信度估计每个视角的 Metric3D 深度尺度；
- 保留原始 PLY 的 1,572,864 个位置和颜色；
- 原始 PLY 的 1,572,864 个法线全部无效/为零，因此用有效的 Metric3D 法线初始化这些点的旋转；
- 额外生成 5,447 个 Metric3D 深度引导点；
- 初始总点数为 1,578,311；
- 从第 1 次迭代开始使用 `lambda_normal_prior=0.1` 的 Metric3D 法线损失；
- `lambda_normal=0.05` 的 2DGS 原生法线一致性项从第 7001 次迭代开始启用；
- `lambda_dist=0`，因此没有 distortion loss；
- 当前实现没有逐像素 Metric3D depth loss，深度只参与初始化尺度和新增点生成。

### 5.3 完整命令

```bash
python train.py \
  -s /root/autodl-tmp/datasets/hot_cell/39 \
  -m /root/autodl-tmp/outputs/hot_cell_39_depth_normal_30k_v1 \
  -r 2 \
  --iterations 30000 \
  --initialization_prior monocular_depth_normal \
  --depth_ratio 0 \
  --lambda_dist 0 \
  --lambda_normal 0.05 \
  --lambda_normal_prior 0.1 \
  --opacity_cull 0.05 \
  --densify_from_iter 500 \
  --densify_until_iter 15000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.0002 \
  --cluster_prune_iterations -1 \
  --test_iterations 7000 15000 30000 \
  --save_iterations 7000 15000 30000 \
  --checkpoint_iterations 7000 15000 30000 \
  --skip_2dgs_render \
  --skip_sugar_refine
```

### 5.4 训练指标

以下均为训练视角指标，并非独立测试集指标。

| Iteration | Train L1 | Train PSNR |
|---:|---:|---:|
| 7,000 | 0.00247485 | 48.5635 dB |
| 15,000 | 0.00201026 | 49.6099 dB |
| 30,000 | 0.00135041 | 53.7983 dB |

最终进度栏：

| 项目 | 数值 |
|---|---:|
| Photometric `Loss` EMA | 0.00194 |
| Distortion loss EMA | 0.00000 |
| 原生 normal loss EMA | 0.00610 |
| 最终 Gaussian 数量 | 321,024 |

进度栏中的 `normal` 只表示原生 2DGS 法线一致性项，不包含 Metric3D `normal_prior_loss`。

### 5.5 深度尺度诊断

| 图像 | Scale | Relative MAD |
|---|---:|---:|
| `...095` | 0.0721793 | 0.1182 |
| `...096` | 0.0620185 | 0.0580 |
| `...097` | 0.0576815 | 0.0440 |
| `...098` | 0.0746325 | 0.1092 |
| `...099` | 0.0652717 | 0.0162 |
| `...100` | 0.0645210 | 0.0246 |
| `...101` | 0.0686025 | **0.3173** |
| `...102` | **0.0351145** | 0.0669 |

重点检查 `...101` 的高 MAD，以及 `...102` 相对其他视角明显偏低的尺度。如果结果出现局部拉伸、双层表面或漂浮点，应优先检查这两个视角。

### 5.6 产物

```text
/root/autodl-tmp/outputs/hot_cell_39_depth_normal_30k_v1/train.log
/root/autodl-tmp/outputs/hot_cell_39_depth_normal_30k_v1/chkpnt7000.pth
/root/autodl-tmp/outputs/hot_cell_39_depth_normal_30k_v1/chkpnt15000.pth
/root/autodl-tmp/outputs/hot_cell_39_depth_normal_30k_v1/chkpnt30000.pth
/root/autodl-tmp/outputs/hot_cell_39_depth_normal_30k_v1/point_cloud/iteration_7000/point_cloud.ply
/root/autodl-tmp/outputs/hot_cell_39_depth_normal_30k_v1/point_cloud/iteration_15000/point_cloud.ply
/root/autodl-tmp/outputs/hot_cell_39_depth_normal_30k_v1/point_cloud/iteration_30000/point_cloud.ply
```

## 6. 下一组严格受控 A/B：训练期法线先验

### 6.1 实验变量

| 实验 | 初始化点 | `initialization_prior` | `lambda_normal_prior` | 目的 |
|---|---:|---|---:|---|
| A1：No prior loss | 同一 100,000 点 | 空字符串 | 0.0 | 对照组 |
| B1：Normal prior loss | 同一 100,000 点 | 空字符串 | 0.1 | 只测试训练期 Metric3D 法线损失 |
| C1：Full depth+normal | 1,578,311 点 | `monocular_depth_normal` | 0.1 | 已完成的完整方法，非单变量对照 |

A1 与 B1 使用同一份数据目录、同一 PLY、同一 Metric3D 文件、同一代码提交和同一训练随机种子。唯一训练变量是 `lambda_normal_prior`。

注意：必须显式传入 `--initialization_prior ""`，因为当前 2D-SuGaR 默认值是 `monocular_depth_normal`。

### 6.2 创建统一的 100k 数据集

建议目录：

```text
/root/autodl-tmp/datasets/hot_cell_39_ab100k_seed39
```

建议在重新抽样时保存 indices，以确保之后可以精确复用：

```bash
SRC=/root/autodl-tmp/datasets/hot_cell/39
DST=/root/autodl-tmp/datasets/hot_cell_39_ab100k_seed39

if [ -e "$DST" ]; then
    echo "STOP: $DST already exists"
else
    mkdir -p "$DST/sparse/0"
    cp -a "$SRC/images" "$DST/"
    cp -a "$SRC/estimated_dense_depth" "$DST/"
    cp -a "$SRC/estimated_dense_normal" "$DST/"

    cp -a \
      "$SRC/sparse/0/cameras.bin" \
      "$SRC/sparse/0/images.bin" \
      "$SRC/sparse/0/cameras.txt" \
      "$SRC/sparse/0/images.txt" \
      "$DST/sparse/0/"

    python - <<'PY'
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

src = Path("/root/autodl-tmp/datasets/hot_cell/39/sparse/0/points3D.ply")
out_dir = Path("/root/autodl-tmp/datasets/hot_cell_39_ab100k_seed39/sparse/0")
dst = out_dir / "points3D.ply"

vertices = PlyData.read(str(src))["vertex"].data
rng = np.random.default_rng(39)
indices = np.sort(rng.choice(len(vertices), size=min(100_000, len(vertices)), replace=False))
sampled = vertices[indices].copy()

PlyData([PlyElement.describe(sampled, "vertex")], text=False).write(str(dst))
np.save(out_dir / "sample_indices_seed39.npy", indices)

print("source_points:", len(vertices))
print("sampled_points:", len(sampled))
print("indices:", out_dir / "sample_indices_seed39.npy")
print("output:", dst)
print("properties:", sampled.dtype.names)
PY
fi
```

### 6.3 A1 与 B1 的共同参数

- 仓库：`/root/autodl-tmp/2D_SuGaR`
- 数据集：`/root/autodl-tmp/datasets/hot_cell_39_ab100k_seed39`
- 初始化：同一 100,000 点 PLY；随机旋转初始化；不新增 Metric3D 深度点；
- 训练种子：0；
- `lambda_normal=0.05`；
- `lambda_dist=0`；
- `cluster_prune_iterations=-1`；
- 保存迭代：7k、15k、30k；
- 跳过渲染、网格和 SuGaR refine，先比较 point cloud 与训练日志。

### 6.4 A1 命令：无 Metric3D 训练损失

```bash
python train.py \
  -s /root/autodl-tmp/datasets/hot_cell_39_ab100k_seed39 \
  -m /root/autodl-tmp/outputs/hot_cell_39_ab100k_no_prior_30k_v1 \
  -r 2 \
  --iterations 30000 \
  --initialization_prior "" \
  --depth_ratio 0 \
  --lambda_dist 0 \
  --lambda_normal 0.05 \
  --lambda_normal_prior 0 \
  --opacity_cull 0.05 \
  --densify_from_iter 500 \
  --densify_until_iter 15000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.0002 \
  --cluster_prune_iterations -1 \
  --test_iterations 7000 15000 30000 \
  --save_iterations 7000 15000 30000 \
  --checkpoint_iterations 7000 15000 30000 \
  --skip_2dgs_render \
  --skip_sugar_refine
```

### 6.5 B1 命令：只增加 Metric3D 法线损失

```bash
python train.py \
  -s /root/autodl-tmp/datasets/hot_cell_39_ab100k_seed39 \
  -m /root/autodl-tmp/outputs/hot_cell_39_ab100k_normal_prior_30k_v1 \
  -r 2 \
  --iterations 30000 \
  --initialization_prior "" \
  --depth_ratio 0 \
  --lambda_dist 0 \
  --lambda_normal 0.05 \
  --lambda_normal_prior 0.1 \
  --opacity_cull 0.05 \
  --densify_from_iter 500 \
  --densify_until_iter 15000 \
  --densification_interval 100 \
  --densify_grad_threshold 0.0002 \
  --cluster_prune_iterations -1 \
  --test_iterations 7000 15000 30000 \
  --save_iterations 7000 15000 30000 \
  --checkpoint_iterations 7000 15000 30000 \
  --skip_2dgs_render \
  --skip_sugar_refine
```

## 7. 比较与验收规则

### 7.1 定量指标

- 记录 7k、15k、30k 的 train L1 与 train PSNR；
- 后续增加统一渲染与独立测试视角后，再记录 test PSNR、SSIM、LPIPS；
- 记录初始和最终 Gaussian 数量、训练时间和显存峰值；
- C1 的 53.7983 dB 是训练 PSNR，不代表新视角泛化质量。

### 7.2 定性检查

统一检查以下区域：

- 铁皮墙高光带是否产生漂浮面或双层面；
- 摄像头、支架和管线边缘是否更锐利；
- 平整墙面是否更平滑且无波纹；
- `...101` 与 `...102` 对应视角附近是否发生尺度不一致；
- 点云离群点、薄片噪声和空洞数量；
- 新视角渲染中高光是否被错误固化为几何。

### 7.3 可以回答的问题

- A1 vs B1：训练期 Metric3D 法线损失的净作用；
- B1 vs C1：完整初始化带来的增量现象，但由于点数和旋转初始化不同，仍不是严格单变量实验；
- L0 vs A1：可以观察代码库差异，但不能只归因到先验。

## 8. 待补录

- [ ] AutoDL GPU、显存、驱动、CUDA 和 PyTorch 版本；
- [ ] L0 普通 2DGS 的 7k/15k/30k 指标、最终点数和耗时；
- [ ] A1 受控无先验结果；
- [ ] B1 受控法线先验结果；
- [ ] 三组 iteration 30000 点云截图；
- [ ] 同一视角的 RGB/normal/depth 渲染对比；
- [ ] 独立测试视角或 leave-one-out 指标；
- [ ] 对 `...101` 和 `...102` 尺度异常的敏感性实验。

