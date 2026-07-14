# hot_cell/39：普通 2DGS 与 2D-SuGaR 对比实验记录

## 1. 实验目的

只回答一个问题：在铁皮高光、强反射的热室场景中，2D-SuGaR 相比普通 2DGS 是否有改善。

正式对比只设两组：

| 组别 | 算法 | 共同输入 |
|---|---|---|
| P0 | 普通 2DGS | seed=39 从原始 PLY 抽出的同一批 100,000 点 |
| S1 | 2D-SuGaR | 同一批 100,000 点，并使用 Metric3D 深度/法线先验 |

这是一项“算法整体效果”对比，不要求两种算法内部每一步完全相同。S1 根据深度先验额外生成约 5,000 个点、用法线先验初始化并参与训练，这些属于 2D-SuGaR 方法本身，应保留并在结果中记录。

## 2. 数据集概况

| 项目 | 内容 |
|---|---|
| 原始数据路径 | `/root/autodl-tmp/datasets/hot_cell/39` |
| 图像数量 | 8 |
| 相机模型 | `PINHOLE` |
| COLMAP tracks | 无；每张图 observations 均为 0 |
| 原始 PLY 点数 | 1,572,864 |
| `points3D_all.npy` | `(8,384,512,3)` |
| `pointsColor_all.npy` | `(8,384,512,3)` |
| `confidence.npy` | `(8,196608)` |

`1,572,864 = 8 × 384 × 512`。这是逐像素稠密世界坐标，不是传统稀疏 COLMAP 三角化点云。稠密 XYZ 与原始 PLY 对齐，图像与稠密颜色图也已验证顺序一致。

## 3. 已完成的预实验（不作为最终公平对比）

2D-SuGaR 已在全部 1,572,864 个源点上完成一次 30k 训练：

| 项目 | 结果 |
|---|---|
| 输出目录 | `/root/autodl-tmp/outputs/hot_cell_39_depth_normal_30k_v1` |
| 状态 | 完成，`TRAIN_RC=0` |
| 初始点数 | 1,572,864 源点 + 5,447 深度生成点 = 1,578,311 |
| 最终点数 | 321,024 |
| 耗时 | 6 分 35 秒 |
| 7k train PSNR | 48.5635 dB |
| 15k train PSNR | 49.6099 dB |
| 30k train PSNR | 53.7983 dB |

这次结果证明 trackless 数据能够训练，也证明法线先验已经生效；但普通 2DGS 使用的是 10 万点，因此这次 157 万点结果只作为预实验，不能直接下结论。

当前实现中：

- Metric3D 深度用于估计尺度和生成新增点；
- Metric3D 法线用于源点旋转初始化，并以 `lambda_normal_prior=0.1` 参与训练；
- 当前没有逐像素 Metric3D depth loss；
- `lambda_normal=0.05` 是 2DGS 原生法线一致性项；
- `lambda_dist=0`，未使用 distortion loss。

## 4. 创建两种算法共用的 100k 数据集

目标目录：

```text
/root/autodl-tmp/datasets/hot_cell_39_common100k_seed39
```

下面的脚本使用与旧普通 2DGS baseline 完全相同的 `default_rng(39)` 和排序方式抽样，同时保存索引。`points3D_sample_indices.npy` 仅供 2D-SuGaR 将 10 万点映射回稠密先验；普通 2DGS 会忽略它。

```bash
set -euo pipefail

SRC=/root/autodl-tmp/datasets/hot_cell/39
DST=/root/autodl-tmp/datasets/hot_cell_39_common100k_seed39

if [ -e "$DST" ]; then
    echo "STOP: $DST already exists"
    exit 1
fi

mkdir -p "$DST/sparse/0"
cp -a "$SRC/images" "$DST/"
cp -a "$SRC/estimated_dense_depth" "$DST/"
cp -a "$SRC/estimated_dense_normal" "$DST/"

for f in \
    cameras.bin images.bin cameras.txt images.txt \
    points3D_all.npy pointsColor_all.npy confidence.npy \
    confidence_dsp.npy non_scaled_focals.npy; do
    if [ -f "$SRC/sparse/0/$f" ]; then
        cp -a "$SRC/sparse/0/$f" "$DST/sparse/0/"
    fi
done

python - <<'PY'
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

src = Path("/root/autodl-tmp/datasets/hot_cell/39/sparse/0/points3D.ply")
out_dir = Path("/root/autodl-tmp/datasets/hot_cell_39_common100k_seed39/sparse/0")
dst = out_dir / "points3D.ply"

vertices = PlyData.read(str(src))["vertex"].data
rng = np.random.default_rng(39)
indices = np.sort(
    rng.choice(len(vertices), size=min(100_000, len(vertices)), replace=False)
)
sampled = vertices[indices].copy()

PlyData([PlyElement.describe(sampled, "vertex")], text=False).write(str(dst))
np.save(out_dir / "points3D_sample_indices.npy", indices)

print("source_points:", len(vertices))
print("sampled_points:", len(sampled))
print("first_last_indices:", int(indices[0]), int(indices[-1]))
print("output:", dst)
print("indices:", out_dir / "points3D_sample_indices.npy")
print("properties:", sampled.dtype.names)
PY
```

预期关键输出为 `source_points: 1572864` 和 `sampled_points: 100000`。

## 5. P0：普通 2DGS

普通 2DGS 仓库使用上一步创建的共同数据集：

```bash
set -euo pipefail

cd /root/autodl-tmp/noob_2dgs

DATA=/root/autodl-tmp/datasets/hot_cell_39_common100k_seed39
OUT=/root/autodl-tmp/outputs/hot_cell_39_common100k_2dgs_30k_v1

if [ -e "$OUT" ]; then
    echo "STOP: $OUT already exists"
    exit 1
fi

mkdir -p "$OUT"
export OMP_NUM_THREADS=8

python train.py \
  -s "$DATA" \
  -m "$OUT" \
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
  --checkpoint_iterations 7000 15000 30000 \
  2>&1 | tee "$OUT/train.log"
```

如果旧输出 `/root/autodl-tmp/outputs/hot_cell_39_rgb_only_30k_v1` 确实由同一个 seed=39 脚本生成，可以保留为 P0；但应先确认其输入 PLY 与新共同数据集的 PLY 完全一致。

## 6. S1：2D-SuGaR

先跑 100 次 smoke test。预期日志包含：

- `Using 100000 sampled source points via ...points3D_sample_indices.npy`；
- `Dense XYZ/PLY validation passed`；
- 初始总点数约 10.5 万，而不是 157 万；
- `TRAIN_RC=0`。

```bash
cd /root/autodl-tmp/2D_SuGaR
conda activate 2d-sugar

DATA=/root/autodl-tmp/datasets/hot_cell_39_common100k_seed39
OUT=/root/autodl-tmp/outputs/hot_cell_39_common100k_2dsugar_smoke100_v1

if [ -e "$OUT" ]; then
    echo "STOP: 输出目录已存在：$OUT"
else
    mkdir -p "$OUT"
    export OMP_NUM_THREADS=8

    python train.py \
      -s "$DATA" \
      -m "$OUT" \
      -r 2 \
      --iterations 100 \
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
      --test_iterations -1 \
      --save_iterations 100 \
      --checkpoint_iterations 100 \
      --skip_2dgs_render \
      2>&1 | tee "$OUT/train.log"

    echo "TRAIN_RC=${PIPESTATUS[0]}"
fi
```

smoke test 成功后，只需把输出目录改为 `hot_cell_39_common100k_2dsugar_30k_v1`，把迭代与保存节点恢复为 7k/15k/30k，即可跑正式 S1。

## 7. 对比规则

两组保持一致：输入图像、相机、100k 源点、分辨率 `-r 2`、30k 迭代、densification 参数、`lambda_normal=0.05`、`lambda_dist=0`。

记录以下结果：

- 7k、15k、30k 的 train L1 和 train PSNR；
- 初始和最终 Gaussian 数量；
- 训练时间与显存峰值；
- 同一视角的 RGB、depth、normal 渲染；
- 铁皮高光带是否出现漂浮面、双层面或错误几何；
- 摄像头、支架、管线边缘是否更清晰；
- 平整墙面是否更平滑，离群点和孔洞是否减少。

训练 PSNR 只反映对训练图像的拟合，不能单独代表新视角质量。由于只有 8 张图，最终结论应以同视角可视化和几何质量为主；如需严格评价泛化，再补 leave-one-out 实验。

## 8. 当前结论状态

- [x] 原始 157 万点 2D-SuGaR 预实验成功；
- [x] 2D-SuGaR 已支持通过采样索引读取同一批 100k 源点；
- [ ] 创建共同 100k 数据集；
- [ ] S1 100 次 smoke test；
- [ ] P0 普通 2DGS 30k；
- [ ] S1 2D-SuGaR 30k；
- [ ] 汇总指标与同视角可视化，判断高光场景是否改善。
