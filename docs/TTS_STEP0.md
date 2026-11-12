# 第零步：评估基础设施（test-time scaling）

本分支 `step0-eval-infra` 让推理与评估流水线支持“每个场景 N 个候选”，并保留每个候选的 verifier 信号与分项标签，用来画 pass@N / best-of-N 曲线。

## 改了什么

| 文件 | 改动 |
| --- | --- |
| `model/troe_graph.py` | `inference(batch, noise_seeds=None)`：给出每个候选的整数种子时，该候选的全部噪声（初始噪声 + 每步 DDIM 噪声）都由这个种子生成，与 batch 大小、分块、顺序无关。不传则保持原行为。 |
| `test.py` | 新配置 `test.num_samples` / `test.seed` / `test.per_sample_seed` / `test.chunk_size`；每个场景复制 N 次；种子 = `crc32(seed/物体/场景/样本)`；输出中每个候选新增 `sample_ids`、`noise_seeds`、`ik_pos_err`（米）、`ik_rot_err`（弧度）、`ik_select_err`（多起点 IK 选中解的残差）。`valid_scene_ids` 仍是场景下标（N > 1 时重复 N 次），下游无需改。打印的 batch IK 误差与原来数值一致。 |
| `dataset/TROFilteredDatasetTestV2.py` | 物体列表按名字排序（原来是 `os.listdir` 的随机顺序）；`dataset.max_objects` 只取前 N 个物体，用于冒烟测试。 |
| `validation/convert_before_eval.py` | 硬编码路径改为命令行参数；`val_data.json` 里多存 `sample_ids`。 |
| `validation/val_lift.py` | 硬编码路径改为命令行参数；`--bsz`（每个 Isaac 进程的物体数）、`--gpus`。物体列表改为“子目录”，原来的 `"sr" not in file` 会误删名字里含 `sr` 的物体，所以评估的物体集合可能与论文略有不同。 |
| `validation/collision_check.py` | 每个物体的结果里多存 `scene_ids`、`sample_ids` 和每个候选的连续碰撞统计（`collision_stats`：`close_ratio`、`min_surface_distance` 等，可作 4a 的独立指标）。 |
| `tools/tts_curves.py` | 新脚本：pass@n（无偏估计）、各 verifier 的 best-of-n（对所有 n 子集的精确期望）、追回比例、失败分解；分别对 `success` / `lifted` / `non_collision` 三种标签。 |

## 运行顺序

以下用 `EXP` 表示实验根目录（放在持久盘上），`TAG` 表示一次运行的名字，`DATA` 表示数据根目录。

```bash
export EXP=/mnt/pfs/5w88n7/<你的名字>/tro_exp
export DATA=/mnt/pfs/5w88n7/<你的名字>/data/TRO-1B-Dataset
export TAG=tts_N16_seed0
export HAND=xhand
```

**1. 推理（TROV2 环境，仓库根目录）**

```bash
python test.py --config config/val.yaml \
  dataset.data_root=$DATA/TRO-val-dataset dataset.hand_type=$HAND test.embodiment=$HAND \
  test.num_samples=16 test.seed=0 test.chunk_size=2048 \
  test.save_path=$EXP/val/$HAND/$TAG.pth
```

冒烟测试先加 `dataset.max_objects=5`。显存不够就调小 `test.chunk_size` 或 `dataset.batch_size`。

**2. 转成 Isaac 输入（TROV2 环境，`validation/` 目录）**

```bash
cd validation
python convert_before_eval.py --exp-root $EXP --steps $TAG --robot-names $HAND \
  --val-data-root $DATA/TRO-val-dataset --object-urdf-root $DATA/object_urdf
```

需要 `trimesh` 和 `pymeshlab`（`install.sh` 没装：`pip install trimesh pymeshlab`）。

**3. Isaac 提起测试（TROV2-isaac 环境，`validation/` 目录）**

```bash
python val_lift.py --sim-root $EXP/sim --steps $TAG --robot-names $HAND --gpus 0 1 2 3 4 5 6 7 --bsz 10
```

每个物体会产生（场景数 × N）个仿真环境，N 大时把 `--bsz` 调小。需要 `termcolor`。

**4. 点云碰撞检查（TROV2 环境，`validation/` 目录）**

```bash
python collision_check.py --sim-root $EXP/sim --pred-root $EXP/val --steps $TAG --robot-names $HAND
```

输出 `$EXP/sim/$TAG/$HAND/sr_final_surface_pc.pt`（每个候选的 `lifted` / `non_collision` / `success`）。

**5. 画曲线（任意有 torch、numpy、matplotlib 的环境，仓库根目录）**

```bash
cd ..
python tools/tts_curves.py --pred $EXP/val/$HAND/$TAG.pth \
  --labels $EXP/sim/$TAG/$HAND/sr_final_surface_pc.pt \
  --out $EXP/analysis/${HAND}_$TAG.json --plot
```

## 怎么读结果

- `single`：随机取一个候选的期望成功率，对应论文的单次成功率。
- `pass@n`：n 个候选里至少一个成功的概率，即任何 verifier 的上限（RQ1）。
- `best_of_n[verifier]`：该 verifier 从 n 个候选里挑出的那个的期望成功率；括号里是追回比例。
- `oracle_noncollision*`：用的是评判标准自己的碰撞检查，属于部分 oracle，只作参考线，不能当作 verifier 的结论。
- 三种标签：`success` 是论文指标；`lifted` 只看 Isaac 提起；`non_collision` 只看点云碰撞。RQ2 的主要结论看 `lifted` 一栏（碰撞已由评判标准单独处理）。
- `breakdown`：候选层面的失败分解（碰撞 / 仅提起失败），以及“N 个全失败”的场景占比（决定是否做干预点 2）。

成功率的平均方式与仓库一致：先在物体内对场景取平均，再对物体取平均。

## 复现 baseline

- `test.num_samples=1` 对应论文的单次设置；`test.per_sample_seed=false` 可完全回到原来的全局随机数行为（结果在统计上相同，逐样本不同）。
- `num_samples=4` 时 `pass@4` 对应论文的 98.46%。
