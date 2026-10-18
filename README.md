# [CoRL 2026] Official Repo for: T(R,O) Grasp++: Cross-Embodiment Dexterous Grasping in Cluttered Environments at Scale

## Install the main environment

The main code uses Python 3.10 and the `TROV2` Conda environment created by `install.sh`:

```bash
bash install.sh
conda activate TROV2

# Pyroki is needed by grasp inference and visualization.
pip install -e ./third_party/pyroki

# Install the Hugging Face CLI used below to download the dataset.
pip install -U huggingface_hub
```

### Install Isaac Gym for simulation evaluation

The evaluation needs Isaac Gym Preview 4, which uses Python 3.8. Create a separate environment and run `install_isaac.sh` inside it:

```bash
conda create -n TROV2-isaac python=3.8 -y
conda activate TROV2-isaac
bash install_isaac.sh
```

## Download Dataset and Checkpoint

Pretrained checkpoints are included in `ckpt/`. Download the validation scenes and object assets from [TROV2Dataset](https://huggingface.co/datasets/Barrybarry-Smith/TROV2Dataset) for visualization and inference (`zstd` is required to extract them):

```bash
conda activate TROV2
DATA_ROOT=/mnt/fx/TRO-1B-Dataset
hf download Barrybarry-Smith/TROV2Dataset \
  tar_shards/TRO-val-dataset.tar.zst \
  tar_shards/object_pc.tar.zst \
  tar_shards/object_urdf.tar.zst \
  --repo-type dataset --local-dir "$DATA_ROOT"

for name in TRO-val-dataset object_pc object_urdf; do
  tar -I zstd -xf "$DATA_ROOT/tar_shards/$name.tar.zst" -C "$DATA_ROOT"
done
```

We publish the training data for eight hands: Ability, Allegro, Barrett, DClaw, LEAP, Shadow, Schunk and XHand. For training, download all Zarr shards and extract them into `$DATA_ROOT/object_zarrs` (the full set requires substantial storage):

```bash
hf download Barrybarry-Smith/TROV2Dataset \
  --include 'tar_shards/object_zarrs_*.tar' \
  --repo-type dataset --local-dir "$DATA_ROOT"
mkdir -p "$DATA_ROOT/object_zarrs"
for file in "$DATA_ROOT"/tar_shards/object_zarrs_*.tar; do
  tar -xf "$file" -C "$DATA_ROOT/object_zarrs"
done
```

## Visualize the grasp synthesis

The repository includes XHand predictions in `ckpt/test_meta/test_xhand.pth`. With the validation data and object assets in place, launch the interactive Viser viewer:

```bash
python vis_grasp.py --tro-hand xhand --mode tro-only --port 8091
```

## Run grasp inference

`test.py` generates grasps for validation scenes. Run it by:

```bash
python test.py --config config/val.yaml
```

You can override configuration values on the command line:

```bash
python test.py --config config/val.yaml \
  dataset.hand_type=allegro_hand \
  test.embodiment=allegro_hand \
  test.save_path=./test/allegro_hand.pth
```

Additionally, we provide the checkpoint trained with Wuji Hand in `ckpt/nine_hand_with_wuji.pth`. You can run inference on Wuji Hand by:

```bash
python test.py --config config/val_wuji_hand.yaml
```

## Train

Edit the main settings in `config/train.yaml` before training. `dataset.sources` is a list: you can combine multiple Zarr datasets, each with its own point clouds, hands, and sampling weight.

| Setting | What it controls |
| --- | --- |
| `dataset.sources` | Add one entry per source with a unique `name` and `object_pc_root`. The `original_8_hands` source reads `data_root/object_zarrs`; other sources must set `zarr_root` (and `data_root`). `weight` sets the relative sampling frequency. |
| Source `hand_type`, `dataset.hand_type`, `model.embodiment` | Source hands, dataset hand IDs, and model hands. Selected hands must have the same index in `dataset.hand_type` and `model.embodiment`. |
| `dataset.batch_size`, `dataset.num_workers` | Samples per process and data-loading workers. |
| `vqvae.object_pretrain`, `vqvae.env_pretrain` | Pretrained object and environment encoders in `ckpt/`. |
| `train.lr`, `train.warmup_steps`, `train.total_steps` | Learning rate, warmup, and training length. |
| `train.accumulator`, `train.mixed_precision`, `model.dtype` | Gradient accumulation and precision. |
| `train.save_dir`, `train.save_every`, `train.log_every` | Output directory and checkpoint/log intervals. |
| `train.resume_from`, `train.load_optimizer` | Checkpoint to resume from; set `load_optimizer: true` to restore optimizer and scheduler state. |

For example, append a single-hand dataset to the existing `dataset.sources` list:

```yaml
    - name: extra_xhand
      data_root: /path/to/extra
      zarr_root: /path/to/extra/object_zarrs
      object_pc_root: /path/to/extra/object_pc
      force_hand_type: xhand
      weight: 0.5
```

Launch training with:

```bash
accelerate launch train.py --config config/train.yaml
```

Checkpoints and TensorBoard logs appear in `<train.save_dir>/ckpt/` and `<train.save_dir>/tensorboard/`. If using `train.sh`, first replace its machine-specific acceleration config path.

## Simulation evaluation

Save predictions as `<experiment>/val/<hand>/step_<step>.pth`. Set the data paths, steps, and hands in `validation/convert_before_eval.py`, then prepare simulation inputs:

```bash
conda activate TROV2
python validation/convert_before_eval.py
```

The converter reads `$DATA_ROOT/object_urdf` and writes to `<experiment>/sim/step_<step>/<hand>/`. For the lift test, set the root, steps, hands, and GPUs in `validation/val_lift.py`. Also change its subprocess path from `validation/isaac_main.py` to `validation/validation/isaac_main.py`:

```bash
conda activate TROV2-isaac
python validation/val_lift.py
```

Check collisions with:

```bash
conda activate TROV2
python validation/collision_check.py \
  --sim-root /path/to/experiment/sim \
  --pred-root /path/to/experiment/val \
  --steps step_820000 \
  --robot-names xhand
```

