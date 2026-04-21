# Minecraft Build VAE

This project trains a 3D variational autoencoder on the voxel builds in `mc_builds/`.

The dataset in this workspace already uses:

- a shared palette at `mc_builds/global_palette.json`
- voxel tensors stored as `32 x 32 x 32` integer block IDs
- `YZX` axis order
- 12,379 build samples in `mc_builds/files/*.npz`

The VAE treats each voxel as a categorical block ID, learns a latent embedding for whole builds, and can:

- reconstruct an existing build
- sample a new build from the latent prior
- write predictions back out as `.npz` files that match the dataset format
- export Java Edition structure files as compressed `.nbt`

## Install

Create a virtual environment, then install the package:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If you want GPU training, install a CUDA-enabled PyTorch build that matches your system before running `pip install -e .`.

## Train

```powershell
python -m mcvae.train `
  --data-dir mc_builds/files `
  --palette-json mc_builds/global_palette.json `
  --output-dir runs\baseline `
  --epochs 40 `
  --batch-size 2 `
  --latent-dim 256 `
  --beta 0.01 `
  --air-loss-weight 0.15
```

Notes:

- `air` is the dominant class, so the default loss downweights it slightly.
- Start with `--batch-size 1` or `2` if you hit GPU memory limits.
- Use `--limit 512` for a fast smoke test before full training.

Training writes:

- `runs/<name>/checkpoints/best.pt`
- `runs/<name>/checkpoints/last.pt`
- `runs/<name>/metrics.jsonl`
- `runs/<name>/config.json`

## Reconstruct Existing Builds

```powershell
python -m mcvae.generate reconstruct `
  --checkpoint runs\baseline\checkpoints\best.pt `
  --input mc_builds/files\build_1.npz `
  --output-dir outputs\reconstructions
```

## Sample New Builds

```powershell
python -m mcvae.generate sample `
  --checkpoint runs\baseline\checkpoints\best.pt `
  --count 16 `
  --output-dir outputs\samples
```

## Export As Structure Files

To export vanilla Java structure files instead of NPZ:

```powershell
python -m mcvae.generate sample `
  --checkpoint runs\baseline\checkpoints\best.pt `
  --count 4 `
  --output-dir outputs\structures `
  --format structure
```

You can also write both formats at once:

```powershell
python -m mcvae.generate reconstruct `
  --checkpoint runs\baseline\checkpoints\best.pt `
  --input mc_builds/files\build_1.npz `
  --output-dir outputs\reconstructions `
  --format both
```

If you want a `DataVersion` tag in the structure file, add:

```powershell
--structure-data-version <your_version_number>
```

The exporter assumes Java structure format and converts the dataset's `YZX` tensors into structure-block `[x, y, z]` positions automatically.

## Avoid Empty Samples

Generated samples can collapse to mostly air because Minecraft builds are sparse. The generator rejects trivial outputs by default with `--min-non-air 256`. You can make this stricter, or bias decoding away from air:

```powershell
python -m mcvae.generate sample `
  --checkpoint runs\baseline\checkpoints\best.pt `
  --count 12 `
  --output-dir outputs\non_empty_samples `
  --format structure `
  --device cuda `
  --min-non-air 800 `
  --air-logit-penalty 1.5
```

If samples become noisy, lower `--air-logit-penalty`. If too many are rejected, lower `--min-non-air`.

## What Comes Out

Each generated file includes:

- `blocks.npy`
- `shape.npy`
- `original_shape.npy`
- `axis_order.npy`
- `source_format.npy`
- `dataset_path.npy`

That makes the output easy to inspect with the same tools you already use for the source dataset.
