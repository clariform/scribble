# scribble

A small, clean **MNIST DCGAN** project that focuses on the fundamentals:

- dataset loading + batching
- GAN training loop (Generator / Discriminator)
- training stability basics (BCEWithLogitsLoss, Adam betas)
- checkpoints + reproducible sampling
- sample grids over time + TensorBoard loss curves

This repo is intentionally **training-first** (no API yet). It’s meant to be a tight, interview-friendly “I can train and ship artifacts” demo.

---

## What you get

### Artifacts (produced during training)

- **checkpoints** (`.pt`)
- **sample grids** (`.png`) saved every *N* steps
- **TensorBoard logs** (loss curves, sample previews)

### Deterministic sampling

A CLI command that loads a checkpoint and generates a seeded grid so results are reproducible.

---

## Repo layout

```text
scribble/
  configs/
    train.yaml

  src/
    scribble/
      cli.py          # CLI entrypoint
      config.py       # YAML config loader
      data.py         # MNIST dataloader
      models.py       # DCGAN-ish G/D
      train.py        # training loop + artifact writing
      sample.py       # deterministic sampler
      utils.py        # seeding + run paths
```

Runtime outputs go to WHISK paths if available, otherwise to local `./outputs/`.

---

## Requirements

- Linux with NVIDIA GPU (recommended)
- CUDA-enabled PyTorch (or CPU fallback)
- Python managed via **uv**

This is designed to run well inside a CUDA dev container (example: your `neuron` container).

---

## Environment variables (WHISK)

If these are set, artifacts will go to your centralized mount:

```bash
WHISK_ML_DATASETS=/mnt/whisk/work/ml/datasets
WHISK_ML_EXPERIMENTS=/mnt/whisk/work/ml/experiments
WHISK_ML_MODELS=/mnt/whisk/work/ml/models
WHISK_ML_LOGS=/mnt/whisk/work/ml/logs
```

If they are **not** set, artifacts will be written under `./outputs/`.

Optional:

```bash
SCRIBBLE_RUN_NAME=mnist_dcgan
```

---

## Setup (uv)

From the repo root:

```bash
uv sync
```

If you need CUDA wheels for PyTorch, use the appropriate PyTorch index for your CUDA version (example shown for cu124):

```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Train

Train using the default config:

```bash
uv run scribble train --config configs/train.yaml
```

During training you’ll see:

- loss curves in TensorBoard
- grids saved as PNGs in the run folder
- periodic checkpoints

---

## View training in TensorBoard

If you’re using WHISK, logs will land under:

- `$WHISK_ML_LOGS/scribble/<run_name>/<timestamp>/`

Otherwise, they’ll be local.

Run TensorBoard (adjust the logdir depending on your environment):

```bash
uv run tensorboard --logdir /mnt/whisk/work/ml/logs/scribble
```

Local fallback example:

```bash
uv run tensorboard --logdir outputs
```

---

## Sample from a checkpoint (deterministic)

Generate a seeded sample grid from a checkpoint:

```bash
uv run scribble sample --ckpt /path/to/final.pt --out outputs/grid.png --seed 42 --n 64
```

Tips:

- use `--n 64` (8×8) or `--n 100` (10×10) for nice grids
- keep the same `--seed` to reproduce exact output

---

## Where outputs go

Each training run creates a timestamped folder:

```text
<run_root>/
  samples/
    step_0000000.png
    step_0000500.png
    ...
  checkpoints/
    ckpt_step_0002000.pt
    ...
    final.pt
```

Where `<run_root>` resolves to one of:

- **preferred**: `$WHISK_ML_EXPERIMENTS/scribble/<run_name>/<timestamp>/`
- **fallback**: `./outputs/<run_name>/<timestamp>/`

---

## Configuration

Training config lives in:

- `configs/train.yaml`

Key knobs:

- `train.epochs`
- `train.lr`, `train.betas`
- `train.sample_every_steps`
- `train.ckpt_every_steps`
- `train.amp`
- `model.z_dim`, `model.feature_g`, `model.feature_d`

---

## Notes on choices (why this is stable enough)

- **BCEWithLogitsLoss**: discriminator outputs raw logits (no sigmoid in the model)
- **Adam betas (0.5, 0.999)**: common DCGAN default
- **Normalization to [-1, 1]** + generator `tanh` output: standard GAN pairing
- **Fixed latent**: keeps sample grids comparable across steps

---

## Suggested “portfolio deliverables”

If you want this to look great to a reviewer, keep these in the repo (small + curated):

- `report.md`
  - training settings
  - loss curves screenshot
  - 2–3 best grids
  - 2–3 failure cases + what changed
- `assets/` (optional)
  - a GIF made from sample grids

---

## License

MIT (see repository license).

