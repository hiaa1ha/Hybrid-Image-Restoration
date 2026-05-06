# ARANet-B — Image Restoration via Attention Residual Architecture

ARANet-B is a U-Net-style image restoration network that combines **depthwise-separable convolutions**, **grouped residual blocks**, and dual attention modules — a **Spatial Attention Module (SAM)** and a **Frequency Attention Module (FAM)** — at the bottleneck. It is designed for blind Gaussian image denoising but can be adapted to other degradation tasks.

---


## Installation

```bash
pip install -r requirements.txt
```

---

## Training

### 1. Prepare training data

Place your high-quality training images (PNG, JPG, BMP, etc.) in:

```
trainsets/trainH/
```

The dataset loader synthesises noisy inputs on the fly — you only need clean images.

Popular training sets: **BSD400**, **WED (4744 images)**, or **DIV2K**.

### 2. Configure options

Open `options/train_aranet_b.json` and adjust as needed:

| Key | Description | Default |
|-----|-------------|---------|
| `gpu_ids` | List of GPU IDs to use | `[0]` |
| `n_channels` | `1` = grayscale, `3` = color | `3` |
| `sigma` | Noise level range `[min, max]` during training | `[0, 50]` |
| `sigma_test` | Noise level used at validation time | `25` |
| `H_size` | Training patch size | `128` |
| `dataloader_batch_size` | Batch size | `64` |
| `G_optimizer_lr` | Initial learning rate | `1e-4` |
| `checkpoint_save` | Save model every N iterations | `5000` |
| `checkpoint_test` | Evaluate on test set every N iterations | `5000` |
| `checkpoint_print` | Print training log every N iterations | `200` |

### 3. Run training

```bash
python main_train_aranet_b.py --opt options/train_aranet_b.json
```

**Multi-GPU (distributed) training:**

```bash
python -m torch.distributed.launch \
    --nproc_per_node=<NUM_GPUS> \
    main_train_aranet_b.py \
    --opt options/train_aranet_b.json \
    --dist True
```

### Training outputs

All outputs are written to `denoising/aranet_b/` (controlled by `task` + `root` in the JSON):

```
denoising/aranet_b/
├── models/           # Saved checkpoints  (aranet_b_G_<iter>.pth)
├── images/           # Validation images saved during training
├── options/          # Copy of the resolved option JSON
└── log/train.log     # Training log
```

---

## Resuming training

Training resumes automatically from the latest checkpoint in `denoising/aranet_b/models/`. Just re-run the same command — no extra flags needed.

---

## Testing / Inference

### 1. Place your trained model

Copy your `.pth` file to `model_zoo/` and name it `aranet_b.pth`:

```
model_zoo/aranet_b.pth
```

### 2. Configure the test script

Open `main_test_aranet_b.py` and edit the configuration block at the top:

```python
noise_level_img   = 25          # noise level to apply to test images
noise_level_model = 25          # must match noise_level_img
n_channels        = 3           # 1 or 3
model_name        = 'aranet_b'  # name of .pth file in model_zoo/
testset_name      = 'set12'     # folder under testsets/
need_degradation  = True        # True: add noise; False: images already noisy
x8                = False       # True: use 8-fold self-ensemble
```

### 3. Run inference

```bash
python main_test_aranet_b.py
```

Results are saved to `results/set12_aranet_b/` together with a log file reporting per-image PSNR/SSIM and the dataset average.

---

## Model architecture summary

```
Input (B, C+1, H, W)    ← C = n_channels, +1 for noise-level map
     │
  Head Conv              3×3 conv → nf channels
     │
  Down 1                 nb × GroupedResidualBlock → stride-2 conv
  Down 2                 nb × GroupedResidualBlock → stride-2 conv
  Down 3                 nb × GroupedResidualBlock → stride-2 conv
     │
  Bottleneck             SAM(x) + FAM(x)
     │
  Up 3                   nb × GroupedResidualBlock → ConvTranspose2d ×2  + skip from Down 3
  Up 2                   nb × GroupedResidualBlock → ConvTranspose2d ×2  + skip from Down 2
  Up 1                   nb × GroupedResidualBlock → ConvTranspose2d ×2  + skip from Down 1
     │
  Tail Conv              3×3 conv → out_nc channels
     │
Output (B, out_nc, H, W)
```

**GroupedResidualBlock** uses depthwise-separable convolutions with a residual shortcut.  
**SAM** applies learned channel-wise spatial attention.  
**FAM** applies frequency-domain channel attention via 2-D FFT magnitude.

---

## Tips

- **Grayscale denoising**: set `n_channels = 1`, `in_nc = 2` (1 channel + noise map) in the JSON.
- **Stronger model**: increase `nb` (number of residual blocks per level) or `nc[0]` (feature width `nf`).
- **Faster training**: reduce `H_size` to 64 or lower `dataloader_batch_size` if GPU memory is limited.
- **Custom test images**: add your images to `testsets/` in a new subfolder and point `testset_name` to it.
