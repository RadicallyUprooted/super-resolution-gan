# Super-Resolution GAN (SRGAN) in PyTorch

This project provides a PyTorch implementation of the Super-Resolution Generative Adversarial Network (SRGAN) by [(Ledig et al., 2016)](https://arxiv.org/abs/1609.04802). Originally, it was my undegraduate studies' coursework.

<div style="display: flex; justify-content: center;">
  <img src="test_imgs/sodachi_bicubic.jpg" alt="Bicubic interpolation" style="width: 48%; margin-right: 2%;">
  <img src="test_imgs/sodachi_srgan.jpg" alt="SRGAN 4x upscale" style="width: 48%;">
</div>

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd super-resolution-gan
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

The training process is split into two stages: pre-training the generator and then training the full SRGAN.

### 1. Pre-train the Generator

First, pre-train the generator using L1 pixel loss to get a good starting point for the GAN training. This helps the generator produce plausible images before the discriminator is introduced.

```bash
python train_generator.py \
  --image_dir /path/to/your/dataset \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4 \
  --n_res_blocks 16 \
  --upscale_factor 4 \
  --test_image_path /path/to/your/low_res_test_image.png \
  --output_dir results/pretrain
```

- `--image_dir`: Path to your high-resolution training images.
- `--test_image_path`: A low-resolution image to test the generator's output at each epoch.
- `--output_dir`: Where the upscaled test images will be saved.

Checkpoints will be saved as `generator_epoch_{epoch}.pth`.

### 2. Train the SRGAN

Once the generator is pre-trained, you can train the full SRGAN, which introduces the discriminator and the perceptual (VGG) loss.

```bash
python train_srgan.py \
  --image_dir /path/to/your/dataset \
  --epochs 200 \
  --batch_size 16 \
  --generator_checkpoint /path/to/your/generator_epoch_100.pth \
  --test_image_path /path/to/your/low_res_test_image.png \
  --output_dir results/srgan
```

- `--generator_checkpoint`: **Important:** Path to the pre-trained generator checkpoint from the previous step.
- The other arguments are similar to the pre-training script.

Checkpoints for both the generator and discriminator will be saved during training.

## Inference

To use a trained generator (either from SRResNet pre-training or SRGAN training), use the `inference.py` script.

```bash
python inference.py \
  --checkpoint /path/to/your/generator_checkpoint.pth \
  --input_path /path/to/input_image_or_directory \
  --output_path /path/to/output_directory
```

- `--checkpoint`: Path to the trained generator checkpoint file (`.pth`). This can be a checkpoint from `train_generator.py` (SRResNet) or `train_srgan.py` (SRGAN).
- `--input_path`: Path to a single low-resolution input image (e.g., `image.png`) or a directory containing multiple low-resolution images.
- `--output_path`: Path to the directory where the upscaled images will be saved. The script will create this directory if it doesn't exist.
- `--upscale_factor`: (Optional) The upscale factor used during training (default is 4).
