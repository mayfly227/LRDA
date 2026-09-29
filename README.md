# Code for "Learning Robust Vehicle Camouflage via Low-Rank Diffusion Adaptation"

## 1. Quick Start

The `Texture/` directory contains the texture presented in the paper. Import it into your CARLA simulator to test it.

To generate a texture matching the paper's setup without training, run `inference.py`. For inference, you only need PyTorch and Diffusers, the SDXL-Turbo model weights, and the pretrained LoRA weights. Set `LORA_PATH` in `inference.py` to the downloaded LoRA weights, and pass the SDXL-Turbo model directory with `--model_path`.

Download the pretrained LoRA weights from [Baidu Netdisk](https://pan.baidu.com/s/REPLACE_WITH_BAIDU_LINK) or [Google Drive](https://drive.google.com/file/d/REPLACE_WITH_FILE_ID/view). Replace these placeholder links with the actual download URLs.

## 2. Installation

We recommend Ubuntu 22.04 and an NVIDIA RTX 4090 GPU. Our environment uses Python 3.11, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, PyTorch3D 0.7.9, and nvdiffrast 0.4.0. Install them in the following order.

### 2.1 Install Python

```bash
conda create -n fastadv python=3.11
conda activate fastadv
```

### 2.2 Install PyTorch and Other Dependencies

We recommend PyTorch 2.8.0+cu128, although other versions may also work. If you need another version, select the appropriate command from the [PyTorch installation page](https://pytorch.org/get-started/locally/) and keep the `torch` and `torchvision` versions compatible.

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining dependencies:

```bash
python -m pip install \
  accelerate==1.13.0 diffusers==0.37.1 transformers==5.3.0 \
  peft==0.18.1 ultralytics==8.4.30 kornia==0.8.2 \
  numpy==2.4.3 Pillow==12.1.1 matplotlib==3.10.8 tensorboard==2.20.0
```

### 2.3 Install PyTorch3D and nvdiffrast

For PyTorch3D and nvdiffrast, follow the [torch_packages_builder installation guide](https://github.com/MiroPsota/torch_packages_builder#usage-with-pip). Select prebuilt packages compatible with your Python, PyTorch, and CUDA versions from its [package index](https://miropsota.github.io/torch_packages_builder/).

## 3. Train a Texture

### 3.1 Dataset

All training and test data (approximately 50 GB) will be released after the paper is accepted. You can also collect your own data using the CARLA simulator.

### 3.2 SDXL-Turbo

Download the complete Diffusers model directory from the [official SDXL-Turbo repository](https://huggingface.co/stabilityai/sdxl-turbo).

### 3.3 Training

```bash
python train_sdxlturbo_single2.py \
  --pretrained_model_name_or_path /path/to/sdxl-turbo \
  --data_root_prefix /path/to/carla-data-root \
  --reward_model_path /path/to/yolo-weights.pt \
  --mesh_obj_path /path/to/pytorch3d_Etron.obj \
  --uv_mask_path /path/to/modified_mask.png \
  --sample_seed -1 \
  --export_seed 42 \
  --lora_dropout 0 \
  --num_inference_steps 1 \
  --output_dir outputlora/example
```

`train_sdxlturbo_single2.py` is the main training script. It freezes the SDXL-Turbo base weights, optimizes the UNet LoRA parameters, and exports a deterministic texture at the end of training using `--export_seed`.

| Argument | Description |
| --- | --- |
| `--sample_seed -1` | Resample noise at each training step; a nonnegative value fixes the sampling trajectory. |
| `--export_seed 42` | Fix the seed used to generate the final texture so it can be reproduced. |
| `--prompt_text` | Texture generation prompt; defaults to `colorful camouflage`. |
| `--max_train_steps` | Number of training steps; defaults to `3000`. |
| `--num_reward_views` | Number of CARLA views per step; defaults to `8`. |
| `--nvdiffrast_texture_filter` | Texture filtering mode used for rendering; defaults to `linear-mipmap-linear`. |
| `--output_dir` | Directory for LoRA weights, checkpoints, logs, and the final texture. |

The main outputs are:

```text
outputlora/example/
├── final_texture.png    # Full texture exported by the diffusion model
├── uv_texture.png       # Vehicle texture after applying the UV mask
├── final_texture.pt     # Texture tensor
├── final_texture.json   # Export configuration
├── pytorch_lora_weights.safetensors
└── checkpoint-*/        # Training checkpoints and their textures
```
