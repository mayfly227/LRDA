"""SDXL Turbo 推理脚本

用法:
    python shortimescript/test_sdxl_turbo.py "a red sports car on a highway"
    python shortimescript/test_sdxl_turbo.py "a cat" --steps 4 --seed 123 -o out/cat.png
"""

import argparse
import glob
import os

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

# 更换为你下载的权重位置
MODEL_PATH = "/mnt/data/dk/hf/AI-ModelScope--sdxl-turbo" 

#已经训练好的lora参数
LORA_PATH = "outputlora/0825_pido_psfweak_roasda_factorial_seed42_real/pido_roasda__rep2/pytorch_lora_weights.safetensors"
def resolve_snapshot(path: str) -> str:
    """ModelScope 缓存目录需要定位到 snapshots/<hash>/ 实际模型目录"""
    if os.path.exists(os.path.join(path, "model_index.json")):
        return path
    snaps = glob.glob(os.path.join(path, "snapshots", "*"))
    if len(snaps) == 1:
        return snaps[0]
    raise FileNotFoundError(f"在 {path} 下找不到模型目录 (model_index.json)")


def parse_args():
    parser = argparse.ArgumentParser(description="SDXL Turbo 文生图推理")
    parser.add_argument("--prompt", type=str, default="colorful camouflage",help="文本提示词")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="模型路径")
    parser.add_argument("--steps", type=int, default=1, help="推理步数, turbo 建议 1~4")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("-o", "--output", type=str, default="sdxl_turbo_out.png", help="输出图片路径")
    return parser.parse_args()


def main():
    args = parse_args()

    pipe = AutoPipelineForText2Image.from_pretrained(
        resolve_snapshot(args.model_path),
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to("cuda")
    pipe.load_lora_weights(LORA_PATH)
    generator = torch.Generator("cuda").manual_seed(args.seed)

    # SDXL Turbo: guidance_scale 必须为 0, 步数 1~4
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    image.save(args.output)
    print(f"saved: {args.output}")

    mask = Image.open("Texture/modified_mask.png").convert("RGB").resize((640, 640), Image.Resampling.NEAREST)
    background = Image.open("Texture/img_optim.png").convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
    texture = image.convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
    uv = np.where(np.asarray(mask) > 127, np.asarray(texture), np.asarray(background)).astype(np.uint8)
    uv_output = os.path.splitext(args.output)[0] + "_uv.png"
    Image.fromarray(uv).save(uv_output)
    print(f"saved: {uv_output}")


if __name__ == "__main__":
    main()
