from meshrender.nvdiffrast_uv import NvdiffrastTextureRenderModel
import glob
import os
from datetime import datetime

from data.dataset import build_cameras
from utils.aug import ROA_SDA, ROA_FCA, ROADiffusionImage
from utils.core import get_free_gpu, load_uv_mask
from utils.loss import (
    lora_regularization,
    margin_detection_loss,
)
from utils.pido import PIDO

if os.environ.get("CUDA_VISIBLE_DEVICES"):
    # 调用方已显式绑卡（多卡并行场景），get_free_gpu 会无条件覆盖该变量，必须跳过
    pass
else:
    try:
        get_free_gpu()
    except Exception as e:
        print(
            f"Warning: get_free_gpu() failed with error: {e}. Proceeding without setting CUDA_VISIBLE_DEVICES."
        )
        exit(0)

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Iterable, List

import diffusers
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoPipelineForText2Image,  # pyright: ignore[reportPrivateImportUsage]
    StableDiffusionXLPipeline,  # pyright: ignore[reportPrivateImportUsage]
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,  # pyright: ignore[reportPrivateImportUsage]
)
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image

from meshrender import tensor_rgb_to_pil

logger = get_logger(__name__)
YOLO_REWARD_IMAGE_SIZE = 640

LORA_TARGET_LEAF_MODULES = (
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "add_k_proj",
    "add_v_proj",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reward-driven LoRA fine-tuning with UV masking, PyTorch3D render, and YOLO reward (SDXL-Turbo)."
    )
    # 预训练扩散模型路径（SDXL-Turbo）
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/mnt/data/dk/hf/AI-ModelScope--sdxl-turbo",
    )
    # 模型版本号或分支名
    parser.add_argument("--revision", type=str, default=None)
    # 模型权重变体，例如 fp16
    parser.add_argument("--variant", type=str, default="fp16")
    # 训练输出目录
    parser.add_argument("--output_dir", type=str, default="rewardyolosdxlturbo")
    # 日志子目录
    parser.add_argument("--logging_dir", type=str, default="logs")
    # 随机种子
    parser.add_argument("--seed", type=int, default=42)
    # 混合精度训练模式
    parser.add_argument(
        "--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]
    )
    parser.add_argument("--use_margin_loss", action="store_true")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="Detection confidence margin used by --use_margin_loss.",
    )
    # 梯度累积步数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # 是否启用梯度检查点节省显存
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # LoRA 秩
    parser.add_argument("--rank", type=int, default=64)
    # LoRA dropout 概率
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    # LoRA 注入范围
    parser.add_argument(
        "--lora_scope",
        type=str,
        default="both",
        choices=["attn1", "attn2", "both"],
        help="Choose whether LoRA is injected into self-attention, cross-attention, or both.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # 优化器类型。SGD 下 loss 里的 L2(loss_reg) 等价于真正的 weight decay，
    # 且平解偏置更强；代价是收敛慢，lr 需要比 AdamW 大几个数量级
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer for LoRA parameters.",
    )
    # SGD 动量
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    # SGD 是否启用 Nesterov 动量
    parser.add_argument("--sgd_nesterov", action="store_true")
    # 梯度裁剪上限
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # 学习率调度器类型
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    # 学习率预热步数
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    # 学习率循环次数
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    # 多项式调度的幂指数
    parser.add_argument("--lr_power", type=float, default=1.0)
    # 总训练步数
    parser.add_argument("--max_train_steps", type=int, default=3000)
    # 每隔多少步保存 checkpoint
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    # 最多保留多少个 checkpoint
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    # reward 分支的 batch 大小
    parser.add_argument("--reward_batch_size", type=int, default=1)
    # 每步用于平均 reward 的渲染视角数量
    parser.add_argument("--num_reward_views", type=int, default=8)
    # 扩散采样步数（SDXL-Turbo 推荐 1~4）
    parser.add_argument("--num_inference_steps", type=int, default=1)
    # SDXL-Turbo 蒸馏时不带 CFG，>1 才会启用双分支 guidance
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    # 仅在 guidance_scale > 1 时用到
    parser.add_argument("--negative_prompt", type=str, default="")
    # 训练时喂给 SDXL 的文本提示
    parser.add_argument("--prompt_text", type=str, default="colorful camouflage")
    # 只对最后 N 个去噪步保留计算图（截断反传），-1 表示全部步都保留。
    # Turbo 只有 1~4 步，默认全保留；显存吃紧时可设成 1 或 2。
    parser.add_argument("--grad_last_steps", type=int, default=-1)
    # 采样噪声的固定种子：每个训练步都用同一个 generator 重新播种，
    # 让 LoRA 只针对这一条采样轨迹优化 —— 目标是产出唯一的一张对抗纹理。
    # 设为 -1 则每步重新随机（优化噪声分布上的期望，产出的是"生成器"而非单张图）。
    parser.add_argument("--sample_seed", type=int, default=0)
    # 导出最终纹理时用的种子。不指定时：--sample_seed >= 0 就沿用它（导出的正是被
    # 优化的那条轨迹）；--sample_seed < 0（训练走随机噪声）则退回 0，保证交付物
    # 本身始终是可复现的。
    parser.add_argument("--export_seed", type=int, default=None)
    # 扩散生成分辨率
    parser.add_argument("--resolution", type=int, default=512)
    # 渲染图像分辨率
    parser.add_argument("--render_image_size", type=int, default=640)
    # UV 纹理图分辨率
    parser.add_argument("--texture_image_size", type=int, default=640)
    # YOLO 奖励模型路径
    parser.add_argument("--reward_model_path", type=str, default="yolomodel/yolov3u.pt")
    parser.add_argument("--reward_mode", type=str, default="non_targeted")
    # reward 项的损失权重
    parser.add_argument("--reward_weight", type=float, default=1.0)
    # LoRA 正则项权重
    parser.add_argument("--lora_reg_weight", type=float, default=1e-4)
    # 渲染所用 OBJ 模型路径
    parser.add_argument(
        "--mesh_obj_path", type=str, default="carmodel/audi/pytorch3d_Etron.obj"
    )
    parser.add_argument(
        "--use_carla_mesh_transform",
        type=bool,
        default=True,
        help="Use CARLA-aligned mesh centering, scaling, and yaw rotation.",
    )
    # UV 掩码图片路径
    parser.add_argument(
        "--uv_mask_path", type=str, default="carmodel/audi/modified_mask.png"
    )
    # UV 掩码外区域的填充值
    parser.add_argument("--background_fill_value", type=float, default=0.0)
    # nvdiffrast 纹理采样过滤模式
    parser.add_argument(
        "--nvdiffrast_texture_filter",
        type=str,
        default="linear-mipmap-linear",
        choices=["nearest", "linear", "linear-mipmap-nearest", "linear-mipmap-linear"],
    )
    # nvdiffrast 纹理 mipmap 最大层级；-1 = 自动计算（按纹理尺寸，取不产生奇数降采样的最高层级）
    parser.add_argument("--texture_max_mip_level", type=int, default=-1)
    # PIDO 观测模型消融开关：默认关闭（旧的显示域 alpha 合成）；设此 flag 才启用物理观测
    parser.add_argument("--use_pido", action="store_true")
    parser.add_argument(
        "--pido_scope",
        choices=("foreground", "full"),
        default="foreground",
        help="Apply PIDO to the rendered foreground or the full composited image.",
    )
    parser.add_argument(
        "--pido_apply_prob",
        type=float,
        default=1.0,
        help="Probability of applying the complete PIDO module independently per reward view.",
    )
    parser.add_argument(
        "--pido_psf_sigma_scale_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.75, 1.5),
        help="Random PIDO PSF sigma multiplier range.",
    )
    parser.add_argument(
        "--pido_visibility_km_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(1.0, 20.0),
        help="Random PIDO atmospheric visibility range in kilometres.",
    )
    # CARLA 数据的根目录前缀（相对路径会拼在它后面）
    parser.add_argument(
        "--data_root_prefix", type=str, default="/home/dk/project/FastAdv"
    )

    parser.add_argument("--use_diffusion_image_aug", action="store_true")
    parser.add_argument("--use_roa", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.pido_apply_prob <= 1.0:
        parser.error("--pido_apply_prob must be in [0, 1]")
    return args


def ensure_required_paths(args):
    required_paths = {
        "mesh_obj_path": args.mesh_obj_path,
        "uv_mask_path": args.uv_mask_path,
        "reward_model_path": args.reward_model_path,
    }
    for label, path in required_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"`{label}` does not exist: {path}")


def cycle_dataloader(dataloader):
    while True:
        yield from dataloader


def make_sample_generator(seed, device):
    """每个训练步都用同一个种子重新播种一个 generator，把初始噪声和 Euler-Ancestral
    每步注入的噪声一起钉死。这样 LoRA 优化的就是唯一一条采样轨迹 —— 产出一张确定的
    纹理，而不是一族纹理的期望。seed < 0 时返回 None（回到每步重新随机）。"""
    if seed is None or seed < 0:
        return None
    return torch.Generator(device=device).manual_seed(int(seed))


def resolve_export_seed(args) -> int:
    """导出纹理用的种子。显式给了 --export_seed 就用它；否则固定 seed 训练时沿用
    --sample_seed（导出的就是被优化的那条轨迹），随机训练时退回 0。
    永远返回 >= 0 的值 —— 交付物必须可复现。"""
    if args.export_seed is not None:
        return int(args.export_seed)
    return int(args.sample_seed) if args.sample_seed >= 0 else 0


def prepare_sampling_scheduler(pipe, num_inference_steps: int, device):
    """每个训练步都新建一份 scheduler：Euler 系 scheduler 内部有 `_step_index`
    等状态，复用同一个实例会导致第二步开始 sigma 取错。"""
    scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(num_inference_steps, device=device)
    return scheduler


@torch.no_grad()
def encode_prompts(
    pipe,
    prompts: List[str],
    negative_prompt: str,
    do_classifier_free_guidance: bool,
    device,
    dtype,
):
    """文本编码器全程冻结，这里不需要计算图。"""
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=[negative_prompt] * len(prompts)
        if do_classifier_free_guidance
        else None,
    )

    def _cast(tensor):
        return None if tensor is None else tensor.to(device=device, dtype=dtype)

    return (
        _cast(prompt_embeds),
        _cast(pooled_prompt_embeds),
        _cast(negative_prompt_embeds),
        _cast(negative_pooled_prompt_embeds),
    )


def build_sdxl_time_ids(
    batch_size: int, resolution: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    # SDXL micro-conditioning: [original_h, original_w, crop_y, crop_x, target_h, target_w]
    add_time_ids = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        device=device,
        dtype=dtype,
    )
    return add_time_ids.unsqueeze(0).repeat(batch_size, 1)


def _denoise_step(
    unet,
    scheduler,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    add_time_ids: torch.Tensor,
    guidance_scale: float,
    dtype: torch.dtype,
    generator=None,
) -> torch.Tensor:
    """单步去噪。guidance_scale <= 1 时走单分支（SDXL-Turbo 的默认用法）。"""
    do_cfg = guidance_scale > 1.0

    if do_cfg:
        latent_model_input = torch.cat([latents, latents], dim=0)
        encoder_hidden_states = torch.cat(
            [negative_prompt_embeds, prompt_embeds], dim=0
        )
        text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )
        time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
    else:
        latent_model_input = latents
        encoder_hidden_states = prompt_embeds
        text_embeds = pooled_prompt_embeds
        time_ids = add_time_ids

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep).to(
        dtype=dtype
    )

    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=encoder_hidden_states.to(dtype=dtype),
        added_cond_kwargs={
            "text_embeds": text_embeds.to(dtype=dtype),
            "time_ids": time_ids.to(device=latents.device, dtype=dtype),
        },
        return_dict=False,
    )[0]

    if do_cfg:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # uncond 分支不接收 reward 梯度，只当作固定基准
        noise_pred_uncond = noise_pred_uncond.detach()
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    # EulerAncestralDiscreteScheduler 每步都会往 prev_sample 里注入 sigma_up * noise，
    # 这里必须把 generator 传进去，否则即使初始 latent 固定，采样结果依然是随机的。
    latents = scheduler.step(
        noise_pred.to(dtype=dtype),
        timestep,
        latents,
        generator=generator,
        return_dict=False,
    )[0]
    return latents.to(dtype=dtype)


def sample_images_with_grad(
    unet,
    vae,
    scheduler,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    add_time_ids: torch.Tensor,
    device,
    dtype: torch.dtype,
    batch_size: int,
    resolution: int,
    guidance_scale: float,
    grad_last_steps: int = -1,
    generator=None,
) -> torch.Tensor:
    """手工展开去噪循环 + VAE 解码，全程保留计算图，让 reward 的梯度能回传到 LoRA。

    diffusers 的 pipeline `__call__` 上带 `@torch.no_grad()` 且只吐 PIL，用它采样
    等于把梯度链剪断，LoRA 只会被正则项拉向 0，所以这里必须自己跑。

    返回 (B, 3, H, W)、取值范围 [0, 1] 的图像。
    """
    latent_channels = unet.config.in_channels
    latent_size = resolution // (2 ** (len(vae.config.block_out_channels) - 1))

    # 固定用 fp32 抽噪声再转 dtype：randn 在 fp16 下抽出来的数跟 fp32 不同，
    # 这样才能和 latent 版对照实验从完全相同的初始 latent 出发。
    latents = torch.randn(
        (batch_size, latent_channels, latent_size, latent_size),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    total_steps = len(scheduler.timesteps)
    # grad_start 之前的步用 no_grad 跑并 detach，只有后 grad_last_steps 步进反传
    if grad_last_steps is None or grad_last_steps < 0:
        grad_start = 0
    else:
        grad_start = max(total_steps - grad_last_steps, 0)

    for step_idx, timestep in enumerate(scheduler.timesteps):
        if step_idx < grad_start:
            with torch.no_grad():
                latents = _denoise_step(
                    unet,
                    scheduler,
                    latents,
                    timestep,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    add_time_ids,
                    guidance_scale,
                    dtype,
                    generator=generator,
                )
            latents = latents.detach()  # 显式断开计算图
        else:
            latents = _denoise_step(
                unet,
                scheduler,
                latents,
                timestep,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                add_time_ids,
                guidance_scale,
                dtype,
                generator=generator,
            )

    # VAE 保持 fp32（半精度下 SDXL 的 VAE 容易出黑图/NaN），解码时把 latents 对齐到
    # VAE 的 dtype——梯度照样能穿过这个 cast 回到 latents。
    vae_dtype = next(vae.parameters()).dtype
    images = vae.decode(
        (latents / vae.config.scaling_factor).to(dtype=vae_dtype),
        return_dict=False,
    )[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    return images


@torch.no_grad()
def export_texture(
    args, pipe, prompt: str, device, dtype, save_dir: Path, texture_render_model=None
):
    """用固定种子重采一次并落盘 —— 这是"只要一张对抗纹理"这个目标的最终交付物。

    采样前切到 eval：LoRA dropout 在 train 模式下会让同一个种子采出不同的图，
    评估和交付都必须关掉它。

    种子走 `resolve_export_seed`，而不是直接用 --sample_seed：训练可以用
    --sample_seed -1 走随机噪声（当正则），但导出这一张必须是确定的。
    """
    export_seed = resolve_export_seed(args)
    was_training = pipe.unet.training
    pipe.unet.eval()
    try:
        scheduler = prepare_sampling_scheduler(pipe, args.num_inference_steps, device)
        (
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = encode_prompts(
            pipe,
            [prompt],
            args.negative_prompt,
            do_classifier_free_guidance=args.guidance_scale > 1.0,
            device=device,
            dtype=dtype,
        )
        texture = sample_images_with_grad(
            pipe.unet,
            pipe.vae,
            scheduler,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            build_sdxl_time_ids(1, args.resolution, device, dtype),
            device,
            dtype,
            1,
            args.resolution,
            args.guidance_scale,
            grad_last_steps=0,  # 导出不需要计算图
            generator=make_sample_generator(export_seed, device),
        )
    finally:
        pipe.unet.train(was_training)

    save_dir.mkdir(parents=True, exist_ok=True)
    tensor_rgb_to_pil(texture).save(save_dir / "final_texture.png")
    # 同时落盘 UV 掩码后的纹理图（车身实际使用的区域）
    if texture_render_model is not None:
        uv_masked = texture_render_model.get_uv_masked(texture)
        tensor_rgb_to_pil(uv_masked).save(save_dir / "uv_texture.png")
    torch.save(texture.cpu(), save_dir / "final_texture.pt")
    with open(save_dir / "final_texture.json", "w") as f:
        json.dump(
            {
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
                "sample_seed": args.sample_seed,
                "export_seed": export_seed,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "resolution": args.resolution,
                "lora_dropout": args.lora_dropout,
                "use_pido": args.use_pido,
                "pido_scope": args.pido_scope,
                "pido_apply_prob": args.pido_apply_prob,
                "pido_psf_sigma_scale_range": args.pido_psf_sigma_scale_range,
                "pido_visibility_km_range": args.pido_visibility_km_range,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return texture


def render_reward_multi_view_carla(
    texture_render_model: NvdiffrastTextureRenderModel,
    pido,
    yolo_reward_model,
    diffusion_images: torch.Tensor,
    carla_batch,
    carla_dataset,
    num_views: int,
    reward_mode: str = "non_targeted",
    target_class_id: int = 2,
    roa=None,
    pido_scope: str = "foreground",
    pido_apply_prob: float = 1.0,
):
    rendered_views = []
    reward_values = []
    reward_statuses = []
    view_cameras = []

    device = diffusion_images.device
    carla_images = carla_batch["image"].to(device)
    batch_size = carla_images.shape[0]
    views_to_render = min(num_views, batch_size)
    if views_to_render < 1:
        raise ValueError("CARLA batch is empty; cannot render reward views.")

    for view_index in range(views_to_render):
        cam = build_cameras(
            carla_batch["R"][view_index : view_index + 1],
            carla_batch["T"][view_index : view_index + 1],
            carla_dataset,
            device=device,
        )
        view_background = carla_images[view_index : view_index + 1]
        foreground_rgba = texture_render_model(
            cam,
            diffusion_images,
            background=None,
            return_rgba=True,
        )
        foreground_size = foreground_rgba.shape[1:3]
        if view_background.shape[-2:] != foreground_size:
            view_background = F.interpolate(
                view_background,
                size=foreground_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        background_hwc = view_background.permute(0, 2, 3, 1)
        apply_pido = pido is not None and (
            pido_apply_prob >= 1.0
            or (
                pido_apply_prob > 0.0
                and torch.rand((), device=device).item() < pido_apply_prob
            )
        )
        if not apply_pido:
            # 消融对照组：旧的显示域 alpha 合成
            rgb, alpha = foreground_rgba[..., :3], foreground_rgba[..., 3:4]
            rendered = rgb * alpha + background_hwc * (1.0 - alpha)
        elif pido_scope == "foreground":
            rendered = pido.forward_rgba(
                foreground_rgba,
                background_hwc,
                carla_batch["dist"][view_index : view_index + 1].to(
                    device=device, dtype=torch.float32
                ),
            )
        else:
            rgb, alpha = foreground_rgba[..., :3], foreground_rgba[..., 3:4]
            composited = rgb * alpha + background_hwc * (1.0 - alpha)
            opaque_rgba = torch.cat((composited, torch.ones_like(alpha)), dim=-1)
            rendered = pido.forward_rgba(
                opaque_rgba,
                torch.zeros_like(background_hwc),
                carla_batch["dist"][view_index : view_index + 1].to(
                    device=device, dtype=torch.float32
                ),
            )
        rendered = rendered.permute(0, 3, 1, 2).clamp(0, 1)
        reward_input = rendered
        if reward_input.shape[-2:] != (
            YOLO_REWARD_IMAGE_SIZE,
            YOLO_REWARD_IMAGE_SIZE,
        ):
            reward_input = F.interpolate(
                reward_input,
                size=(YOLO_REWARD_IMAGE_SIZE, YOLO_REWARD_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if roa is not None:
            rendered = roa(reward_input)
        else:
            rendered = reward_input
        reward_status, reward_value = yolo_reward_model(
            rendered, reward_mode, target_class_id
        )
        rendered_views.append(rendered)
        reward_values.append(reward_value.reshape(()))
        reward_statuses.append(reward_status)
        view_cameras.append(cam)

    reward_tensor = torch.stack(reward_values)
    mean_reward = reward_tensor.mean()
    any_detected = any(reward_statuses)

    return {
        "reward_status": any_detected,
        "reward_value": mean_reward,
        "reward_values": reward_tensor,
        "rendered_views": rendered_views,
        "view_cameras": view_cameras,
    }


def resolve_unet_lora_target_modules(unet, lora_scope: str) -> List[str]:
    if lora_scope not in {"attn1", "attn2", "both"}:
        raise ValueError(f"Unsupported LoRA scope: {lora_scope}")

    target_modules = []
    for module_name, _ in unet.named_modules():
        if not module_name or not module_name.endswith(LORA_TARGET_LEAF_MODULES):
            continue
        if lora_scope == "attn1" and ".attn1." not in module_name:
            continue
        if lora_scope == "attn2" and ".attn2." not in module_name:
            continue
        target_modules.append(module_name)

    target_modules = sorted(set(target_modules))
    if not target_modules:
        raise ValueError(
            f"No UNet modules matched LoRA scope `{lora_scope}` with leaves {LORA_TARGET_LEAF_MODULES}."
        )
    return target_modules


def build_optimizer(args, params: Iterable[torch.nn.Parameter]):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=args.learning_rate,
            momentum=args.sgd_momentum,
            nesterov=args.sgd_nesterov,
        )
    return torch.optim.AdamW(
        params,
        lr=args.learning_rate,
    )


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def prune_checkpoints(output_dir: str, total_limit):
    if total_limit is None:
        return
    checkpoints = sorted(
        [path for path in os.listdir(output_dir) if path.startswith("checkpoint-")],
        key=lambda item: int(item.split("-")[1]),
    )
    if len(checkpoints) < total_limit:
        return
    remove_count = len(checkpoints) - total_limit + 1
    for checkpoint in checkpoints[:remove_count]:
        shutil.rmtree(Path(output_dir) / checkpoint, ignore_errors=True)


def main(args):
    ensure_required_paths(args)

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    tracker_name = f"fastadv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    temp_args = vars(args).copy()
    temp_args = {
        key: str(value)
        if value is None or isinstance(value, (list, tuple, dict))
        else value
        for key, value in temp_args.items()
    }

    accelerator.init_trackers(tracker_name, config=temp_args)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    MODEL_PATH = args.pretrained_model_name_or_path
    if not os.path.exists(os.path.join(MODEL_PATH, "model_index.json")):
        MODEL_PATH = glob.glob(os.path.join(MODEL_PATH, "snapshots", "*"))[0]
    # 通过 Pipeline API 一次性加载 SDXL-Turbo 全部组件
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_PATH,
        torch_dtype=weight_dtype,
        variant=args.variant,
        revision=args.revision,
    )
    pipe.to(accelerator.device)
    if args.reward_model_path in ["yolomodel/yolov3lz.pt","yolomodel/yolov3.pt"]:
        from thirdpart.yolov3.yolov3_detector import YoloV3Detector

        yolo_reward_model = YoloV3Detector(
            args.reward_model_path, device=accelerator.device
        )
    else:
        from utils.yolo import YoloRewardModel

        yolo_reward_model = YoloRewardModel(
            args.reward_model_path,
        )
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # SDXL 训练中保持 VAE 为 fp32，避免解码出现黑图/NaN。
    pipe.vae.to(torch.float32)
    if args.use_roa:
        roa = ROA_SDA()  # 实例化 ROA
        roa.to(accelerator.device)
    else:
        roa = None

    if args.use_pido:
        # Unknown capture distance is handled with an EOT distribution calibrated
        # to the 8--15 m CARLA/physical capture range.
        pido = PIDO(
            data_format="NHWC",
            randomize_psf=True,
            random_psf_distance_range_m=(8.0, 15.0),
            random_psf_sigma_scale_range=tuple(args.pido_psf_sigma_scale_range),
            randomize_visibility=True,
            random_visibility_km_range=tuple(args.pido_visibility_km_range),
        ).to(accelerator.device)
    else:
        # 默认：不做物理观测，直接用旧的显示域 alpha 合成
        pido = None

    roa_diffusion_image = ROADiffusionImage()
    roa_diffusion_image.to(accelerator.device)
    yolo_reward_model.to(accelerator.device)

    uv_mask = load_uv_mask(
        args.uv_mask_path, args.texture_image_size, accelerator.device
    )
    from meshrender.nvdiffrast_uv import NvdiffrastTextureRenderModel

    texture_render_model = NvdiffrastTextureRenderModel(
        device=accelerator.device,
        mesh_obj_path=args.mesh_obj_path,
        uv_mask=uv_mask,
        render_image_size=args.render_image_size,
        texture_image_size=args.texture_image_size,
        background_fill_value=args.background_fill_value,
        use_carla_mesh_transform=args.use_carla_mesh_transform,
        use_antialias=True,
        texture_filter_mode=args.nvdiffrast_texture_filter,
        texture_max_mip_level=(
            None
            if args.texture_max_mip_level == -1
            else args.texture_max_mip_level
        ),
    )
    texture_render_model.to(accelerator.device)

    if args.gradient_checkpointing:
        pipe.unet.enable_gradient_checkpointing()

    lora_target_modules = resolve_unet_lora_target_modules(pipe.unet, args.lora_scope)
    logger.info(
        "LoRA scope `%s` matched %d UNet modules.",
        args.lora_scope,
        len(lora_target_modules),
    )
    if args.lora_scope == "both":
        print(
            "重要:LoRA will be applied to both attn1 and attn2 modules, which may increase training time and memory usage but can potentially improve reward optimization."
        )
        lora_target_modules = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "add_k_proj",
            "add_v_proj",
        ]
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=lora_target_modules,
    )
    pipe.unet.add_adapter(unet_lora_config)
    cast_training_params([pipe.unet], dtype=torch.float32)
    pipe.unet.train()  # LoRA dropout 只在 train 模式下生效
    pipe.set_progress_bar_config(disable=True)

    trainable_lora_names = [
        name
        for name, parameter in pipe.unet.named_parameters()
        if parameter.requires_grad and "lora" in name.lower()
    ]
    logger.info(
        "Enabled %d trainable LoRA tensors for scope `%s`.",
        len(trainable_lora_names),
        args.lora_scope,
    )

    params_to_optimize = [
        parameter for parameter in pipe.unet.parameters() if parameter.requires_grad
    ]
    optimizer = build_optimizer(args, params_to_optimize)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    from data.dataset import make_loader

    root = [
        "data/datas/origin/si103/day_clear/10m",
        "data/datas/origin/si103/day_clear/12m",
        "data/datas/origin/si103/day_clear/8m",
        "data/datas/origin/si103/day_cloudy/10m",
        "data/datas/origin/si103/day_cloudy/12m",
        "data/datas/origin/si103/day_cloudy/8m",
        "data/datas/origin/si103/night_19/15m",
        "data/datas/origin/si103/night_clear/10m",
        "data/datas/origin/si103/night_clear/12m",
        "data/datas/origin/si103/night_clear/8m",
        "data/datas/origin/si103/sunny/15m",
        "data/datas/origin/si123/fog/8m",
        "data/datas/origin/si123/night/10m",
        "data/datas/origin/si123/rainy/12m",
        "data/datas/origin/si123/sunny/15m",
        "data/datas/origin/si12/fog/15m",
        "data/datas/origin/si12/night/10m",
        "data/datas/origin/si12/rainy/12m",
        "data/datas/origin/si12/rainy/8m",
        "data/datas/origin/si12/sunny/12m",
        "data/datas/origin/si12/sunny/8m",
        "data/datas/origin/si139/fog/10m",
        "data/datas/origin/si139/night_19/15m",
        "data/datas/origin/si139/night/8m",
        "data/datas/origin/si139/rainy/15m",
        "data/datas/origin/si139/sunny/10m",
        "data/datas/origin/si139/sunny/12m",
        "data/datas/origin/si139/sunny/15m",
        "data/datas/origin/si139/sunny/8m",
        "data/datas/origin/si148/fog/15m",
        "data/datas/origin/si148/night/12m",
        "data/datas/origin/si148/rainy/10m",
        "data/datas/origin/si148/sunny/8m",
        "data/datas/origin/si152/fog/15m",
        "data/datas/origin/si152/night/10m",
        "data/datas/origin/si152/rainy/12m",
        "data/datas/origin/si152/sunny/8m",
        "data/datas/origin/si153/fog/8m",
        "data/datas/origin/si153/night/12m",
        "data/datas/origin/si153/rainy/15m",
        "data/datas/origin/si153/sunny/10m",
        "data/datas/origin/si27/fog/12m",
        "data/datas/origin/si27/night/15m",
        "data/datas/origin/si27/night/8m",
        "data/datas/origin/si27/rainy/10m",
        "data/datas/origin/si27/sunny/15m",
        "data/datas/origin/si27/sunny/8m",
        "data/datas/origin/si33/fog/10m",
        "data/datas/origin/si33/fog/8m",
        "data/datas/origin/si33/night/12m",
        "data/datas/origin/si33/night/15m",
        "data/datas/origin/si33/rainy/12m",
        "data/datas/origin/si33/rainy/15m",
        "data/datas/origin/si33/sunny/10m",
        "data/datas/origin/si33/sunny/8m",
        "data/datas/origin/si47/day_clear/10m",
        "data/datas/origin/si47/day_clear/12m",
        "data/datas/origin/si47/day_clear/8m",
        "data/datas/origin/si47/day_cloudy/10m",
        "data/datas/origin/si47/day_cloudy/12m",
        "data/datas/origin/si47/day_cloudy/8m",
        "data/datas/origin/si47/night_19/15m",
        "data/datas/origin/si47/night_clear/10m",
        "data/datas/origin/si47/night_clear/12m",
        "data/datas/origin/si47/night_clear/8m",
        "data/datas/origin/si47/sunny/15m",
        "data/datas/origin/si54/sunny/10m",
        "data/datas/origin/si54/sunny/12m",
        "data/datas/origin/si54/sunny/6m",
        "data/datas/origin/si54/sunny/8m",
        "data/datas/origin/si62/fog/8m",
        "data/datas/origin/si62/night/10m",
        "data/datas/origin/si62/night/12m",
        "data/datas/origin/si62/rainy/12m",
        "data/datas/origin/si62/rainy/15m",
        "data/datas/origin/si62/sunny/10m",
        "data/datas/origin/si62/sunny/15m",
        "data/datas/origin/si70/day_clear/10m",
        "data/datas/origin/si70/day_clear/12m",
        "data/datas/origin/si70/day_clear/8m",
        "data/datas/origin/si70/day_cloudy/10m",
        "data/datas/origin/si70/day_cloudy/12m",
        "data/datas/origin/si70/day_cloudy/8m",
        "data/datas/origin/si70/night_19/15m",
        "data/datas/origin/si70/night_clear/10m",
        "data/datas/origin/si70/night_clear/12m",
        "data/datas/origin/si70/night_clear/8m",
        "data/datas/origin/si70/sunny/15m",
        # "data/datas/origin/si75/sunny/10m",
        # "data/datas/origin/si75/sunny/12m",
        # "data/datas/origin/si75/sunny/8m",
        "data/datas/origin/si87/fog/10m",
        "data/datas/origin/si87/fog/12m",
        "data/datas/origin/si87/night/8m",
        "data/datas/origin/si87/rainy/15m",
        "data/datas/origin/si87/sunny/10m",
        "data/datas/origin/si87/sunny/12m",
        "data/datas/origin/si95/fog/12m",
        "data/datas/origin/si95/fog/15m",
        "data/datas/origin/si95/night/8m",
        "data/datas/origin/si95/rainy/10m",
        "data/datas/origin/si95/rainy/15m",
        "data/datas/origin/si95/sunny/10m",
        "data/datas/origin/si95/sunny/12m",
    ]

    # CARLA 数据不在本仓库里，相对路径统一拼到 --data_root_prefix 下
    root = [
        os.path.join(args.data_root_prefix, p) if not os.path.isabs(p) else p
        for p in root
    ]
    carla_ds, carla_loader = make_loader(
        root=root,
        batch_size=args.num_reward_views,
        shuffle=True,
        num_workers=6,
    )
    optimizer, lr_scheduler, carla_loader = accelerator.prepare(
        optimizer, lr_scheduler, carla_loader
    )

    global_step = 0

    carla_iter = cycle_dataloader(carla_loader)
    prompt_text = args.prompt_text
    if args.sample_seed >= 0 and args.lora_dropout > 0:
        logger.warning(
            "--sample_seed 已固定但 --lora_dropout=%s：dropout 会让同一条轨迹每步采出不同的图，"
            "只优化单张纹理时建议设成 0。",
            args.lora_dropout,
        )
    while global_step < args.max_train_steps:
        batch = next(carla_iter)
        prompts = prompt_text
        # Euler 系 scheduler 带内部状态，每步重建
        scheduler = prepare_sampling_scheduler(
            pipe, args.num_inference_steps, accelerator.device
        )
        # 固定种子 -> 每个训练步走同一条采样轨迹，优化目标收敛到唯一一张纹理
        sample_generator = make_sample_generator(args.sample_seed, accelerator.device)
        with accelerator.accumulate(pipe.unet):
            (
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = encode_prompts(
                pipe,
                [prompts] * args.reward_batch_size,
                args.negative_prompt,
                do_classifier_free_guidance=args.guidance_scale > 1.0,
                device=accelerator.device,
                dtype=weight_dtype,
            )
            add_time_ids = build_sdxl_time_ids(
                batch_size=args.reward_batch_size,
                resolution=args.resolution,
                device=accelerator.device,
                dtype=weight_dtype,
            )
            # 手工展开去噪 + VAE 解码，保住 reward -> LoRA 的梯度链，(B, 3, H, W) in [0, 1]
            diffusion_images = sample_images_with_grad(
                pipe.unet,
                pipe.vae,
                scheduler,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                add_time_ids,
                accelerator.device,
                weight_dtype,
                args.reward_batch_size,
                args.resolution,
                args.guidance_scale,
                grad_last_steps=args.grad_last_steps,
                generator=sample_generator,
            )
            if args.use_diffusion_image_aug:
                diffusion_images = roa_diffusion_image(diffusion_images)

            reward_outputs = render_reward_multi_view_carla(
                texture_render_model=texture_render_model,
                pido=pido,
                yolo_reward_model=yolo_reward_model,
                diffusion_images=diffusion_images,
                carla_batch=batch,
                carla_dataset=carla_ds,
                num_views=args.num_reward_views,
                reward_mode=args.reward_mode,
                target_class_id=2,
                roa=roa,
                pido_scope=args.pido_scope,
                pido_apply_prob=args.pido_apply_prob,
            )
            rendered = reward_outputs["rendered_views"]
            rewardStatus = reward_outputs["reward_status"]
            reward_value = reward_outputs["reward_value"]
            reward_values = reward_outputs["reward_values"]

            # 每20步保存一次渲染结果和对应的UV纹理图
            if accelerator.is_main_process and global_step % 20 == 0:
                save_path = Path(args.output_dir) / "save" / str(global_step)
                save_path.mkdir(parents=True, exist_ok=True)

                texture_min = diffusion_images.detach().amin().item()
                texture_max = diffusion_images.detach().amax().item()
                texture_mean = diffusion_images.detach().mean().item()
                logger.info(
                    "Step %s: diffusion_images stats min=%.6f max=%.6f mean=%.6f",
                    global_step,
                    texture_min,
                    texture_max,
                    texture_mean,
                )
                uv_masked = texture_render_model.get_uv_masked(diffusion_images)
                tensor_rgb_to_pil(uv_masked).save(f"{save_path}/uv_texture.png")

                texture_image = tensor_rgb_to_pil(diffusion_images)
                texture_image.save(f"{save_path}/texture.png")

                rendered_image = tensor_rgb_to_pil(rendered[0])

                combined_image = Image.new(
                    "RGB",
                    (
                        rendered_image.width + texture_image.width,
                        max(rendered_image.height, texture_image.height),
                    ),
                )
                combined_image.paste(rendered_image, (0, 0))
                combined_image.paste(texture_image, (rendered_image.width, 0))
                combined_image.save(f"{save_path}/combined_images.png")

                log_prompt_file = f"{args.output_dir}/log.txt"
                with open(log_prompt_file, "a+") as f:
                    f.write(f"{global_step}_{prompts}\n")
                    f.flush()

            # clip正则化项
            loss_reg = args.lora_reg_weight * lora_regularization(pipe.unet)
            loss_reward = args.reward_weight * reward_value
            loss_reward_active = (
                loss_reward if rewardStatus else torch.zeros_like(loss_reward)
            )
            loss_reward_values_active = (
                reward_values if rewardStatus else torch.zeros_like(reward_values)
            )
            if not rewardStatus:
                logger.info(f"Step {global_step}: 攻击成功")

            if args.use_margin_loss:
                loss_reward_active2, frac_below_margin = margin_detection_loss(
                    # loss_reward_active,  # (V,) per-view tensor，现在派上用场
                    loss_reward_values_active,  # (V,) per-view tensor，现在派上用场
                    margin=args.margin,
                    loss_type="squared_hinge",
                    # loss_type="softplus",
                )
            else:
                loss_reward_active2 = loss_reward_values_active.mean()
                frac_below_margin = torch.tensor(-1.0)

            reward_loss = loss_reward_active2 * args.reward_weight
            loss = reward_loss

            if not rewardStatus:
                loss = loss_reg

            # end clip gate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            # optimizer.zero_grad(set_to_none=False)
            optimizer.zero_grad()
            if accelerator.is_main_process and global_step % 1 == 0:
                logger.info(
                    f"Step {global_step}: Loss={loss.detach().item():.4f}, "
                    f"LossReward={reward_loss.detach().item():.4f}, "
                    f"LossReward_raw={loss_reward_active.detach().item():.4f}, "
                )
                logger.info(f"Step {global_step}: Prompts: {prompts}")

        if accelerator.sync_gradients:
            global_step += 1
            logs = {
                "loss": loss.detach().item(),
                "loss_reward": reward_loss.detach().item(),
                "loss_reward_raw": loss_reward_active.detach().item(),
                "loss_reg": loss_reg.detach().item(),
                "reward": reward_value.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "FracBelowMargin": frac_below_margin.detach().item(),
            }
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
            # 每 N次 梯度步保存一次 LoRA 权重检查点，并进行评估
            if (
                accelerator.is_main_process
                and global_step % args.checkpointing_steps == 0
            ):
                prune_checkpoints(args.output_dir, args.checkpoints_total_limit)
                checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                unwrapped_unet = unwrap_model(accelerator, pipe.unet)
                # unwrapped_unet = unwrap_model(accelerator, pipe.unet).to(torch.float32)
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_unet)
                )
                StableDiffusionXLPipeline.save_lora_weights(
                    save_directory=checkpoint_dir,
                    unet_lora_layers=unet_lora_state_dict,
                )
                # 同时落盘该 checkpoint 对应的确定性纹理，方便回头挑最好的一张
                export_texture(
                    args,
                    pipe,
                    prompt_text,
                    accelerator.device,
                    weight_dtype,
                    checkpoint_dir,
                    texture_render_model,
                )
                torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 先导出纹理：下面那行会把整个 UNet 就地转成 fp32，采样 dtype 就对不上了
        export_texture(
            args,
            pipe,
            prompt_text,
            accelerator.device,
            weight_dtype,
            Path(args.output_dir),
            texture_render_model,
        )
        logger.info(
            "Final texture saved to %s (export_seed=%s, sample_seed=%s)",
            Path(args.output_dir) / "final_texture.png",
            resolve_export_seed(args),
            args.sample_seed,
        )
        final_unet = unwrap_model(accelerator, pipe.unet).to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(final_unet)
        )
        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
        )
        free_memory()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
