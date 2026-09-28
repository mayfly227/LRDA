import os
import random
import subprocess
import time

import numpy as np
import torch
from PIL import Image


def load_uv_mask(mask_path: str, texture_image_size: int, device) -> torch.Tensor:
    mask = Image.open(mask_path).convert("RGB")
    mask = mask.resize(
        (texture_image_size, texture_image_size), Image.Resampling.NEAREST
    )

    mask_array = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
    mask_tensor = (
        mask_array.view(texture_image_size, texture_image_size, 3)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    mask_tensor = (mask_tensor > 0.5).float()

    return mask_tensor.to(device=device)


def pil_from_tensor(image: torch.Tensor) -> Image.Image | list[Image.Image]:
    image = image.detach().cpu().clamp(0, 1)

    if image.ndim == 4:
        return [pil_from_tensor(img) for img in image][0]

    if image.ndim != 3:
        raise ValueError(
            f"`pil_from_tensor` expects a 3D or 4D tensor, got shape {tuple(image.shape)}"
        )

    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    elif image.shape[-1] not in (1, 3):
        raise ValueError(
            f"`pil_from_tensor` expects CHW or HWC image with 1 or 3 channels, got shape {tuple(image.shape)}"
        )

    image = (image * 255).round().byte().numpy()
    if image.shape[-1] == 1:
        image = image[..., 0]
    return Image.fromarray(image)


def tensor_from_pil(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    return image_tensor.permute(2, 0, 1)


def get_free_gpu(
    threshold_mem=0.1,
    threshold_util=10,
    max_wait_seconds=None,
    verbose=True,
):
    """
    随机获取一张空闲GPU，并设置 CUDA_VISIBLE_DEVICES

    Args:
        threshold_mem (float): 显存占用比例阈值，例如 0.1 表示 <10%
        threshold_util (int): GPU利用率阈值，例如 10 表示 <10%
        max_wait_seconds (int | None): 最多轮询等待的秒数；None 表示只检查一次
        verbose (bool): 是否打印信息

    Returns:
        int: 选中的 GPU id
    """

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        deadline = None if max_wait_seconds is None else time.time() + max_wait_seconds

        while True:
            result = subprocess.check_output(cmd).decode("utf-8").strip().split("\n")

            free_gpus = []

            for line in result:
                idx, mem_used, mem_total, util = map(int, line.split(","))

                mem_ratio = mem_used / mem_total

                if mem_ratio < threshold_mem and util < threshold_util:
                    free_gpus.append(idx)

            if len(free_gpus) > 0:
                selected_gpu = random.choice(free_gpus)
                break

            if deadline is not None and time.time() >= deadline:
                raise RuntimeError(
                    "No free GPU available under given thresholds within the waiting window."
                )

            if verbose:
                print(
                    f"[INFO] No free GPU available yet. Retrying in 1 second... "
                    f"threshold_mem={threshold_mem}, threshold_util={threshold_util}"
                )

            time.sleep(1)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)

        if verbose:
            print(f"[INFO] Available GPUs: {free_gpus}")
            print(f"[INFO] Selected GPU: {selected_gpu}")

        return selected_gpu

    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found. Ensure NVIDIA driver is installed.")
    except Exception as e:
        raise RuntimeError(f"Failed to get GPU info: {e}")
