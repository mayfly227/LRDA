import random

import torch
import torch.nn.functional as F
import kornia.augmentation as KR
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import torch.nn as nn


class DifferentiableFog(nn.Module):
    """可微分雾化：基于大气散射模型 I_fog = I * t + A * (1 - t)

    透射率 t 使用可微分的 Perlin-like 随机场生成空间变化，
    而非均匀标量，更接近真实雾的非均匀性。
    """

    def __init__(
        self,
        t_range=(0.4, 0.9),
        atmos_light=(0.7, 0.95),
        spatial_var=0.1,
        blur_kernel=21,
        p=0.3,
    ):
        super().__init__()
        self.t_range = t_range
        self.atmos_light = atmos_light
        self.spatial_var = spatial_var  # 透射率空间变化幅度
        self.blur_kernel = blur_kernel  # 平滑核大小，控制雾的空间频率
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 逐样本概率mask
        mask = (torch.rand(B, 1, 1, 1, device=x.device) < self.p).float()

        # 采样全局透射率 t ∈ [t_min, t_max]
        t_base = torch.empty(B, 1, 1, 1, device=x.device).uniform_(*self.t_range)

        # 空间变化：低频随机场（模拟雾的不均匀）
        noise = torch.randn(B, 1, H, W, device=x.device) * self.spatial_var
        # 用大核高斯模糊使其低频化（可微分）
        k = self.blur_kernel
        padding = k // 2
        # 1D分离卷积高斯核，避免 O(k^2) 计算
        sigma = k / 6.0
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_h = gauss_1d.view(1, 1, 1, k).expand(1, 1, 1, k)
        kernel_v = gauss_1d.view(1, 1, k, 1).expand(1, 1, k, 1)
        noise = F.conv2d(noise, kernel_h, padding=(0, padding))
        noise = F.conv2d(noise, kernel_v, padding=(padding, 0))

        t_map = (t_base + noise).clamp(0.05, 1.0)  # (B,1,H,W)

        # 大气光
        A = torch.empty(B, 1, 1, 1, device=x.device).uniform_(*self.atmos_light)

        fog_img = x * t_map + A * (1.0 - t_map)
        return x * (1.0 - mask) + fog_img * mask

class ROADiffusionImage(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

        self.rotate = KR.RandomAffine(
            degrees=(-25.0, 25.0),
            translate=None,
            scale=None,
            keepdim=True,
            p=1.0,
        )
        self.translate = KR.RandomAffine(
            degrees=0.0,
            translate=(0.25, 0.1),
            scale=None,
            keepdim=True,
            p=1.0,
        )
        self.scale = KR.RandomAffine(
            degrees=0.0,
            translate=None,
            scale=(0.8, 1.2),
            keepdim=True,
            p=1.0,
        )

    def forward(self, x):
        if random.random() > self.p:
            return x.clamp(0.0, 1.0)

        # op = random.choice(["rotate", "translate", "scale"])
        op = "translate"

        if op == "rotate":
            x = self.rotate(x)
        elif op == "translate":
            x = self.translate(x)
        else:
            x = self.scale(x)

        return x.clamp(0.0, 1.0)

class ROA(nn.Module):
    def __init__(self):
        super().__init__()
        self.randomAffine = KR.RandomAffine(
            degrees=(-10.0, 10.0),
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            p=0.5,
            # padding_mode='border',
            keepdim=True,
        )
        self.colorJiggle = KR.ColorJiggle(
            brightness=(0.85, 1.2),
            contrast=(0.8, 1.25),
            saturation=0.2,
            hue=0.03,
            p=0.5,
            keepdim=True,
        )
        self.gaussianNoise = KR.RandomGaussianNoise(
            mean=0.0, std=0.03, p=0.25, keepdim=True
        )
        self.motionBlur = KR.RandomMotionBlur(
            kernel_size=(3, 7),
            angle=(-30, 30),
            direction=(-1, 1),
            p=0.3,
            keepdim=True,
        )
        self.fog = DifferentiableFog(
            t_range=(0.5, 0.8), atmos_light=(0.7, 0.9), spatial_var=0.05, p=0.5
        )
        self.rain = KR.RandomRain(
            number_of_drops=(500, 1500),
            drop_height=(5, 15),
            drop_width=(-5, 5),
            same_on_batch=False,
            p=0.5,
            keepdim=True,
        )
        self.snow = KR.RandomSnow(
            snow_coefficient=(0.05, 0.2),
            brightness=(1, 2),
            same_on_batch=False,
            p=0.5,
            keepdim=True,
        )

    def forward(self, x):
        x = self.randomAffine(x)
        if torch.isnan(x).any():
            return x.clamp(0.0, 1.0)

        x = self.colorJiggle(x)

        # x = random.choice([self.fog, self.rain, self.snow])(x)

        # 3) 光学系统
        # x = self.motionBlur(x)

        # # 4) 传感器
        # x = self.gaussianNoise(x)

        return x.clamp(0.0, 1.0)



class DifferentiableGaussianBlur(nn.Module):
    """可微高斯模糊：逐样本随机 sigma（封装 kornia RandomGaussianBlur）。

    模拟部署链路里的平滑项（TAA / mipmap / 光学 PSF / 下采样）。
    kernel_size 按 sigma 上限一次定死（保证 batch 内形状一致），
    border_type='reflect' 比 zero-padding 边界更自然。对输入完全可微。
    """

    def __init__(self, sigma_range=(0.3, 1.5), p=0.5):
        super().__init__()
        kernel_size = 2 * int(np.ceil(2.5 * sigma_range[1])) + 1
        self.blur = KR.RandomGaussianBlur(
            kernel_size=(kernel_size, kernel_size),
            sigma=sigma_range,
            border_type="reflect",
            separable=True,
            same_on_batch=False,
            p=p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blur(x)


class DifferentiableBlockCompress(nn.Module):
    """4x4 块量化近似 BC1 纹理压缩（UE4 默认压缩格式）。

    每个 4x4 块取 min/max 端点色（并按 RGB565 取整），块内像素硬指派到
    {c0, c1, (2c0+c1)/3, (c0+2c1)/3} 四色中最近者。前向硬量化（no_grad），
    反向用 STE 恒等回传梯度。
    """

    def __init__(self, p=0.5, block=4):
        super().__init__()
        self.p = p
        self.block = int(block)

    @staticmethod
    def _to_rgb565(c: torch.Tensor) -> torch.Tensor:
        r = torch.round(c[..., 0] * 31.0) / 31.0
        g = torch.round(c[..., 1] * 63.0) / 63.0
        b = torch.round(c[..., 2] * 31.0) / 31.0
        return torch.stack([r, g, b], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        mask = (torch.rand(B, 1, 1, 1, device=device) < self.p).float()

        bs = self.block
        Hc, Wc = (H // bs) * bs, (W // bs) * bs
        xc, xr = x[:, :, :Hc, :Wc], x[:, :, :Hc, Wc:]
        xb = x[:, :, Hc:, :]  # 底部余数行原样保留
        Hb, Wb = Hc // bs, Wc // bs

        with torch.no_grad():
            blocks = (
                xc.reshape(B, C, Hb, bs, Wb, bs)
                .permute(0, 1, 2, 4, 3, 5)
                .reshape(B, C, Hb, Wb, bs * bs)
            )
            px = blocks.permute(0, 2, 3, 4, 1)  # (B,Hb,Wb,16,C)
            c0 = self._to_rgb565(px.amax(dim=3))  # (B,Hb,Wb,C)
            c1 = self._to_rgb565(px.amin(dim=3))
            cand = torch.stack(
                [c0, c1, (2 * c0 + c1) / 3.0, (c0 + 2 * c1) / 3.0], dim=3
            )  # (B,Hb,Wb,4,C)

            # 最近候选色指派（用二次型展开避免显式构造 C 维差分）
            px2 = px.pow(2).sum(-1, keepdim=True)  # (B,Hb,Wb,16,1)
            c2 = cand.pow(2).sum(-1)  # (B,Hb,Wb,4)
            dot = torch.einsum("bhwpc,bhwkc->bhwpk", px, cand)
            idx = (px2 - 2 * dot + c2.unsqueeze(3)).argmin(dim=-1)  # (B,Hb,Wb,16)

            cand_e = cand.unsqueeze(3).expand(-1, -1, -1, bs * bs, -1, -1)
            idx_e = idx.view(B, Hb, Wb, bs * bs, 1, 1).expand(-1, -1, -1, -1, -1, C)
            q = torch.gather(cand_e, 4, idx_e).squeeze(4)  # (B,Hb,Wb,16,C)
            q = (
                q.permute(0, 4, 1, 2, 3)
                .reshape(B, C, Hb, Wb, bs, bs)
                .permute(0, 1, 2, 4, 3, 5)
                .reshape(B, C, Hc, Wc)
            )

        # STE: 前向取量化值，反向对 x 恒等
        out = xc + (q - xc).detach()
        if Wc < W:
            out = torch.cat([out, xr], dim=3)  # 右余数列直通
        if Hc < H:
            out = torch.cat([out, xb], dim=2)

        return x * (1 - mask) + out * mask


class DifferentiableToneJitter(nn.Module):
    """gamma / tonemap jitter：曝光 -> ACES 近似曲线混合 -> gamma。

    模拟 UE4 线性 HDR 到 sRGB 的显示变换扰动。对输入可微。
    """

    def __init__(
        self,
        p=0.5,
        gamma_range=(0.75, 1.33),
        exposure_range=(0.8, 1.25),
        aces_mix=(0.0, 0.6),
    ):
        super().__init__()
        self.p = p
        self.gamma_range = gamma_range
        self.exposure_range = exposure_range
        self.aces_mix = aces_mix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device = x.device
        mask = (torch.rand(B, 1, 1, 1, device=device) < self.p).float()
        gamma = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.gamma_range)
        exposure = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.exposure_range)
        mix = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.aces_mix)

        y = (x * exposure).clamp(0.0, 1.0)
        aces = (y * (2.51 * y + 0.03)) / (y * (2.43 * y + 0.59) + 0.14)
        y = y * (1 - mix) + aces * mix
        y = y.clamp(1e-4, 1.0) ** gamma
        return x * (1 - mask) + y * mask


class ROA_FCA(nn.Module):
    """FCA 训练用可微增广链。

    几何/颜色：randomAffine(scale) + colorJiggle（原有）。
    渲染链鲁棒性（2026-08 新增，模拟 UE4/CARLA 部署链路，不动部署侧配置）：
    gaussianBlur（TAA/mipmap/光学平滑）-> blockCompress（BC1 纹理压缩近似）
    -> toneJitter（曝光/ACES/gamma 显示变换）。各模块独立概率触发，
    概率与强度经构造函数可调，p=0 即关闭。
    """

    def __init__(
        self,
        blur_p=0.5,
        blur_sigma=(0.3, 1.5),
        bc_p=0.3,
        tone_p=0.3,
        gamma_range=(0.75, 1.33),
        exposure_range=(0.8, 1.25),
        aces_mix=(0.0, 0.6),
    ):
        super().__init__()
        self.rotate = KR.RandomAffine(
            degrees=(-15.0, 15.0),
            translate=None,
            scale=None,
            p=0.25,
            # padding_mode='border',
            keepdim=True,
        )
        self.translate = KR.RandomAffine(
            degrees=0.0,
            translate=(0.15, 0.15),
            scale=None,
            p=0.25,
            # padding_mode='border',
            keepdim=True,
        )
        self.scale = KR.RandomAffine(
            degrees=0.0,
            translate=None,
            scale=(0.8, 1.2),
            p=0.25,
            # padding_mode='border',
            keepdim=True,
        )
        self.colorJiggle = KR.ColorJiggle(
            brightness=(0.85, 1.2),
            contrast=(0.8, 1.25),
            saturation=0.2,
            hue=0.03,
            p=0.5,
            keepdim=True,
        )
        self.gaussianNoise = KR.RandomGaussianNoise(
            mean=0.0, std=0.03, p=0.25, keepdim=True
        )
        self.motionBlur = KR.RandomMotionBlur(
            kernel_size=(3, 7),
            angle=(-30, 30),
            direction=(-1, 1),
            p=0.3,
            keepdim=True,
        )
        self.blur = DifferentiableGaussianBlur(sigma_range=blur_sigma, p=blur_p)
        self.blockCompress = DifferentiableBlockCompress(p=bc_p)
        self.toneJitter = DifferentiableToneJitter(
            p=tone_p,
            gamma_range=gamma_range,
            exposure_range=exposure_range,
            aces_mix=aces_mix,
        )

    def forward(self, x):

        x = random.choice([self.rotate, self.translate, self.scale])(x)
        # x = self.rotate(x)
        # x = self.translate(x)
        # x = self.scale(x)
        if torch.isnan(x).any():
            return x.clamp(0.0, 1.0)

        # x = self.colorJiggle(x)
        # x = random.choice([self.fog, self.rain, self.snow])(x)

        # 渲染链失真（部署侧 UE4 效应的 EOT 近似）
        # x = self.blur(x)
        x = self.motionBlur(x)
        x = self.blockCompress(x)
        x = self.toneJitter(x)

        # 3) 光学系统
        # x = self.motionBlur(x)

        # # 4) 传感器
        # x = self.gaussianNoise(x)

        return x.clamp(0.0, 1.0)

class ROA_SDA(nn.Module):
    """训练用可微增广链。

    几何/颜色：randomAffine(scale) + colorJiggle（原有）。
    渲染链鲁棒性（2026-08 新增，模拟 UE4/CARLA 部署链路，不动部署侧配置）：
    gaussianBlur（TAA/mipmap/光学平滑）-> blockCompress（BC1 纹理压缩近似）
    -> toneJitter（曝光/ACES/gamma 显示变换）。各模块独立概率触发，
    概率与强度经构造函数可调，p=0 即关闭。
    """

    def __init__(
        self,
        blur_p=0.5,
        blur_sigma=(0.3, 1.5),
        bc_p=0.3,
        tone_p=0.3,
        gamma_range=(0.75, 1.33),
        exposure_range=(0.8, 1.25),
        aces_mix=(0.0, 0.6),
    ):
        super().__init__()
        self.rotate = KR.RandomAffine(
            degrees=(-15.0, 15.0),
            translate=None,
            scale=None,
            p=0.25,
            # padding_mode='border',
            keepdim=True,
        )
        self.translate = KR.RandomAffine(
            degrees=0.0,
            translate=(0.15, 0.15),
            scale=None,
            p=0.25,
            # padding_mode='border',
            keepdim=True,
        )
        self.scale = KR.RandomAffine(
            degrees=0.0,
            translate=None,
            scale=(0.8, 1.2),
            p=0.25,
            # padding_mode='border',
            keepdim=True,
        )
        self.colorJiggle = KR.ColorJiggle(
            brightness=(0.85, 1.2),
            contrast=(0.8, 1.25),
            saturation=0.2,
            hue=0.03,
            p=0.5,
            keepdim=True,
        )
        self.gaussianNoise = KR.RandomGaussianNoise(
            mean=0.0, std=0.03, p=0.25, keepdim=True
        )
        self.motionBlur = KR.RandomMotionBlur(
            kernel_size=(3, 7),
            angle=(-30, 30),
            direction=(-1, 1),
            p=0.3,
            keepdim=True,
        )
        self.blur = DifferentiableGaussianBlur(sigma_range=blur_sigma, p=blur_p)
        self.blockCompress = DifferentiableBlockCompress(p=bc_p)
        self.toneJitter = DifferentiableToneJitter(
            p=tone_p,
            gamma_range=gamma_range,
            exposure_range=exposure_range,
            aces_mix=aces_mix,
        )

    def forward(self, x):

        x = random.choice([self.rotate, self.translate, self.scale])(x)
        # x = self.rotate(x)
        # x = self.translate(x)
        # x = self.scale(x)
        if torch.isnan(x).any():
            return x.clamp(0.0, 1.0)

        # x = self.colorJiggle(x)
        # x = random.choice([self.fog, self.rain, self.snow])(x)

        # 渲染链失真（部署侧 UE4 效应的 EOT 近似）
        # x = self.blur(x)
        # x = self.motionBlur(x)
        # x = self.blockCompress(x)
        x = self.toneJitter(x)

        # 3) 光学系统
        # x = self.motionBlur(x)

        # # 4) 传感器
        # x = self.gaussianNoise(x)

        return x.clamp(0.0, 1.0)

# ── 工具函数 ──────────────────────────────────────────────────────
def load_test_image(path):
    img = Image.open(path).convert("RGB")
    return img


def img_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → (1, C, H, W) float32 [0,1]"""
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) → (H, W, C) uint8"""
    return (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


# ── 主测试 ────────────────────────────────────────────────────────
def run_aug_comparison(n_samples: int = 4, seed: int = 42):
    # torch.manual_seed(seed)
    roa = ROA()

    orig_img = load_test_image(
        "/home/dk/project/FastAdv/rewardyolo/1_rendered_images_0.png"
    )
    orig_t = img_to_tensor(orig_img)  # (1,3,256,256)

    # 生成 n_samples 次增广结果（每次 forward 随机）
    aug_results = []
    nan_count = 0
    for _ in range(n_samples):
        with torch.no_grad():
            out = roa(orig_t.clone())
        if torch.isnan(out).any():
            nan_count += 1
            out = orig_t.clone()  # NaN fallback：用原图填充
        im = tensor_to_np(out)
        Image.fromarray(im).save(f"img/aug_sample_{_ + 1}.png")
        aug_results.append(im)
    return 

    # ── 绘图 ────────────────────────────────────────────────────
    cols = 4
    rows = (n_samples // cols) + 1  # 第一行放原图 + 统计
    fig = plt.figure(figsize=(cols * 3.2, rows * 3.2), facecolor="#0e0e0e")
    gs = gridspec.GridSpec(rows, cols, hspace=0.35, wspace=0.15)

    # 第一行：原图占一格，其余留白放统计文本
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(np.array(orig_img))
    ax0.set_title("Original", color="white", fontsize=10, pad=4)
    ax0.axis("off")
    for spine in ax0.spines.values():
        spine.set_edgecolor("#00ff99")
        spine.set_linewidth(2)

    # 统计信息
    ax_stat = fig.add_subplot(gs[0, 1:])
    ax_stat.axis("off")

    # 统计每张增广图的像素统计
    diffs = [
        np.abs(r.astype(float) - np.array(orig_img).astype(float)).mean()
        for r in aug_results
    ]
    stat_text = (
        f"Samples: {n_samples}    NaN triggered: {nan_count}\n"
        f"Mean pixel Δ  →  avg={np.mean(diffs):.2f}  "
        f"min={np.min(diffs):.2f}  max={np.max(diffs):.2f}\n"
        f"Augmentation prob (affine OR jiggle): ~75%\n"
        f"Scale range: 0.7–1.0   Translate: ±10%   Degrees: ±15°"
    )
    ax_stat.text(
        0.02,
        0.5,
        stat_text,
        transform=ax_stat.transAxes,
        color="#aaaaaa",
        fontsize=9,
        va="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a", edgecolor="#444"),
    )

    # 增广样本
    for i, aug_np in enumerate(aug_results):
        row = (i // cols) + 1
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(aug_np)

        diff = diffs[i]
        color = "#ff4444" if diff > 20 else "#ffaa00" if diff > 8 else "#00ff99"
        ax.set_title(f"Aug #{i + 1}  Δ={diff:.1f}", color=color, fontsize=8, pad=3)
        ax.axis("off")

    fig.suptitle(
        "ROA Augmentation — Before / After",
        color="white",
        fontsize=13,
        y=0.98,
        fontweight="bold",
    )

    plt.savefig(
        "roa_aug_comparison.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.show()
    print(f"\nSaved: roa_aug_comparison.png")
    print(f"NaN events: {nan_count}/{n_samples} ({100 * nan_count / n_samples:.0f}%)")
    print(f"Mean pixel diff per sample: {[f'{d:.1f}' for d in diffs]}")


if __name__ == "__main__":
    import kornia.augmentation as KR
    import torch

    x = torch.rand(1, 3, 64, 64)

    affine = KR.RandomAffine(degrees=15.0, p=1.0).eval()
    jiggle = KR.ColorJiggle(0.5, (0.5, 2.0), p=1.0).eval()

    print((affine(x) - x).abs().mean())  # ≈ 0.0  → eval 生效，恒等
    print((jiggle(x) - x).abs().mean())  # > 0.0  → eval 无效，仍变换
    run_aug_comparison(n_samples=20)
