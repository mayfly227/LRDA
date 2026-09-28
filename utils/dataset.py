import argparse
import glob
import os
import re

import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    DirectionalLights,
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.transforms.transform3d import RotateAxisAngle
from torch.utils.data import DataLoader, Dataset


def _parse_pitch_yaw(stem):
    p, y = stem.split("_")
    return float(p), float(y)


def _dist_from_path(path):
    m = re.search(r"([\d.]+)m(?:[/\\]|$)", path.replace("\\", "/"))
    if not m:
        raise ValueError(f"路径无法解析距离: {path}")
    return float(m.group(1))


def _normalize_roots(root):
    if isinstance(root, (str, os.PathLike)):
        roots = [os.fspath(root)]
    else:
        try:
            roots = [os.fspath(r) for r in root]
        except TypeError as exc:
            raise TypeError("`root` must be a path or a sequence of paths.") from exc
    if not roots:
        raise ValueError("`root` must contain at least one path.")
    return roots


ELEV_SIGN = 1.0  # CARLA pitch -> P3D elev; 上下颠倒改 -1
AZIM_SIGN = -1.0  # 左右镜像 / 转向相反改 -1
HEADING_OFFSET = 0.0  # deg, azim 偏移 (用 MESH_YAW_FIX 修朝向时保持 0)
MESH_YAW_FIX = 270.0  # deg, 绕 up(Y) 轴修车头朝向; 俯视向右转 90, 转反用 +90
SCALE = 1.05  # mesh 非米制时缩放 (如 cm -> 0.01)
LIGHT_DIR = (0.3, -1.0, 0.3)


class CarlaRenderDataset(Dataset):
    # P3D world (Y-up,Z-fwd,X-right) 方向光来向, 仅观感
    def __init__(
        self,
        root,
        fov_h_deg,
        distance_source="npz",
        const_distance=None,
        elev_sign=1.0,
        azim_sign=-1.0,
        heading_offset=0.0,
        image_size=None,
        recursive=True,
    ):
        """
        root            : 数据根或数据根列表; recursive=True 时递归收集所有 .npz
        fov_h_deg       : CARLA 水平 FOV (度), 各距离共用
        distance_source : "npz" | "path" | "calc" | "const"
        const_distance  : distance_source=="const" 时使用
        """
        self.roots = _normalize_roots(root)
        self.root = (
            self.roots[0] if len(self.roots) == 1 else os.path.commonpath(self.roots)
        )
        self.fov_h = float(fov_h_deg)
        self.distance_source = distance_source
        self.const_distance = const_distance
        self.elev_sign = float(elev_sign)
        self.azim_sign = float(azim_sign)
        self.heading_offset = float(heading_offset)

        npz_files = []
        for data_root in self.roots:
            pat = (
                os.path.join(data_root, "**", "*.npz")
                if recursive
                else os.path.join(data_root, "*.npz")
            )
            npz_files.extend(glob.glob(pat, recursive=recursive))
        self.npz_files = sorted(npz_files)
        if not self.npz_files:
            raise RuntimeError(f"未找到 npz: {self.roots}")

        if distance_source == "const" and const_distance is None:
            raise ValueError("distance_source='const' 需提供 const_distance")

        if image_size is None:
            s = os.path.splitext(self.npz_files[0])[0] + ".png"
            self.W, self.H = Image.open(s).size
        else:
            self.W, self.H = image_size

        self.aspect = self.W / self.H
        self.fov_v = float(
            np.degrees(
                2.0 * np.arctan(np.tan(np.radians(self.fov_h) / 2.0) / self.aspect)
            )
        )

    def make_cameras(self, pitch, yaw, veh_yaw, distance, fov_h_deg, W, H, device):
        elev = ELEV_SIGN * pitch
        azim = AZIM_SIGN * (yaw - veh_yaw) + HEADING_OFFSET
        R, T = look_at_view_transform(
            dist=distance, elev=elev, azim=azim, degrees=True, device=device
        )
        aspect = W / H
        fov_v = np.degrees(
            2.0 * np.arctan(np.tan(np.radians(fov_h_deg) / 2.0) / aspect)
        )  # 水平->垂直
        return FoVPerspectiveCameras(
            R=R, T=T, fov=fov_v, aspect_ratio=aspect, degrees=True, device=device
        )

    def _resolve_distance(self, npz_path, d):
        src = self.distance_source
        if src == "const":
            return float(self.const_distance)
        if src == "path":
            return _dist_from_path(npz_path)
        if src == "npz":
            # 优先精确 distance 字段; 缺则回退 bbox_center 反算; 再缺回退 actor 原点
            if "distance" in d.files:
                return float(d["distance"])
            if "bbox_center" in d.files:  # 用 bbox 中心, 无 actor 偏差
                return float(np.linalg.norm(d["cam_trans"][0] - d["bbox_center"]))
            return float(np.linalg.norm(d["cam_trans"][0] - d["veh_trans"][0]))
        if src == "calc":
            # actor 原点反算 -> 带 ~|bb.location| 偏差; 有 bbox_center 时自动优先
            ref = d["bbox_center"] if "bbox_center" in d.files else d["veh_trans"][0]
            return float(np.linalg.norm(d["cam_trans"][0] - ref))
        raise ValueError(f"未知 distance_source: {src}")

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        pitch, yaw = _parse_pitch_yaw(stem)

        d = np.load(npz_path)
        veh_yaw = float(d["veh_trans"][1, 1])
        dist = self._resolve_distance(npz_path, d)
        # elev = ELEV_SIGN * pitch
        # azim = AZIM_SIGN * (yaw - veh_yaw) + HEADING_OFFSET
        elev = self.elev_sign * pitch
        azim = self.azim_sign * (yaw - veh_yaw) + self.heading_offset
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=True)

        img = Image.open(os.path.splitext(npz_path)[0] + ".png").convert("RGB")
        image = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)

        return {
            "image": image,  # (3,H,W) sRGB 含车画布
            "R": R[0],
            "T": T[0],  # (3,3),(3,)
            "elev": torch.tensor(elev, dtype=torch.float32),
            "azim": torch.tensor(azim, dtype=torch.float32),
            "dist": torch.tensor(dist, dtype=torch.float32),
            "pitch": torch.tensor(pitch, dtype=torch.float32),
            "yaw": torch.tensor(yaw, dtype=torch.float32),
            "veh_yaw": torch.tensor(veh_yaw, dtype=torch.float32),
            "filename": os.path.relpath(npz_path, self.root)[:-4] + ".png",
        }


def build_cameras(R, T, dataset, device):
    return FoVPerspectiveCameras(
        R=R.to(device),
        T=T.to(device),
        fov=dataset.fov_v,
        aspect_ratio=dataset.aspect,
        degrees=True,
        device=device,
    )


def make_loader(
    root, fov_h_deg=90, batch_size=4, shuffle=True, num_workers=4, **ds_kwargs
):
    ds = CarlaRenderDataset(root, fov_h_deg, **ds_kwargs)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return ds, dl


def load_mesh(path, device):
    mesh = load_objs_as_meshes([path], device=device)
    verts = mesh.verts_packed()
    center = (verts.max(0).values + verts.min(0).values) / 2.0
    verts = (verts - center) * SCALE
    # ── 绕 up 轴修正车头朝向 ──────────────────────────────────────────────
    Rfix = (
        RotateAxisAngle(angle=MESH_YAW_FIX, axis="Y", degrees=True)
        .get_matrix()[0, :3, :3]
        .to(device)
    )
    verts = verts @ Rfix  # 行向量约定: p @ R
    # ────────────────────────────────────────────────────────────────────
    mesh = mesh.update_padded(verts[None])
    if mesh.textures is None:
        mesh.textures = TexturesVertex(
            verts_features=torch.full_like(mesh.verts_padded(), 0.6)
        )
    return mesh


def linear_to_srgb(x):
    a = 0.055
    return torch.where(
        x <= 0.0031308, 12.92 * x, (1 + a) * x.clamp(min=1e-8) ** (1 / 2.4) - a
    )


def render_rgba(mesh, cameras, device):
    H, W = 1024, 1024
    raster = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1
    )
    lights = DirectionalLights(device=device, direction=[list(LIGHT_DIR)])
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster)
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
    frags = rasterizer(mesh)
    rgb = shader(frags, mesh)[0, ..., :3]  # (H,W,3) linear
    alpha = (frags.pix_to_face[0, ..., 0] >= 0).float()[..., None]  # (H,W,1) 二值
    return linear_to_srgb(rgb), alpha


def save_img(t, path):
    Image.fromarray((t.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)).save(
        path
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/dk/project/FastAdv/composecarla/10m")
    ap.add_argument("--fov", type=float, default=90)
    ap.add_argument(
        "--distance_source", default="path", choices=["npz", "path", "calc", "const"]
    )
    ap.add_argument("--const_distance", type=float, default=None)
    args = ap.parse_args()
    ds, dl = make_loader(
        args.root,
        args.fov,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        distance_source=args.distance_source,
        const_distance=args.const_distance,
    )
    print(f"N={len(ds)} W={ds.W} H={ds.H} fov_v={ds.fov_v:.3f}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # mesh = load_mesh("/home/dk/project/FastAdv/carmodel/audiadjust/Etron_adjust.obj",device)
    mesh = load_mesh(
        "/home/dk/project/FastAdv/carmodel/audi/pytorch3d_Etron.obj", device
    )
    for i, batch in enumerate(dl):
        image = batch["image"]
        image = image[0, ...].permute((1, 2, 0))
        image = image.to(device)
        cam = build_cameras(batch["R"], batch["T"], ds, device=device)
        rgb, alpha = render_rgba(mesh, cam, device)
        # red = torch.tensor([1.0, 0.0, 0.0], device=device)
        red = torch.tensor([1.0, 0.0, 0.0], device=device)
        comp = (1 - 0.5 * alpha) * image + 0.5 * alpha * red
        # comp = alpha * rgb + (1 - alpha) * image

        save_img(comp, f"{i}.png")

        print(image.shape)

    # b = next(iter(dl))
    # for k, v in b.items():

    #     print(k, getattr(v, "shape", v))
    # print("dist 样例:", b["dist"])
