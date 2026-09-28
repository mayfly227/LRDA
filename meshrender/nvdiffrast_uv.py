from __future__ import annotations

from pathlib import Path

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch3d.transforms.transform3d import RotateAxisAngle
from torch import nn

CARLA_MESH_YAW_FIX = 270.0
CARLA_MESH_SCALE = 1.05


def _parse_obj_index(index_text: str, length: int) -> int:
    index = int(index_text)
    if index > 0:
        return index - 1
    if index < 0:
        return length + index
    raise ValueError("OBJ indices are 1-based; index 0 is invalid.")


def _parse_face_token(token: str, num_verts: int, num_uvs: int) -> tuple[int, int]:
    parts = token.split("/")
    if len(parts) < 2 or parts[1] == "":
        raise RuntimeError(
            "OBJ face is missing UV indices; nvdiffrast UV rendering requires vt data."
        )
    vert_index = _parse_obj_index(parts[0], num_verts)
    uv_index = _parse_obj_index(parts[1], num_uvs)
    return vert_index, uv_index


def load_obj_with_expanded_uvs(
    obj_path: str | Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load OBJ vertices/UVs and expand vertices by (vertex_idx, uv_idx) pairs.

    PyTorch3D can keep geometry and UV indices separate. nvdiffrast uses one
    triangle index buffer for all interpolated attributes, so shared geometry
    vertices with different UVs must be split.
    """
    obj_path = Path(obj_path)
    raw_verts: list[list[float]] = []
    raw_uvs: list[list[float]] = []
    expanded_pairs: dict[tuple[int, int], int] = {}
    expanded_verts: list[list[float]] = []
    expanded_uvs: list[list[float]] = []
    triangles: list[list[int]] = []

    with obj_path.open("r", encoding="utf-8", errors="ignore") as obj_file:
        for line in obj_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            prefix = parts[0]
            if prefix == "v" and len(parts) >= 4:
                raw_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif prefix == "vt" and len(parts) >= 3:
                raw_uvs.append([float(parts[1]), float(parts[2])])
            elif prefix == "f" and len(parts) >= 4:
                face_indices = []
                for token in parts[1:]:
                    pair = _parse_face_token(token, len(raw_verts), len(raw_uvs))
                    if pair not in expanded_pairs:
                        expanded_pairs[pair] = len(expanded_verts)
                        expanded_verts.append(raw_verts[pair[0]])
                        expanded_uvs.append(raw_uvs[pair[1]])
                    face_indices.append(expanded_pairs[pair])

                for tri_index in range(1, len(face_indices) - 1):
                    triangles.append(
                        [
                            face_indices[0],
                            face_indices[tri_index],
                            face_indices[tri_index + 1],
                        ]
                    )

    if not raw_verts:
        raise RuntimeError(f"OBJ has no vertices: {obj_path}")
    if not raw_uvs:
        raise RuntimeError(f"OBJ has no UV coordinates: {obj_path}")
    if not triangles:
        raise RuntimeError(f"OBJ has no triangulatable UV faces: {obj_path}")

    verts = torch.tensor(expanded_verts, dtype=torch.float32, device=device)
    uvs = torch.tensor(expanded_uvs, dtype=torch.float32, device=device)
    tris = torch.tensor(triangles, dtype=torch.int32, device=device)
    return verts, uvs, tris


class NvdiffrastTextureRenderModel(nn.Module):
    """Pure-albedo UV renderer with the same forward shape as TextureRenderModel.

    The forward method accepts a PyTorch3D FoVPerspectiveCameras instance because
    the CARLA dataset already produces camera R/T in that convention. Projection
    matrices are reused, while rasterization, UV interpolation, and texture
    sampling are handled by nvdiffrast.
    """

    def __init__(
        self,
        device,
        uv_mask: torch.Tensor,
        render_image_size: int,
        texture_image_size: int,
        background_fill_value: float,
        mesh_obj_path: str = "pytorch3d_Etron.obj",
        bg2_path: str = "/home/dk/project/FastAdv/carmodel/audi/img_optim.png",
        use_carla_mesh_transform: bool = False,
        mesh_yaw_fix: float = CARLA_MESH_YAW_FIX,
        mesh_scale: float = CARLA_MESH_SCALE,
        znear: float = 0.01,
        zfar: float = 100.0,
        flip_x: bool = True,
        flip_output_y: bool = True,
        flip_uv_v: bool = True,
        use_antialias: bool = True,
        texture_filter_mode: str = "linear-mipmap-linear",
        texture_max_mip_level: int | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.render_image_size = int(render_image_size)
        self.texture_image_size = int(texture_image_size)
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.flip_x = bool(flip_x)
        self.flip_output_y = bool(flip_output_y)
        self.flip_uv_v = bool(flip_uv_v)
        self.use_antialias = bool(use_antialias)
        self.texture_filter_mode = texture_filter_mode
        self.use_texture_derivatives = "mipmap" in texture_filter_mode
        self.texture_max_mip_level = texture_max_mip_level
        self._glctx = None

        verts, uvs, tris = load_obj_with_expanded_uvs(mesh_obj_path, self.device)
        if use_carla_mesh_transform:
            center = (verts.max(0).values + verts.min(0).values) / 2.0
            verts = (verts - center) * mesh_scale
            rfix = (
                RotateAxisAngle(
                    angle=mesh_yaw_fix,
                    axis="Y",
                    degrees=True,
                )
                .get_matrix()[0, :3, :3]
                .to(self.device)
            )
            verts = verts @ rfix

        if self.flip_uv_v:
            uvs = uvs.clone()
            uvs[:, 1] = 1.0 - uvs[:, 1]

        self.register_buffer("verts", verts, persistent=False)
        self.register_buffer("uvs", uvs, persistent=False)
        self.register_buffer("tris", tris, persistent=False)
        self.register_buffer(
            "uv_mask", uv_mask.unsqueeze(0).to(self.device), persistent=False
        )
        self.register_buffer(
            "background_rgb",
            torch.full(
                (1, 3, self.texture_image_size, self.texture_image_size),
                float(background_fill_value),
                device=self.device,
            ),
            persistent=False,
        )

        bg2_path = Path(bg2_path)
        if bg2_path.exists():
            bg2_image = (
                Image.open(bg2_path)
                .convert("RGB")
                .resize(
                    (self.texture_image_size, self.texture_image_size),
                    Image.Resampling.BILINEAR,
                )
            )
            bg2 = torch.from_numpy(np.asarray(bg2_image).astype(np.float32) / 255.0)
            bg2 = bg2.permute(2, 0, 1).unsqueeze(0)
        else:
            bg2 = self.background_rgb.detach().cpu()
        self.register_buffer("background_rgb2", bg2.to(self.device), persistent=False)

    def _get_glctx(self):
        if self._glctx is None:
            if not self.verts.is_cuda:
                raise RuntimeError(
                    "nvdiffrast requires CUDA tensors; run this backend on a CUDA device."
                )
            self._glctx = dr.RasterizeCudaContext(device=self.verts.device)
        return self._glctx

    def _prepare_background(self, background, batch_size: int) -> torch.Tensor | None:
        if background is None:
            return None

        if isinstance(background, Image.Image):
            array = np.asarray(background.convert("RGB")).astype(np.float32) / 255.0
            bg = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(background, torch.Tensor):
            bg = background.float()
            if bg.ndim == 3:
                bg = bg.unsqueeze(0)
            if bg.ndim != 4:
                raise ValueError(
                    f"background tensor must be 3-D or 4-D, got {bg.ndim}-D"
                )
            if bg.max() > 1.0 + 1e-3:
                bg = bg / 255.0
        else:
            raise TypeError(
                f"background must be None, PIL.Image, or torch.Tensor, got {type(background)}"
            )

        bg = bg.to(device=self.verts.device, dtype=torch.float32)
        target_size = (self.render_image_size, self.render_image_size)
        if bg.shape[-2:] != target_size:
            bg = F.interpolate(
                bg, size=target_size, mode="bilinear", align_corners=False
            )
        bg = bg.clamp(0.0, 1.0)

        if bg.shape[0] == 1 and batch_size > 1:
            bg = bg.expand(batch_size, -1, -1, -1)
        elif bg.shape[0] != batch_size:
            raise ValueError(
                f"background batch size {bg.shape[0]} != input batch size {batch_size}"
            )

        return bg.permute(0, 2, 3, 1).contiguous()

    def prepare_uv_texture(self, texture_image: torch.Tensor) -> torch.Tensor:
        texture = F.interpolate(
            texture_image.float(),
            size=(self.texture_image_size, self.texture_image_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        uv_mask = self.uv_mask.to(device=texture.device, dtype=texture.dtype)
        bg2 = self.background_rgb2.to(device=texture.device, dtype=texture.dtype)
        return texture * uv_mask + bg2 * (1.0 - uv_mask)

    def _camera_clip_space(self, cameras, batch_size: int) -> torch.Tensor:
        verts = self.verts.to(dtype=torch.float32)
        verts_h = torch.cat([verts, torch.ones_like(verts[:, :1])], dim=-1)
        verts_h = verts_h.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        transform = cameras.get_full_projection_transform(
            znear=self.znear,
            zfar=self.zfar,
        )
        matrix = transform.get_matrix().to(device=verts_h.device, dtype=verts_h.dtype)
        if matrix.shape[0] == 1 and batch_size > 1:
            matrix = matrix.expand(batch_size, -1, -1)
        elif matrix.shape[0] != batch_size:
            raise ValueError(
                f"camera batch size {matrix.shape[0]} != render batch size {batch_size}"
            )

        pos_clip = torch.bmm(verts_h, matrix)
        if self.flip_x:
            pos_clip = pos_clip.clone()
            pos_clip[..., 0] = -pos_clip[..., 0]
        return pos_clip.contiguous()

    def _sample_texture(
        self,
        uv_bchw: torch.Tensor,
        rast: torch.Tensor,
        rast_db: torch.Tensor | None,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uv_bhwc = uv_bchw.permute(0, 2, 3, 1).contiguous()
        if uv_bhwc.shape[0] == 1 and batch_size > 1:
            uv_bhwc = uv_bhwc.expand(batch_size, -1, -1, -1).contiguous()
        elif uv_bhwc.shape[0] != batch_size:
            raise ValueError(
                f"texture batch size {uv_bhwc.shape[0]} != render batch size {batch_size}"
            )

        uv_attr = self.uvs.to(device=uv_bhwc.device, dtype=uv_bhwc.dtype)
        uv_attr = uv_attr.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        if self.use_texture_derivatives:
            max_mip_level = self.texture_max_mip_level
            if max_mip_level is None:
                max_mip_level = self._safe_max_mip_level(
                    width=int(uv_bhwc.shape[2]),
                    height=int(uv_bhwc.shape[1]),
                )
            uv, uv_da = dr.interpolate(
                uv_attr,
                rast,
                self.tris,
                rast_db=rast_db,
                diff_attrs="all",
            )
            rgb = dr.texture(
                uv_bhwc,
                uv,
                uv_da=uv_da,
                filter_mode=self.texture_filter_mode,
                boundary_mode="clamp",
                max_mip_level=max_mip_level,
            )
        else:
            uv, _ = dr.interpolate(uv_attr, rast, self.tris)
            rgb = dr.texture(
                uv_bhwc,
                uv,
                filter_mode=self.texture_filter_mode,
                boundary_mode="clamp",
            )
        alpha = (rast[..., 3:4] > 0).to(rgb.dtype)
        return rgb, alpha

    @staticmethod
    def _safe_max_mip_level(width: int, height: int) -> int:
        """Highest mip level nvdiffrast can build without downsampling odd extents."""
        level = 0
        while width > 1 or height > 1:
            if (width > 1 and width % 2 != 0) or (height > 1 and height % 2 != 0):
                break
            width = max(1, width // 2)
            height = max(1, height // 2)
            level += 1
        return level

    def get_uv_masked(self, texture_image: torch.Tensor):
        return self.prepare_uv_texture(texture_image)

    def forward(
        self,
        cameras,
        texture_image: torch.Tensor,
        background=None,
        return_rgba: bool = False,
    ) -> torch.Tensor:
        uv_texture = self.prepare_uv_texture(texture_image)
        batch_size = max(int(uv_texture.shape[0]), int(cameras.R.shape[0]))
        pos_clip = self._camera_clip_space(cameras, batch_size)

        rast, rast_db = dr.rasterize(
            self._get_glctx(),
            pos_clip,
            self.tris,
            resolution=(self.render_image_size, self.render_image_size),
            grad_db=self.use_texture_derivatives,
        )
        rgb, alpha = self._sample_texture(uv_texture, rast, rast_db, batch_size)

        if self.use_antialias:
            rgba = torch.cat([rgb, alpha], dim=-1)
            rgba = dr.antialias(
                rgba,
                rast,
                pos_clip,
                self.tris,
            ).clamp(0.0, 1.0)
            rgb = rgba[..., :3]
            alpha = rgba[..., 3:4]

        if self.flip_output_y:
            rgb = torch.flip(rgb, dims=[1])
            alpha = torch.flip(alpha, dims=[1])

        if return_rgba:
            return torch.cat((rgb, alpha), dim=-1).clamp(0.0, 1.0)

        bg = self._prepare_background(background, batch_size=batch_size)
        if bg is None:
            return rgb.clamp(0.0, 1.0)

        return (rgb * alpha + bg * (1.0 - alpha)).clamp(0.0, 1.0)
