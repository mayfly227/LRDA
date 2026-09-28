from torch import nn
import os
import numpy as np
import torch
from PIL import Image

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    SoftSilhouetteShader,
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
)
from pytorch3d.transforms.transform3d import RotateAxisAngle
import torch.nn.functional as F
import torchvision.transforms.functional as TF

CARLA_MESH_YAW_FIX = 270.0
CARLA_MESH_SCALE = 1.05


def pil_to_tensor_rgb(path, device, image_size=None):
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).to(device)  # (H, W, 3)
    return ten


def tensor_rgb_to_pil(t):
    t = t.detach().cpu().clamp(0.0, 1.0).numpy()
    # 支持 (H, W, 3) 或 (B, C, H, W) 或 (B, H, W, C) 等格式
    if t.ndim == 4:
        # (B, C, H, W) -> (H, W, 3) 取第一张图
        if t.shape[1] == 3:
            t = t[0].transpose(1, 2, 0)
        # (B, H, W, C) -> (H, W, 3) 取第一张图
        else:
            t = t[0]
    arr = (t * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def build_renderer(
    device,
    image_size=1024,
    dist=5.0,
    elev=20.0,
    azim=150.0,
    bin_size=0,
    max_faces_per_bin=200000,
):
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
    )
    lights = PointLights(device=device, location=[[0.0, 3.0, -3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    return renderer


def load_mesh_with_explicit_uv(obj_path, texture_path, device):
    # 显式读取几何 + UV 索引，不自动加载材质贴图
    verts, faces, aux = load_obj(
        obj_path,
        load_textures=False,
        create_texture_atlas=False,
        device=device,
    )

    if aux.verts_uvs is None or faces.textures_idx is None:
        raise RuntimeError("OBJ 没有有效 UV 数据，无法使用 TexturesUV。")

    tex_rgb = pil_to_tensor_rgb(texture_path, device)  # (H, W, 3)
    tex_map = tex_rgb[None, ...]  # (1, H, W, 3)

    textures = TexturesUV(
        maps=tex_map,
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=aux.verts_uvs[None, ...],
    )

    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=textures,
    )

    return mesh, faces, aux, tex_map


def render_and_save(mesh, renderer, out_path):
    images = renderer(mesh)  # (1, H, W, 4)
    rgb = images[0, ..., :3]
    tensor_rgb_to_pil(rgb).save(out_path)
    print(f"渲染图已保存: {os.path.abspath(out_path)}")
    return rgb


# =====================================
# Texture Model（可替换 / 可训练）
# =====================================
class TextureModel(nn.Module):
    def __init__(self, device, texture=None, trainable=False):
        super().__init__()
        self.device = device

        tex = self._to_tensor(texture) if texture is not None else None

        if tex is not None:
            if trainable:
                self.tex = nn.Parameter(tex)
            else:
                self.register_buffer("tex", tex)
        else:
            self.tex = None

    # =====================================
    # 核心：统一输入 → (B,H,W,3)
    # =====================================
    def _to_tensor(self, texture):

        # ---------- 1. path ----------
        if isinstance(texture, str):
            img = Image.open(texture).convert("RGB")
            arr = np.asarray(img).astype(np.float32) / 255.0
            tex = torch.from_numpy(arr)

        # ---------- 2. numpy ----------
        elif isinstance(texture, np.ndarray):
            tex = torch.from_numpy(texture.astype(np.float32))

            if tex.max() > 1.0:
                tex = tex / 255.0

        # ---------- 3. tensor ----------
        elif isinstance(texture, torch.Tensor):
            tex = texture.float()

            if tex.max() > 1.0:
                tex = tex / 255.0

        else:
            raise TypeError(f"Unsupported texture type: {type(texture)}")

        # ---------- shape 处理 ----------
        if tex.ndim == 3:
            tex = self._handle_3d(tex)

        elif tex.ndim == 4:
            tex = self._handle_4d(tex)

        else:
            raise ValueError(f"Texture must be 3D or 4D, got shape {tex.shape}")

        return tex.to(self.device)

    # =====================================
    # 3D tensor: CHW or HWC
    # =====================================
    def _handle_3d(self, tex):
        # 情况1: CHW
        if tex.shape[0] == 3 and tex.shape[-1] != 3:
            tex = tex.permute(1, 2, 0)  # → HWC

        # 情况2: HWC
        elif tex.shape[-1] == 3:
            pass

        else:
            raise ValueError(
                f"Invalid 3D texture shape {tex.shape}, expected CHW or HWC"
            )

        tex = tex.unsqueeze(0)  # → (1,H,W,3)
        return tex

    # =====================================
    # 4D tensor: BCHW or BHWC
    # =====================================
    def _handle_4d(self, tex):
        # BCHW
        if tex.shape[1] == 3 and tex.shape[-1] != 3:
            tex = tex.permute(0, 2, 3, 1)

        # BHWC
        elif tex.shape[-1] == 3:
            pass

        else:
            raise ValueError(
                f"Invalid 4D texture shape {tex.shape}, expected BCHW or BHWC"
            )

        return tex

    # 动态更新
    def set_texture(self, texture):
        tex = self._to_tensor(texture)

        if isinstance(self.tex, nn.Parameter):
            with torch.no_grad():
                self.tex.copy_(tex)
        else:
            self.tex = tex

    def get_map(self):
        return self.tex


# =====================================
# Mesh Geometry（只管 OBJ + UV index）
# =====================================
class MeshGeometry:
    def __init__(
        self,
        obj_path,
        device="cuda",
        use_carla_mesh_transform: bool = False,
        mesh_yaw_fix: float = CARLA_MESH_YAW_FIX,
        mesh_scale: float = CARLA_MESH_SCALE,
    ):
        self.device = device
        self.use_carla_mesh_transform = use_carla_mesh_transform
        self.mesh_yaw_fix = mesh_yaw_fix
        self.mesh_scale = mesh_scale
        self.load_obj(obj_path)

    def load_obj(self, obj_path):
        verts, faces, aux = load_obj(
            obj_path,
            load_textures=False,
            create_texture_atlas=False,
            device=self.device,
        )

        if aux.verts_uvs is None or faces.textures_idx is None:
            raise RuntimeError("OBJ lacks UV info")

        if self.use_carla_mesh_transform:
            center = (verts.max(0).values + verts.min(0).values) / 2.0
            verts = (verts - center) * self.mesh_scale
            rfix = RotateAxisAngle(
                angle=self.mesh_yaw_fix,
                axis="Y",
                degrees=True,
            ).get_matrix()[0, :3, :3].to(self.device)
            verts = verts @ rfix

        self.verts = verts
        self.faces = faces
        self.aux = aux

    def get_uv_info(self):
        return self.faces.textures_idx, self.aux.verts_uvs


# =====================================
# Renderer Wrapper（核心）
# =====================================
class FlexibleRenderer(nn.Module):
    def __init__(
        self,
        device,
        image_size=1024,
        blur_radius=1e-6,
        faces_per_pixel=5,
        bin_size=None,
        max_faces_per_bin=None,
    ):
        super().__init__()
        self.device = device

        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            # bin_size=bin_size,
            max_faces_per_bin=max_faces_per_bin,
        )

        self.lights = PointLights(device=device, location=[[0.0, 5.0, -3.0]])
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            shader=SoftSilhouetteShader(),
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            # shader=SoftPhongShader(device=device, lights=self.lights),
            shader=SoftPhongShader(device=device),
        )

        self.geometry = None
        self.texture = None

    # ============================
    # 动态绑定 geometry / texture
    # ============================
    def set_geometry(self, geometry: MeshGeometry):
        self.geometry = geometry

    def set_texture(self, texture: TextureModel):
        self.texture = texture

    def get_light_location(self) -> torch.Tensor:
        return self.lights.location.detach().clone()

    def set_light_location(self, location) -> None:
        location = torch.as_tensor(location, dtype=torch.float32, device=self.device)
        if location.ndim == 1:
            location = location.unsqueeze(0)
        if location.shape[-1] != 3:
            raise ValueError(
                f"Light location must end with size 3, got {tuple(location.shape)}"
            )
        self.lights.location = location

    # ============================
    # 构建 mesh（运行时）
    # ============================
    def build_mesh(self):
        assert self.geometry is not None
        assert self.texture is not None

        faces_uvs, verts_uvs = self.geometry.get_uv_info()
        tex_map = self.texture.get_map()
        batch_size = tex_map.shape[0]

        textures = TexturesUV(
            maps=tex_map,
            faces_uvs=faces_uvs[None].expand(batch_size, -1, -1).contiguous(),
            verts_uvs=verts_uvs[None].expand(batch_size, -1, -1).contiguous(),
        )

        mesh = Meshes(
            verts=[self.geometry.verts] * batch_size,
            faces=[self.geometry.faces.verts_idx] * batch_size,
            textures=textures,
        )
        return mesh

    # ============================
    # Camera
    # ============================
    def get_camera(self, dist=5.0, elev=20.0, azim=150.0):
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        return FoVPerspectiveCameras(device=self.device, R=R, T=T)

    def get_camera_RT(self, R, T):
        return FoVPerspectiveCameras(device=self.device, R=R, T=T)

    def render_silhouette(self, meshes, cameras) -> torch.Tensor:
        """
        用 SoftSilhouetteShader 渲染 alpha mask。
        输出第 3 通道（index=3）即 alpha，shape (b, H, W, 1)。
        """
        sil = self.silhouette_renderer(meshes, cameras=cameras)  # (b, H, W, 4)
        return sil[..., 3:4]

    # ============================
    # Render
    # ============================
    def forward(self, cameras):
        mesh = self.build_mesh()
        images = self.renderer(mesh, cameras=cameras)
        return images[..., :]
        # return images[..., :3]

    def render_rgb_and_hard_alpha(self, cameras):
        """
        使用可见三角形命中结果构建硬 alpha（轮廓内为 1，外部为 0）。
        这样可避免 SoftPhong 输出 alpha 在物体内部出现半透明导致的漏背景。
        """
        mesh = self.build_mesh()
        images = self.renderer(mesh, cameras=cameras)  # (b, h, w, 4)
        fragments = self.renderer.rasterizer(mesh, cameras=cameras)
        alpha = (fragments.pix_to_face[..., 0] >= 0).to(images.dtype).unsqueeze(-1)
        return images[..., :3], alpha

    def get_pil_image(self, img):
        img = img[0].detach().cpu().clamp(0, 1).numpy()
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)

    def save(self, img, path):
        img = img[0].detach().cpu().clamp(0, 1).numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(path)


class TextureRenderModel(torch.nn.Module):
    def __init__(
        self,
        device,
        uv_mask: torch.Tensor,
        render_image_size: int,
        texture_image_size: int,
        background_fill_value: float,
        mesh_obj_path: str = "pytorch3d_Etron.obj",
        bg2_path: str = "/home/dk/project/FastAdv/carmodel/audi/img_optim.png",
        raster_blur_radius: float = 1e-6,
        raster_faces_per_pixel: int = 5,
        raster_bin_size: int = 0,
        raster_max_faces_per_bin: int = 200000,
        use_carla_mesh_transform: bool = False,
        mesh_yaw_fix: float = CARLA_MESH_YAW_FIX,
        mesh_scale: float = CARLA_MESH_SCALE,
    ):
        super().__init__()
        self.geo = MeshGeometry(
            mesh_obj_path,
            device=device,
            use_carla_mesh_transform=use_carla_mesh_transform,
            mesh_yaw_fix=mesh_yaw_fix,
            mesh_scale=mesh_scale,
        )
        self.tex = TextureModel(device, trainable=False)
        self.renderer = FlexibleRenderer(
            device,
            image_size=render_image_size,
            blur_radius=raster_blur_radius,
            faces_per_pixel=raster_faces_per_pixel,
            bin_size=raster_bin_size,
            max_faces_per_bin=raster_max_faces_per_bin,
        )
        self.device = device
        self.render_image_size = render_image_size  # ← 新增
        self.texture_image_size = texture_image_size
        self.renderer.set_geometry(self.geo)
        self.renderer.set_texture(self.tex)
        self.register_buffer("uv_mask", uv_mask.unsqueeze(0), persistent=False)
        self.register_buffer(
            "background_rgb",
            torch.full(
                (1, 3, texture_image_size, texture_image_size),
                background_fill_value,
                device=device,
            ),
            persistent=False,
        )
        bg2 = np.array(
            Image.open(bg2_path).convert("RGB").resize(
                (texture_image_size, texture_image_size)
            )
        )
        bg2 = (
            torch.from_numpy(bg2.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        # print("Loaded background image for optimization:", bg2.shape, bg2.dtype)
        self.register_buffer(
            "background_rgb2",
            bg2,
            persistent=False,
        )

    # ──────────────────────────────────────────────
    # 新增：将各种背景格式统一为 (b, h, w, 3) float32
    # ──────────────────────────────────────────────
    def _prepare_background(
        self,
        background,
        batch_size: int,
    ) -> torch.Tensor | None:
        """
        将背景图像归一化为 (b, H, W, 3) float32 张量，值域 [0, 1]。

        支持输入：
          - None                 → 返回 None（后续使用默认纯色背景）
          - PIL.Image.Image      → 单张图，自动扩展到 batch
          - torch.Tensor (b,c,h,w) 或 (1,c,h,w)  → 自动广播到 batch
        """
        H = W = self.render_image_size

        if background is None:
            return None

        # ── PIL ──────────────────────────────────────────────────────────
        if isinstance(background, Image.Image):
            bg = TF.to_tensor(background.convert("RGB"))  # (3, h, w)
            bg = bg.unsqueeze(0)  # (1, 3, h, w)

        # ── Tensor (b/1, c, h, w) ────────────────────────────────────────
        elif isinstance(background, torch.Tensor):
            if background.ndim == 3:  # (c,h,w) → (1,c,h,w)
                background = background.unsqueeze(0)
            if background.ndim != 4:
                raise ValueError(
                    f"background tensor must be 3-D or 4-D, got {background.ndim}-D"
                )
            bg = background.float()
            # 值域自动归一化：uint8 风格 [0,255] → [0,1]
            if bg.max() > 1.0 + 1e-3:
                bg = bg / 255.0
        else:
            raise TypeError(
                f"background must be None, PIL.Image, or torch.Tensor, got {type(background)}"
            )

        # ── 统一尺寸 & 设备 ────────────────────────────────────────────────
        bg = bg.to(self.device)
        if bg.shape[-2:] != (H, W):
            bg = F.interpolate(bg, size=(H, W), mode="bilinear", align_corners=False)
        bg = bg.clamp(0.0, 1.0)

        # ── 广播到 batch ────────────────────────────────────────────────
        if bg.shape[0] == 1 and batch_size > 1:
            bg = bg.expand(batch_size, -1, -1, -1)
        elif bg.shape[0] != batch_size:
            raise ValueError(
                f"background batch size {bg.shape[0]} != input batch size {batch_size}"
            )

        # (b, 3, h, w) → (b, h, w, 3)，与 PyTorch3D BHWC 输出对齐
        return bg.permute(0, 2, 3, 1).contiguous()

    # ──────────────────────────────────────────────
    def prepare_uv_texture(self, diffusion_image: torch.Tensor) -> torch.Tensor:
        texture = F.interpolate(
            diffusion_image,
            size=(self.texture_image_size, self.texture_image_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)
        # return texture * self.uv_mask 
        return texture * self.uv_mask + self.background_rgb2 * (1.0 - self.uv_mask)

    def get_camera(self, dist=5.0, elev=20.0, azim=150.0):
        return self.renderer.get_camera(dist=dist, elev=elev, azim=azim)

    def set_texture(self, tex):
        self.tex.set_texture(tex)

    def get_light_location(self) -> torch.Tensor:
        return self.renderer.get_light_location()

    def set_light_location(self, location) -> None:
        self.renderer.set_light_location(location)

    def get_uv(self):
        return self.tex.get_map()

    def forward(
        self,
        cam,
        diffusion_image: torch.Tensor,
        background=None,  # ← 新增：PIL / Tensor(b,c,h,w) / None
    ) -> torch.Tensor:
        """
        渲染并与背景合成。

        返回值：
          - 若 background 为 None：直接返回渲染结果 (b, h, w, 4)，行为与原来相同。
          - 否则：用 alpha 通道将前景合成到背景上，返回 (b, h, w, 3)。
        """
        uv = self.prepare_uv_texture(diffusion_image)
        self.set_texture(uv)

        rendered = self.renderer(cam)  # (b, h, w, 4)  BHWC, rgba ∈ [0,1]

        bg = self._prepare_background(background, batch_size=rendered.shape[0])
        if bg is None:
            return rendered[..., :3]  # 保持原有行为

        # 用硬轮廓 alpha 合成，避免 SoftPhong 的软 alpha 让背景渗入前景内部。
        rgb, alpha = self.renderer.render_rgb_and_hard_alpha(cam)

        composite = rgb * alpha + bg * (1.0 - alpha)  # (b, h, w, 3)
        return composite
