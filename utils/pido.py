"""Differentiable, distance-conditioned imaging for rendered vehicles.

The renderer is responsible for geometry and perspective. This module only
models the observation process between an ideal RGBA render and a CARLA
background: atmospheric transport, an approximate optical PSF, pixel-area
integration, and alpha compositing.

Repository images and renderer outputs are display-referred sRGB tensors in the
floating-point [0, 1] range. PIDO converts them to linear RGB for the physical
operations and converts the composited image back to sRGB on output.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PIDOOutput(NamedTuple):
    """Optional diagnostic output returned by :class:`PIDO`.

    ``image`` is display-referred sRGB. ``foreground_premultiplied`` remains
    linear RGB so it can be inspected without applying a transfer function to
    premultiplied values.
    """

    image: Tensor
    foreground_premultiplied: Tensor
    alpha: Tensor
    transmission: Tensor
    coc_diameter_px: Tensor
    psf_sigma_px: Tensor


def _as_batch_parameter(
    value: float | Tensor,
    batch_size: int,
    reference: Tensor,
) -> Tensor:
    value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if value.ndim == 0:
        return value.expand(batch_size)
    return value


def srgb_to_linear(image: Tensor) -> Tensor:
    """Decode a display-referred sRGB tensor without breaking gradients."""

    return torch.where(
        image <= 0.04045,
        image / 12.92,
        ((image.clamp_min(0.0) + 0.055) / 1.055).pow(2.4),
    )


def linear_to_srgb(image: Tensor) -> Tensor:
    """Encode a linear-RGB tensor with the standard sRGB transfer function."""

    safe_image = image.clamp_min(torch.finfo(image.dtype).eps)
    return torch.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * safe_image.pow(1.0 / 2.4) - 0.055,
    )


class AtmosphericTransport(nn.Module):
    """Koschmieder atmospheric transport in linear RGB.

    ``visibility_km=None`` disables the atmospheric term. ``airlight`` can be
    a scalar or a linear-RGB tuple, both in the [0, 1] range.
    """

    def __init__(
        self,
        visibility_km: float | None = 10.0,
        airlight: float | tuple[float, float, float] = 1.0,
        randomize_visibility: bool = False,
        visibility_km_range: tuple[float, float] = (1.0, 20.0),
    ) -> None:
        super().__init__()
        airlight_tensor = torch.as_tensor(airlight, dtype=torch.float32)
        if airlight_tensor.ndim == 0:
            airlight_tensor = airlight_tensor.repeat(3)

        self.visibility_km = visibility_km
        self.randomize_visibility = bool(randomize_visibility)
        self.visibility_km_range = tuple(float(v) for v in visibility_km_range)
        self.register_buffer("airlight", airlight_tensor.view(1, 3, 1, 1))

    def _sample_visibility(self, batch_size: int, reference: Tensor) -> Tensor:
        low, high = self.visibility_km_range
        if low <= 0 or high < low:
            raise ValueError("visibility_km_range must satisfy 0 < low <= high")
        return torch.empty(batch_size, device=reference.device, dtype=reference.dtype).uniform_(
            low, high
        )

    def forward(
        self,
        rgb: Tensor,
        distance_m: float | Tensor,
        depth_m: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = rgb.shape[0]

        if depth_m is None:
            distance = _as_batch_parameter(distance_m, batch_size, rgb)
            path_length = distance.view(batch_size, 1, 1, 1)
        else:
            path_length = depth_m.clamp_min(0.0)

        if self.visibility_km is None:
            transmission = torch.ones_like(path_length)
        else:
            if self.randomize_visibility:
                visibility = self._sample_visibility(batch_size, rgb).view(
                    batch_size, 1, 1, 1
                )
            else:
                visibility = torch.as_tensor(
                    self.visibility_km, dtype=rgb.dtype, device=rgb.device
                )
            beta = 3.912 / (visibility * 1000.0)
            transmission = torch.exp(-beta * path_length)

        observed = rgb * transmission + self.airlight.to(rgb) * (1.0 - transmission)
        return observed, transmission


class GaussianOpticalPSF(nn.Module):
    """Gaussian approximation of diffraction, lens blur, and defocus.

    The PSF width is calculated in final sensor pixels. When the foreground is
    supersampled, the kernel is scaled to its high-resolution grid before pixel
    integration.
    """

    def __init__(
        self,
        focal_length_mm: float = 18.0,
        pixel_pitch_um: float = 35.15625,
        f_number: float = 1.4,
        focus_dist_m: float | None = 10.0,
        wavelength_nm: float = 550.0,
        lens_sigma_px: float = 0.35,
        truncate_sigma: float = 3.0,
        max_radius_px: int = 24,
        randomize: bool = False,
        random_distance_range_m: tuple[float, float] = (8.0, 15.0),
        random_sigma_scale_range: tuple[float, float] = (0.75, 1.5),
    ) -> None:
        super().__init__()
        focal_length_m = focal_length_mm / 1000.0
        self.focal_length_m = focal_length_m
        self.pixel_pitch_m = pixel_pitch_um / 1e6
        self.f_number = f_number
        self.focus_dist_m = focus_dist_m
        self.wavelength_m = wavelength_nm / 1e9
        self.lens_sigma_px = lens_sigma_px
        self.truncate_sigma = truncate_sigma
        self.max_radius_px = max_radius_px
        self.randomize = bool(randomize)
        self.random_distance_range_m = tuple(float(v) for v in random_distance_range_m)
        self.random_sigma_scale_range = tuple(float(v) for v in random_sigma_scale_range)

    def psf_parameters(
        self,
        distance_m: float | Tensor,
        batch_size: int,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        distance = _as_batch_parameter(distance_m, batch_size, reference)
        if self.focus_dist_m is None:
            coc_diameter_px = torch.zeros_like(distance)
        else:
            focus = self.focus_dist_m
            coc_m = (
                torch.abs(distance - focus)
                / distance
                * self.focal_length_m**2
                / (self.f_number * (focus - self.focal_length_m))
            )
            coc_diameter_px = coc_m / self.pixel_pitch_m

        # Airy FWHM is approximately 1.028 * wavelength * f-number. Matching
        # that FWHM to a Gaussian gives sigma ~= 0.437 * wavelength * N.
        diffraction_sigma_px = (
            0.437 * self.wavelength_m * self.f_number / self.pixel_pitch_m
        )
        # A uniform defocus disk of diameter c has per-axis sigma c / 4.
        defocus_sigma_px = coc_diameter_px / 4.0
        sigma_px = torch.sqrt(
            defocus_sigma_px.square()
            + diffraction_sigma_px**2
            + self.lens_sigma_px**2
        )
        return coc_diameter_px, sigma_px

    def _sample_random_psf_parameters(
        self, batch_size: int, reference: Tensor
    ) -> tuple[Tensor, Tensor]:
        distance_low, distance_high = self.random_distance_range_m
        scale_low, scale_high = self.random_sigma_scale_range
        if distance_low <= 0 or distance_high < distance_low:
            raise ValueError("random_distance_range_m must satisfy 0 < low <= high")
        if scale_low <= 0 or scale_high < scale_low:
            raise ValueError("random_sigma_scale_range must satisfy 0 < low <= high")
        sampled_distance = torch.empty(
            batch_size, device=reference.device, dtype=reference.dtype
        ).uniform_(distance_low, distance_high)
        coc_px, sigma_px = self.psf_parameters(
            sampled_distance, batch_size, reference
        )
        sigma_scale = torch.empty_like(sigma_px).uniform_(scale_low, scale_high)
        return coc_px, sigma_px * sigma_scale

    def forward(
        self,
        image: Tensor,
        distance_m: float | Tensor,
        output_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, _, input_h, input_w = image.shape
        if self.randomize:
            coc_px, sigma_px = self._sample_random_psf_parameters(batch_size, image)
        else:
            coc_px, sigma_px = self.psf_parameters(distance_m, batch_size, image)
        scale_y = input_h / output_size[0]
        scale_x = input_w / output_size[1]

        blurred_samples = []
        for index in range(batch_size):
            sigma_y = sigma_px[index] * scale_y
            sigma_x = sigma_px[index] * scale_x
            max_radius_y = math.ceil(self.max_radius_px * scale_y)
            max_radius_x = math.ceil(self.max_radius_px * scale_x)
            blurred_samples.append(
                self._blur_sample(
                    image[index : index + 1],
                    sigma_y,
                    sigma_x,
                    max_radius_y,
                    max_radius_x,
                )
            )
        return torch.cat(blurred_samples, dim=0), coc_px, sigma_px

    def _blur_sample(
        self,
        image: Tensor,
        sigma_y: Tensor,
        sigma_x: Tensor,
        max_radius_y: int,
        max_radius_x: int,
    ) -> Tensor:
        _, channels, height, width = image.shape
        if height == 1 or width == 1:
            return image

        radius_y = min(
            max(1, math.ceil(self.truncate_sigma * float(sigma_y.detach()))),
            max_radius_y,
            height - 1,
        )
        radius_x = min(
            max(1, math.ceil(self.truncate_sigma * float(sigma_x.detach()))),
            max_radius_x,
            width - 1,
        )

        y = torch.arange(
            -radius_y, radius_y + 1, dtype=image.dtype, device=image.device
        )
        x = torch.arange(
            -radius_x, radius_x + 1, dtype=image.dtype, device=image.device
        )
        sigma_y = sigma_y.clamp_min(torch.finfo(image.dtype).eps)
        sigma_x = sigma_x.clamp_min(torch.finfo(image.dtype).eps)
        kernel = torch.exp(
            -0.5 * ((y[:, None] / sigma_y).square() + (x[None, :] / sigma_x).square())
        )
        kernel = kernel / kernel.sum()
        weight = kernel.view(1, 1, *kernel.shape).expand(channels, 1, -1, -1)

        padded = F.pad(
            image,
            (radius_x, radius_x, radius_y, radius_y),
            mode="replicate",
        )
        return F.conv2d(padded, weight, groups=channels)


class PixelAreaIntegrator(nn.Module):
    """Integrate a supersampled render into final sensor pixels."""

    def forward(self, image: Tensor, output_size: tuple[int, int]) -> Tensor:
        if image.shape[-2:] == output_size:
            return image
        return F.interpolate(image, size=output_size, mode="area")


class PIDO(nn.Module):
    """Physics-informed distance observation module for nvdiffrast output.

    Inputs and the returned image use display-referred sRGB. Inputs use NCHW
    layout by default; set ``data_format='NHWC'`` to consume nvdiffrast tensors
    directly. The foreground may be supersampled, while the CARLA background
    determines the final output resolution.

    The optical defaults match this repository's CARLA captures: a 1024-pixel
    wide image with a 90-degree horizontal FOV. A 36 mm virtual sensor gives an
    equivalent focal length of 18 mm and a 35.15625 um pixel pitch. CARLA's
    default f/1.4 aperture and 10 m focal distance are used as well.
    """

    def __init__(
        self,
        focal_length_mm: float = 18.0,
        pixel_pitch_um: float = 35.15625,
        visibility_km: float | None = 10.0,
        airlight: float | tuple[float, float, float] = 1.0,
        f_number: float = 1.4,
        focus_dist_m: float | None = 8.0,
        wavelength_nm: float = 550.0,
        lens_sigma_px: float = 0.35,
        data_format: str = "NCHW",
        randomize_psf: bool = False,
        random_psf_distance_range_m: tuple[float, float] = (8.0, 15.0),
        random_psf_sigma_scale_range: tuple[float, float] = (0.75, 1.5),
        randomize_visibility: bool = False,
        random_visibility_km_range: tuple[float, float] = (1.0, 20.0),
    ) -> None:
        super().__init__()
        self.channel_last = {"NCHW": False, "NHWC": True}[data_format.upper()]

        self.atmosphere = AtmosphericTransport(
            visibility_km,
            airlight,
            randomize_visibility=randomize_visibility,
            visibility_km_range=random_visibility_km_range,
        )
        self.optics = GaussianOpticalPSF(
            focal_length_mm=focal_length_mm,
            pixel_pitch_um=pixel_pitch_um,
            f_number=f_number,
            focus_dist_m=focus_dist_m,
            wavelength_nm=wavelength_nm,
            lens_sigma_px=lens_sigma_px,
            randomize=randomize_psf,
            random_distance_range_m=random_psf_distance_range_m,
            random_sigma_scale_range=random_psf_sigma_scale_range,
        )
        self.pixel_integrator = PixelAreaIntegrator()

    def forward(
        self,
        foreground_rgb: Tensor,
        alpha: Tensor,
        background: Tensor,
        distance_m: float | Tensor,
        depth_m: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | PIDOOutput:
        foreground_rgb = srgb_to_linear(self._to_nchw(foreground_rgb))
        alpha = self._to_nchw(alpha)
        background = srgb_to_linear(self._to_nchw(background))
        depth_m = None if depth_m is None else self._to_nchw(depth_m)

        output_size = background.shape[-2:]
        atmospheric_rgb, transmission = self.atmosphere(
            foreground_rgb, distance_m, depth_m
        )

        alpha = alpha.clamp(0.0, 1.0)
        premultiplied = atmospheric_rgb * alpha
        joint = torch.cat((premultiplied, alpha), dim=1)
        joint, coc_px, sigma_px = self.optics(joint, distance_m, output_size)

        premultiplied = self.pixel_integrator(joint[:, :3], output_size)
        observed_alpha = self.pixel_integrator(joint[:, 3:4], output_size).clamp(
            0.0, 1.0
        )
        image = premultiplied + background * (1.0 - observed_alpha)

        image_out = self._from_nchw(linear_to_srgb(image).clamp(0.0, 1.0))
        if not return_details:
            return image_out

        return PIDOOutput(
            image=image_out,
            foreground_premultiplied=self._from_nchw(premultiplied),
            alpha=self._from_nchw(observed_alpha),
            transmission=self._from_nchw(transmission),
            coc_diameter_px=coc_px,
            psf_sigma_px=sigma_px,
        )

    def forward_rgba(
        self,
        foreground_rgba: Tensor,
        background: Tensor,
        distance_m: float | Tensor,
        depth_m: Tensor | None = None,
        return_details: bool = False,
    ) -> Tensor | PIDOOutput:
        """Observe and composite a directly supplied straight-alpha RGBA render.

        ``foreground_rgba`` is display-referred sRGB plus a non-premultiplied
        alpha channel. Its layout follows ``data_format`` just like ``forward``.
        """

        channel_dim = -1 if self.channel_last else 1
        if foreground_rgba.ndim != 4 or foreground_rgba.shape[channel_dim] != 4:
            layout = "NHWC" if self.channel_last else "NCHW"
            raise ValueError(
                f"foreground_rgba must be a 4-channel {layout} tensor; "
                f"got shape {tuple(foreground_rgba.shape)}"
            )
        foreground_rgb, alpha = torch.split(foreground_rgba, (3, 1), dim=channel_dim)
        return self.forward(
            foreground_rgb,
            alpha,
            background,
            distance_m,
            depth_m=depth_m,
            return_details=return_details,
        )

    def _to_nchw(self, tensor: Tensor) -> Tensor:
        if self.channel_last:
            return tensor.permute(0, 3, 1, 2)
        return tensor

    def _from_nchw(self, tensor: Tensor) -> Tensor:
        if self.channel_last:
            return tensor.permute(0, 2, 3, 1)
        return tensor


# More descriptive alias for use in papers and configuration files.
DistanceAwarePhysicalObservation = PIDO
