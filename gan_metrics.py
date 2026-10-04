"""
Image-quality metrics for Stage 2 (Pix2Pix) evaluation.

Expects RGB tensors in [-1, 1], shape (N, 3, H, W).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as skimage_ssim


def tensor_to_uint8_nhwc(t: torch.Tensor) -> np.ndarray:
    """(N, 3, H, W) in [-1, 1] -> uint8 (N, H, W, 3)."""
    x = t.detach().float().cpu()
    x = ((x * 0.5 + 0.5).clamp(0, 1) * 255.0).byte()
    return x.permute(0, 2, 3, 1).numpy()


def mean_ssim(real: torch.Tensor, fake: torch.Tensor) -> float:
    """Per-image SSIM (RGB, data_range=255) averaged over batch."""
    total, count = ssim_sum(real, fake)
    return float(total / max(count, 1))


def ssim_sum(real: torch.Tensor, fake: torch.Tensor) -> tuple[float, int]:
    """Return (sum of per-image SSIM, n) for streaming aggregation."""
    r = tensor_to_uint8_nhwc(real)
    f = tensor_to_uint8_nhwc(fake)
    total = 0.0
    for i in range(r.shape[0]):
        total += skimage_ssim(r[i], f[i], channel_axis=2, data_range=255)
    return total, r.shape[0]


def _get_lpips():
    try:
        import lpips
    except ImportError as e:
        raise ImportError(
            'LPIPS requires: pip install lpips'
        ) from e
    return lpips


_lpips_model = None


def mean_lpips(real: torch.Tensor, fake: torch.Tensor, device: torch.device) -> float:
    total, count = lpips_sum(real, fake, device)
    return float(total / max(count, 1))


def lpips_sum(real: torch.Tensor, fake: torch.Tensor, device: torch.device) -> tuple[float, int]:
    global _lpips_model
    lpips = _get_lpips()
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex').to(device)
        _lpips_model.eval()
    with torch.no_grad():
        d = _lpips_model(real.to(device), fake.to(device))
    n = int(d.numel())
    return float(d.sum().cpu()), n


class _InceptionFID(torch.nn.Module):
    """Minimal Inception-v3 pool3 features for FID (299×299 input)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        weights = Inception_V3_Weights.DEFAULT
        net = inception_v3(weights=weights, transform_input=False)
        net.fc = torch.nn.Identity()
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [-1, 1] -> ImageNet norm @ 299
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        return self.net(x)


_inception_fid = None


def _fid_statistics(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _fid_distance(mu1, sigma1, mu2, sigma2) -> float:
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(
        diff @ diff
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covmean)
    )


def _get_inception_fid(device: torch.device) -> _InceptionFID:
    global _inception_fid
    if _inception_fid is None:
        _inception_fid = _InceptionFID().to(device)
    return _inception_fid


@torch.no_grad()
def fid_features(images: torch.Tensor, device: torch.device, batch_size: int = 16) -> np.ndarray:
    """Inception features for FID; processes one tensor without holding the full val set."""
    net = _get_inception_fid(device)
    feats = []
    n = images.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        feats.append(net(images[start:end].to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def compute_fid(
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> float:
    r = fid_features(real, device, batch_size)
    f = fid_features(fake, device, batch_size)
    return _fid_distance(*_fid_statistics(r), *_fid_statistics(f))


@torch.no_grad()
def compute_fid_from_features(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    return _fid_distance(*_fid_statistics(real_feats), *_fid_statistics(fake_feats))
