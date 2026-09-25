"""Real-ESRGAN x4 super-resolution for small, blurry plate crops.

The network (RRDBNet) is the standard Real-ESRGAN "x4plus" model; the weights file
RealESRGAN_x4plus.pth (from github.com/xinntao/Real-ESRGAN, release v0.1.0) lives in
backend/app/models/. Without that file, super-resolution is simply unavailable and the
rest of the pipeline works as before.

Caveat: this is a generative model. On mildly blurry plates it restores real strokes, but on
extremely low-resolution plates it invents plausible-looking glyphs, so its output should only
be preferred when OCR is clearly more confident with it.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" / "RealESRGAN_x4plus.pth"
SCALE = 4
MAX_INPUT_SIDE = 320   # larger crops are already high-resolution (and would be slow / memory hungry)


class _ResidualDenseBlock(nn.Module):
    def __init__(self, features: int = 64, growth: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(features, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + 4 * growth, features, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.activation(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.activation(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, features: int = 64, growth: int = 32):
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(features, growth)
        self.rdb2 = _ResidualDenseBlock(features, growth)
        self.rdb3 = _ResidualDenseBlock(features, growth)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class _RRDBNet(nn.Module):
    def __init__(self, features: int = 64, blocks: int = 23, growth: int = 32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, features, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(features, growth) for _ in range(blocks)])
        self.conv_body = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_last = nn.Conv2d(features, 3, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.activation(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.activation(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.activation(self.conv_hr(feat)))


class SuperResolver:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.weights_path = Path(weights_path)
        self._net: Optional[_RRDBNet] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def available(self) -> bool:
        return self.weights_path.exists()

    def _load(self) -> _RRDBNet:
        if self._net is None:
            net = _RRDBNet()
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)["params_ema"]
            net.load_state_dict(state, strict=True)
            self._net = net.eval().to(self._device)
        return self._net

    def can_upscale(self, image: np.ndarray) -> bool:
        return self.available and max(image.shape[:2]) <= MAX_INPUT_SIDE

    @torch.no_grad()
    def upscale(self, bgr: np.ndarray) -> np.ndarray:
        """Return the BGR crop enlarged 4x."""
        net = self._load()
        x = torch.from_numpy(bgr[:, :, ::-1].astype(np.float32) / 255).permute(2, 0, 1)[None].to(self._device)
        y = net(x).clamp_(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return (y[:, :, ::-1] * 255).round().astype(np.uint8)
