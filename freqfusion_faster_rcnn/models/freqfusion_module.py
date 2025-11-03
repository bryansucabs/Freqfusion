import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=True):
    if warning and size is not None and align_corners:
        input_h, input_w = tuple(int(x) for x in input.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if output_h > input_h or output_w > input_w:
            if ((output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1)):
                warnings.warn(
                    f"When align_corners={align_corners}, the output would be more aligned if "
                    f"input size {(input_h, input_w)} is `x+1` and out size {(output_h, output_w)} is `nx+1`"
                )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def hamming2D(M, N):
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    return np.outer(hamming_x, hamming_y)


class FreqFusion(nn.Module):
    def __init__(self,
                 hr_channels: int,
                 lr_channels: int,
                 scale_factor: int = 2,
                 lowpass_kernel: int = 5,
                 highpass_kernel: int = 3,
                 up_group: int = 1,
                 encoder_kernel: int = 3,
                 encoder_dilation: int = 1,
                 compressed_channels: int = 64,
                 align_corners: bool = False,
                 upsample_mode: str = "nearest",
                 comp_feat_upsample: bool = True,
                 use_high_pass: bool = True,
                 use_low_pass: bool = True,
                 hr_residual: bool = True,
                 hamming_window: bool = True):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.comp_feat_upsample = comp_feat_upsample

        self.lr_encoder = nn.Conv2d(self.compressed_channels, self.compressed_channels, 3, padding=1, groups=self.up_group)
        self.low_pass_filter = nn.Conv2d(self.compressed_channels, self.compressed_channels, lowpass_kernel,
                                         padding=lowpass_kernel // 2, groups=self.up_group)
        self.low_pass_norm = nn.BatchNorm2d(self.compressed_channels)

        self.high_pass_filter = nn.Conv2d(self.compressed_channels, self.compressed_channels, highpass_kernel,
                                          padding=highpass_kernel // 2, groups=self.up_group)
        self.high_pass_norm = nn.BatchNorm2d(self.compressed_channels)

        self.output_conv = nn.Conv2d(self.compressed_channels, hr_channels, 1)
        self.relu = nn.ReLU(inplace=True)

        if hamming_window:
            window = torch.from_numpy(hamming2D(lowpass_kernel, lowpass_kernel)).float()
            window = window.unsqueeze(0).unsqueeze(0)
            self.register_buffer("low_pass_window", window, persistent=False)
        else:
            self.low_pass_window = None

    def forward(self, hr_feat: torch.Tensor, lr_feat: torch.Tensor) -> torch.Tensor:
        hr = self.hr_channel_compressor(hr_feat)
        lr = self.lr_channel_compressor(lr_feat)

        if self.comp_feat_upsample:
            lr = resize(lr, size=hr.shape[-2:], mode=self.upsample_mode, align_corners=self.align_corners)
        else:
            lr = self.content_encoder(lr)

        lr = self.lr_encoder(lr)
        lr = self.relu(lr)

        low_freq = self.low_pass_filter(lr)
        low_freq = self.low_pass_norm(low_freq)

        if self.low_pass_window is not None:
            low_freq = low_freq * self.low_pass_window.to(low_freq.device)

        high_freq = self.high_pass_filter(hr)
        high_freq = self.high_pass_norm(high_freq)

        fused = 0
        if self.use_low_pass:
            fused = fused + low_freq
        if self.use_high_pass:
            fused = fused + high_freq
        if not self.use_high_pass and not self.use_low_pass:
            fused = hr

        fused = self.relu(fused)
        fused = self.output_conv(fused)

        if self.hr_residual:
            fused = fused + hr_feat
        return fused
