"""
EEGEncoder: SOTA Model for Motor Imagery Classification

Based on paper: "EEGEncoder: Advancing BCI with Transformer-Based Motor Imagery Classification"
Reference: https://github.com/BlackCattt9/EEGEncoder
Scientific Reports (2025)

Key features from the paper:
- Dual-Stream Temporal-Spatial Block (DSTS)
- 5 parallel branches with TCN + Transformer
- MixUp data augmentation
- Label smoothing (0.1-0.2)
- Dropout 0.3
- Weight decay for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Chomp1d for causal convolution."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class ConvBlock(nn.Module):
    """Convolutional front-end (similar to EEGNet)."""

    def __init__(self, F1=16, kernLength=64, poolSize=7, D=2, in_chans=22, dropout=0.3):
        super().__init__()
        F2 = F1 * D

        self.conv1 = nn.Conv2d(1, F1, (kernLength, 1), padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(F1, F1 * D, (1, in_chans), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((8, 1))
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(F1 * D, F2, (16, 1), padding="same", bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((poolSize, 1))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        return x


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with causal convolutions."""

    def __init__(
        self, input_dim, depth, kernel_size, filters, dropout, activation="elu"
    ):
        super().__init__()
        self.depth = depth
        self.activation = getattr(F, activation)
        self.dropout = dropout
        self.blocks = nn.ModuleList()

        self.downsample = (
            nn.Conv1d(input_dim, filters, 1) if input_dim != filters else None
        )

        self.cn1 = nn.Sequential(
            nn.Conv1d(input_dim, filters, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.cn2 = nn.Sequential(
            nn.Conv1d(filters, filters, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(filters),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        for i in range(depth - 1):
            dilation_size = 2 ** (i + 1)
            padding = (kernel_size - 1) * dilation_size
            block_layers = nn.Sequential(
                nn.Conv1d(
                    filters,
                    filters,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation_size,
                ),
                Chomp1d(padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(
                    filters,
                    filters,
                    kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation_size,
                ),
                Chomp1d(padding),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks.append(block_layers)

    def forward(self, x):
        out = x.transpose(1, 2)
        out = self.cn1(out)
        out = self.cn2(out)
        res = self.downsample(out) if self.downsample is not None else out

        for i, block in enumerate(self.blocks):
            if i == 0:
                out = block(out)
                out += res
            else:
                out = block(out)
                out += self.blocks[i - 1](res)
            out = self.activation(out)

        return out.transpose(1, 2)


class TransformerBlock(nn.Module):
    """Simple Transformer block for spatial attention."""

    def __init__(self, embed_dim, num_heads=2, dropout=0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class EEGEncoder(nn.Module):
    """
    EEGEncoder: Transformer-TCN Hybrid for Motor Imagery Classification

    Architecture:
    - ConvBlock: EEGNet-like convolutional front-end
    - 5 parallel DSTS branches (each has TCN + Transformer)
    - Fusion: average or concatenation
    """

    def __init__(
        self,
        n_classes: int = 4,
        in_chans: int = 22,
        in_samples: int = 1125,
        n_windows: int = 5,
        eegn_F1: int = 16,
        eegn_D: int = 2,
        eegn_kernelSize: int = 64,
        eegn_poolSize: int = 7,
        eegn_dropout: float = 0.3,
        tcn_depth: int = 2,
        tcn_kernelSize: int = 4,
        tcn_filters: int = 32,
        tcn_dropout: float = 0.3,
        fuse: str = "average",
    ):
        super().__init__()
        self.n_windows = n_windows
        self.fuse = fuse

        F2 = eegn_F1 * eegn_D

        self.conv_block = ConvBlock(
            F1=eegn_F1,
            kernLength=eegn_kernelSize,
            poolSize=eegn_poolSize,
            D=eegn_D,
            in_chans=in_chans,
            dropout=eegn_dropout,
        )

        self.tcn_blocks = nn.ModuleList(
            [
                TCNBlock(F2, tcn_depth, tcn_kernelSize, tcn_filters, tcn_dropout)
                for _ in range(n_windows)
            ]
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(F2, num_heads=2, dropout=eegn_dropout)
                for _ in range(n_windows)
            ]
        )

        self.dense_layers = nn.ModuleList(
            [nn.Linear(tcn_filters, n_classes) for _ in range(n_windows)]
        )

        self.aa_drop = nn.Dropout(0.3)

        if fuse == "concat":
            self.final_dense = nn.Linear(n_classes * n_windows, n_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = x[:, :, :, 0].permute(0, 2, 1)

        sw_outputs = []
        for i in range(self.n_windows):
            window_slice = self.aa_drop(x)

            tcn_out = self.tcn_blocks[i](window_slice)
            tcn_out = tcn_out[:, -1, :]

            trm_out = self.transformer_blocks[i](window_slice)
            trm_out = trm_out.mean(1)

            combined = tcn_out + F.dropout(trm_out, 0.3)

            dense_output = self.dense_layers[i](combined)
            sw_outputs.append(dense_output)

        if self.fuse == "average":
            out = torch.mean(torch.stack(sw_outputs, dim=0), dim=0)
        elif self.fuse == "concat":
            out = torch.cat(sw_outputs, dim=1)
            out = self.final_dense(out)

        return out


def create_eegencoder(
    n_channels: int = 22,
    n_times: int = 1125,
    n_classes: int = 4,
    n_branches: int = 5,
    hidden_channels: int = 16,
    **kwargs,
) -> EEGEncoder:
    """Create EEGEncoder model with specified parameters."""
    return EEGEncoder(
        n_classes=n_classes,
        in_chans=n_channels,
        in_samples=n_times,
        n_windows=n_branches,
        eegn_F1=hidden_channels,
        eegn_D=2,
        tcn_filters=hidden_channels * 2,
        **kwargs,
    )


class MixUp:
    """MixUp data augmentation for EEG."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(self, data, target):
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1

        batch_size = data.size(0)
        index = torch.randperm(batch_size).to(data.device)

        mixed_data = lam * data + (1 - lam) * data[index, :]
        target_a, target_b = target, target[index]

        return mixed_data, target_a, target_b, lam

    def loss_func(self, pred, target_a, target_b, lam):
        return lam * F.cross_entropy(pred, target_a, label_smoothing=0.1) + (
            1 - lam
        ) * F.cross_entropy(pred, target_b, label_smoothing=0.1)


if __name__ == "__main__":
    model = create_eegencoder(n_channels=22, n_times=1125, n_classes=4)
    x = torch.randn(8, 1, 22, 1125)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
