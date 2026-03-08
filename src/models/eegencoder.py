"""
EEGEncoder: SOTA Model for Motor Imagery Classification

Based on paper: "EEGEncoder: Advancing BCI with Transformer-Based Motor Imagery Classification"
Reference: https://arxiv.org/abs/2404.14869

Simplified working version for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSTCBlock(nn.Module):
    """Dual-Stream Temporal-Spatial (DSTS) Block
    
    Simplified version: TCN + MLP
    """
    
    def __init__(self, channels, tcn_layers=3):
        super().__init__()
        
        # TCN pathway
        self.tcn = nn.ModuleList()
        for i in range(tcn_layers):
            kernel_size = 16
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            self.tcn.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                         dilation=dilation, padding=padding),
                nn.BatchNorm1d(channels),
                nn.ELU()
            ))
        
        # MLP for processing
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 2, channels)
        )
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x: (batch, channels, time)
        
        # TCN pathway
        for layer in self.tcn:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=2)  # (batch, channels)
        
        # MLP
        x = self.mlp(x)
        
        # Dropout
        x = self.dropout(x)
        
        return x


class EEGEncoder(nn.Module):
    """
    EEGEncoder: Model for Motor Imagery Classification
    
    Simplified working version.
    
    Input: (batch, 1, 22, 1125) - 22 EEG channels, 4.5s at 250Hz
    Output: (batch, 4) - 4 classes
    """
    
    def __init__(
        self,
        n_channels=22,
        n_times=1125,
        n_classes=4,
        n_branches=5,
        hidden_channels=16
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.n_branches = n_branches
        self.hidden_channels = hidden_channels
        
        # =====================
        # Downsampling Projector
        # =====================
        # Layer 1: Temporal convolution
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=(1, 64),
            stride=(1, 16),
            padding=(0, 32)
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        # Layer 2: Channel mixing
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1)
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        # Layer 3: Spatial convolution
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(7, 1)
        )
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        
        # Pooling
        self.pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        
        # Dropout
        self.input_dropout = nn.Dropout(0.3)
        
        # =====================
        # Parallel DSTS Branches
        # =====================
        self.branches = nn.ModuleList([
            DSTCBlock(channels=hidden_channels, tcn_layers=3)
            for _ in range(n_branches)
        ])
        
        self.branch_dropout = nn.Dropout(0.3)
        
        # =====================
        # Classification Head
        # =====================
        final_dim = hidden_channels * n_branches
        
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, hidden_channels * 2),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels * 2, n_classes)
        )
    
    def forward(self, x):
        # x: (batch, 1, channels, time) = (batch, 1, 22, 1125)
        batch = x.shape[0]
        
        # Downsampling Projector
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Reshape: (batch, hidden_channels, ~17)
        x = x.view(batch, self.hidden_channels, -1)
        
        # Input dropout
        x = self.input_dropout(x)
        
        # Parallel branches
        branch_outputs = []
        for branch in self.branches:
            h = branch(x)
            branch_outputs.append(h)
        
        # Concatenate
        x = torch.cat(branch_outputs, dim=1)
        
        # Branch dropout
        x = self.branch_dropout(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


def create_eegencoder(
    n_channels=22,
    n_times=1125,
    n_classes=4,
    n_branches=5,
    hidden_channels=16
) -> EEGEncoder:
    """Create EEGEncoder model"""
    return EEGEncoder(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        n_branches=n_branches,
        hidden_channels=hidden_channels
    )


# Test function
if __name__ == '__main__':
    print("Testing EEGEncoder Model")
    print("="*50)
    
    # Create model
    model = create_eegencoder(
        n_channels=22,
        n_times=1125,
        n_classes=4,
        n_branches=5,
        hidden_channels=16
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 22, 1125)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits):\n{output}")
    
    # Get predictions
    predictions = torch.argmax(output, dim=1)
    print(f"Predictions: {predictions}")
    
    print("\n✓ EEGEncoder model test passed!")
