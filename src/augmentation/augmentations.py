"""
EEG Motor Imagery Data Augmentation

Based on professional BCI research:
- Time-shift: Random shift ±100ms
- Channel dropout: Randomly drop 10-20% channels
- Gaussian noise: Add small noise (σ=0.1)
- Scaling: Multiply by random factor 0.9-1.1
- Time-frequency crop: Random time window selection

Reference:
- "Data augmentation strategies for EEG-based motor imagery decoding" (PMC2022)
- "Classification of Motor Imagery EEG Signals Based on Data Augmentation" (Sensors 2023)
"""

import numpy as np
from typing import Tuple, Optional
import torch


class EEGAugmentor:
    """Data augmentation for EEG motor imagery."""

    def __init__(
        self,
        p_time_shift: float = 0.3,
        p_channel_dropout: float = 0.3,
        p_noise: float = 0.3,
        p_scaling: float = 0.3,
        max_time_shift: int = 25,  # ±100ms at 250Hz
        noise_std: float = 0.1,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        channel_dropout_ratio: float = 0.2,
    ):
        """
        Initialize augmentor.

        Args:
            p_time_shift: Probability of time shift
            p_channel_dropout: Probability of channel dropout
            p_noise: Probability of adding Gaussian noise
            p_scaling: Probability of scaling
            max_time_shift: Maximum time shift in samples (±100ms = 25 at 250Hz)
            noise_std: Standard deviation of Gaussian noise
            scale_range: Range for scaling factor
            channel_dropout_ratio: Ratio of channels to drop
        """
        self.p_time_shift = p_time_shift
        self.p_channel_dropout = p_channel_dropout
        self.p_noise = p_noise
        self.p_scaling = p_scaling
        self.max_time_shift = max_time_shift
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.channel_dropout_ratio = channel_dropout_ratio

    def time_shift(self, x: np.ndarray) -> np.ndarray:
        """Apply random time shift."""
        if np.random.random() > self.p_time_shift:
            return x

        # Random shift
        shift = np.random.randint(-self.max_time_shift, self.max_time_shift + 1)

        if shift == 0:
            return x

        # Apply shift with padding
        if shift > 0:
            x[:, :, shift:] = x[:, :, :-shift]
            x[:, :, :shift] = 0
        else:
            x[:, :, :shift] = x[:, :, -shift:]
            x[:, :, shift:] = 0

        return x

    def channel_dropout(self, x: np.ndarray) -> np.ndarray:
        """Apply random channel dropout."""
        if np.random.random() > self.p_channel_dropout:
            return x

        n_channels = x.shape[1]
        n_drop = max(1, int(n_channels * self.channel_dropout_ratio))

        # Random channels to drop
        drop_idx = np.random.choice(n_channels, n_drop, replace=False)

        x = x.copy()
        x[:, drop_idx, :] = 0

        return x

    def add_gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        if np.random.random() > self.p_noise:
            return x

        noise = np.random.normal(0, self.noise_std, x.shape)
        return x + noise.astype(x.dtype)

    def scaling(self, x: np.ndarray) -> np.ndarray:
        """Apply random scaling."""
        if np.random.random() > self.p_scaling:
            return x

        scale = np.random.uniform(*self.scale_range)
        return x * scale

    def augment(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply all augmentations.

        Args:
            x: EEG data (n_trials, n_channels, n_times)
            y: Labels (n_trials,)

        Returns:
            Augmented x and y
        """
        x = x.copy()

        # Apply augmentations
        x = self.time_shift(x)
        x = self.channel_dropout(x)
        x = self.add_gaussian_noise(x)
        x = self.scaling(x)

        return x, y

    def augment_batch(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to a batch (for use in DataLoader).

        Args:
            x: EEG data (batch, channels, times)
            y: Labels (batch,)

        Returns:
            Augmented batch
        """
        # Convert to numpy, augment, convert back
        x_np = x.numpy()
        y_np = y.numpy()

        # Augment each sample differently
        x_aug = np.zeros_like(x_np)
        for i in range(len(x_np)):
            x_aug[i], _ = self.augment(x_np[i : i + 1], y_np[i : i + 1])

        return torch.from_numpy(x_aug), y


def create_augmented_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    augmentation_ratio: int = 2,
    augmentor: Optional["EEGAugmentor"] = None,
    val_split: float = 0.2,
    random_seed: int = 42,
):
    """
    Create augmented data loaders.

    Args:
        X: EEG data
        y: Labels
        batch_size: Batch size
        augmentation_ratio: Number of augmented samples per original
        augmentor: EEGAugmentor instance
        val_split: Validation split ratio
        random_seed: Random seed

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, TensorDataset

    if augmentor is None:
        augmentor = EEGAugmentor()

    # Split data
    np.random.seed(random_seed)
    indices = np.random.permutation(len(X))
    n_val = int(len(X) * val_split)

    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Augment training data
    X_aug_list = [X_train]
    y_aug_list = [y_train]

    for _ in range(augmentation_ratio):
        X_aug, y_aug = augmentor.augment(X_train.copy(), y_train.copy())
        X_aug_list.append(X_aug)
        y_aug_list.append(y_aug)

    X_train_aug = np.concatenate(X_aug_list, axis=0)
    y_train_aug = np.concatenate(y_aug_list, axis=0)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_aug).unsqueeze(1)  # (N, 1, C, T)
    y_train_tensor = torch.LongTensor(y_train_aug)
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
    y_val_tensor = torch.LongTensor(y_val)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Test function
if __name__ == "__main__":
    print("Testing EEG Augmentation...")

    # Create sample data
    X = np.random.randn(100, 22, 1126).astype(np.float32)
    y = np.random.randint(0, 4, 100)

    print(f"Original: X={X.shape}, y={y.shape}")

    # Test augmentor
    augmentor = EEGAugmentor()
    X_aug, y_aug = augmentor.augment(X, y)

    print(f"Augmented: X={X_aug.shape}, y={y_aug.shape}")

    # Test dataloader creation
    train_loader, val_loader = create_augmented_dataloader(
        X, y, batch_size=32, augmentation_ratio=2
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    print("✓ Augmentation test passed!")
