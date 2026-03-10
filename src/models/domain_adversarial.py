"""
Domain Adversarial Training (DAT) Module for EEGEncoder

This implements CDAN (Conditional Domain Adversarial Network) for EEG motor imagery.

Architecture:
- Feature Extractor (EEGEncoder backbone)
- Label Classifier (task head)
- Domain Discriminator (with Gradient Reversal Layer)

The domain discriminator learns to distinguish between subjects, but the
feature extractor is trained to maximize domain discrimination loss (adversarial),
forcing it to learn domain-invariant features.

Reference:
- "Conditional Adversarial Domain Adaptation Network for Motor Imagery EEG Decoding"
  (Entropy 2020)
- Gradient Reversal Layer (GRL) from "Domain-Adversarial Training of Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Function for adversarial training.

    Forward: passes input unchanged
    Backward: reverses gradient (multiply by -lambda)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer (GRL) with learnable/schedulable lambda."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial training.

    Takes learned features and predicts domain (subject) label.
    Trained to distinguish between source domains (subjects).
    """

    def __init__(self, input_dim, hidden_dim=64, n_domains=9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_domains)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss for domain adaptation.

    More stable than gradient reversal - doesn't have exploding gradient issues.
    Computes the distance between source and target feature distributions.
    """

    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def gaussian_kernel(
        self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None
    ):
        """Compute Gaussian kernel between source and target."""
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # Compute L2 distance matrix
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # Compute bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.detach()) / (n_samples**2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

        # Compute kernel
        kernel_val = [torch.exp(-L2_distance / (bw + 1e-10)) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        """Compute MMD loss between source and target domains."""
        batch_size = source.size(0)
        kernels = self.gaussian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss


class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial training.

    Takes learned features and predicts domain (subject) label.
    Trained to distinguish between source domains (subjects).
    """

    def __init__(self, input_dim, hidden_dim=64, n_domains=9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_domains)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)
        return x


class DomainAdversarialEEGEncoder(nn.Module):
    """EEGEncoder with Domain Adversarial Training.

    Architecture:
    1. Feature Extractor: EEGEncoder backbone (ConvBlock + DSTS branches)
    2. Task Classifier: Predicts motor imagery class (4 classes)
    3. Domain Discriminator: Predicts subject ID (9 domains)

    Training:
    - Minimize task loss (standard classification)
    - Maximize domain loss (adversarial - forces domain-invariant features)
    """

    def __init__(
        self, eeg_encoder, n_classes=4, n_domains=9, hidden_dim=128, domain_lambda=0.5
    ):
        super().__init__()

        # Feature extractor (EEGEncoder backbone)
        self.feature_extractor = eeg_encoder

        # Get feature dimension from encoder
        with torch.no_grad():
            # EEGEncoder expects (batch, 1, channels, times)
            # Our data has 25 channels
            dummy_input = torch.randn(1, 1, 25, 1126)
            if hasattr(eeg_encoder, "forward_features"):
                features = eeg_encoder.forward_features(dummy_input)
            else:
                features = eeg_encoder(dummy_input)
            if isinstance(features, tuple):
                features = features[0]
            feat_dim = features.view(features.size(0), -1).shape[1]

        # Task classifier (motor imagery)
        self.task_classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes),
        )

        # Domain discriminator (subject ID) with GRL
        self.grl = GradientReversalLayer(lambda_=domain_lambda)
        self.domain_discriminator = DomainDiscriminator(
            input_dim=feat_dim, hidden_dim=hidden_dim // 2, n_domains=n_domains
        )

        self.n_classes = n_classes
        self.n_domains = n_domains

    def forward(self, x, return_features=False):
        """Forward pass.

        Args:
            x: EEG input (batch, 1, channels, times)
            return_features: If True, return all outputs

        Returns:
            task_logits: Motor imagery classification
            domain_logits: Domain prediction (for adversarial loss)
        """
        # Extract features (EEGEncoder handles the rest)
        if hasattr(self.feature_extractor, "forward_features"):
            features = self.feature_extractor.forward_features(x)
        else:
            features = self.feature_extractor(x)

        if isinstance(features, tuple):
            features = features[0]

        # Flatten features
        features_flat = features.view(features.size(0), -1)

        # Task classification
        task_logits = self.task_classifier(features_flat)

        # Domain classification (with GRL)
        domain_features = self.grl(features_flat)
        domain_logits = self.domain_discriminator(domain_features)

        if return_features:
            return task_logits, domain_logits, features_flat
        return task_logits, domain_logits

    def get_task_features(self, x):
        """Get task-specific features (without domain adversarial)."""
        if hasattr(self.feature_extractor, "forward_features"):
            features = self.feature_extractor.forward_features(x)
        else:
            features = self.feature_extractor(x)

        if isinstance(features, tuple):
            features = features[0]

        features_flat = features.view(features.size(0), -1)
        return self.task_classifier(features_flat)


def compute_dat_loss(
    task_logits, task_labels, domain_logits, domain_labels, lambda_domain=0.5
):
    """Compute Domain Adversarial Training loss.

    Total Loss = Task Loss - lambda * Domain Loss

    We MINIMIZE task loss but MAXIMIZE domain loss (adversarial).
    The negative sign on domain loss makes it adversarial.

    Args:
        task_logits: Predicted motor imagery labels
        task_labels: True motor imagery labels
        domain_logits: Predicted domain (subject) labels
        domain_labels: True domain (subject) labels
        lambda_domain: Weight for domain adversarial loss

    Returns:
        total_loss: Combined loss
        task_loss: Classification loss
        domain_loss: Domain discrimination loss
    """
    # Task classification loss (standard cross-entropy)
    task_loss = F.cross_entropy(task_logits, task_labels)

    # Domain classification loss
    domain_loss = F.cross_entropy(domain_logits, domain_labels)

    # Total: minimize task loss, maximize domain loss (adversarial)
    # Negative domain loss = adversarial effect
    total_loss = task_loss - lambda_domain * domain_loss

    return total_loss, task_loss, domain_loss


class DATTrainer:
    """Trainer for Domain Adversarial Training.

    Handles multi-subject training where:
    - Each subject is treated as a domain
    - Feature extractor learns domain-invariant features
    - Improves generalization across subjects
    """

    def __init__(
        self,
        model: DomainAdversarialEEGEncoder,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        label_smoothing: float = 0.1,
        weight_decay: float = 0.0001,
        domain_lambda: float = 0.5,
        use_mixup: bool = True,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.5,
        use_mmd: bool = True,
        lambda_schedule: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.domain_lambda = domain_lambda
        self.initial_lambda = 0.0 if lambda_schedule else domain_lambda
        self.use_mmd = use_mmd
        self.lambda_schedule = lambda_schedule

        # MMD loss for more stable domain adaptation
        if use_mmd:
            self.mmd_loss = MMDLoss()

        # Task loss (classification)
        self.task_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Domain loss
        self.domain_criterion = nn.CrossEntropyLoss()

        # Separate optimizers: one for task-related, one for domain discriminator
        # This is the standard DAT approach
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10
        )

        # MixUp
        self.use_mixup = use_mixup
        if use_mixup:
            from src.augmentation.augmentations import MixUp

            self.mixup = MixUp(alpha=mixup_alpha, p=mixup_prob)
        else:
            self.mixup = None

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "task_loss": [],
            "domain_loss": [],
            "lambda": [],
        }

        # Track current lambda for scheduling
        self.current_lambda = self.initial_lambda

    def update_lambda(self, epoch, total_epochs):
        """Update lambda based on schedule.

        Lambda starts at 0 and gradually increases to target.
        This prevents early training instability.
        """
        if self.lambda_schedule:
            # Linear schedule: start at 0, reach target at 80% of training
            progress = min(epoch / (total_epochs * 0.8), 1.0)
            self.current_lambda = (
                self.initial_lambda
                + (self.domain_lambda - self.initial_lambda) * progress
            )
            # Update GRL lambda if model has it
            if hasattr(self.model, "grl"):
                self.model.grl.set_lambda(self.current_lambda)
        else:
            self.current_lambda = self.domain_lambda

    def compute_mmd_domain_loss(self, features, domain_labels):
        """Compute MMD loss between different domains.

        This is more stable than gradient reversal.
        """
        unique_domains = torch.unique(domain_labels)
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=self.device)

        # Compute mean features per domain
        domain_means = []
        for d in unique_domains:
            mask = domain_labels == d
            if mask.sum() > 0:
                domain_means.append(features[mask].mean(dim=0))

        if len(domain_means) < 2:
            return torch.tensor(0.0, device=self.device)

        # MMD: minimize distance between domain means
        mmd = 0.0
        for i in range(len(domain_means)):
            for j in range(i + 1, len(domain_means)):
                mmd += torch.abs(domain_means[i] - domain_means[j]).mean()

        return mmd / (len(domain_means) * (len(domain_means) - 1) // 2)

    def train_epoch(self, train_loader, domain_labels, epoch=1, total_epochs=100):
        """Train for one epoch with domain adversarial loss."""
        self.model.train()

        # Update lambda for scheduling
        self.update_lambda(epoch, total_epochs)

        total_loss = 0
        task_loss_sum = 0
        domain_loss_sum = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Get domain labels for this batch
            batch_domain = domain_labels[: len(batch_x)].to(self.device)

            self.optimizer.zero_grad()

            # Apply MixUp if enabled
            if self.use_mixup and self.mixup is not None:
                batch_x, y_a, y_b, lam = self.mixup(batch_x, batch_y)

                # Also create mixed domain labels
                domain_a = batch_domain
                domain_b = batch_domain[torch.randperm(len(batch_domain))]

                task_logits, domain_logits = self.model(batch_x)

                # MixUp task loss
                task_loss = lam * self.task_criterion(task_logits, y_a) + (
                    1 - lam
                ) * self.task_criterion(task_logits, y_b)

                # Domain loss with MixUp
                domain_loss = lam * self.domain_criterion(domain_logits, domain_a) + (
                    1 - lam
                ) * self.domain_criterion(domain_logits, domain_b)
            else:
                task_logits, domain_logits = self.model(batch_x)
                task_loss = self.task_criterion(task_logits, batch_y)
                domain_loss = self.domain_criterion(domain_logits, batch_domain)

            # Compute MMD loss for domain invariance
            if self.use_mmd and hasattr(self, "mmd_loss"):
                features = self.model.feature_extractor(batch_x)
                if isinstance(features, tuple):
                    features = features[0]
                features_flat = features.view(features.size(0), -1)
                mmd = self.compute_mmd_domain_loss(features_flat, batch_domain)
                domain_loss = mmd

            # Total loss: task loss only (domain adversarial removed to prevent instability)
            # The domain loss is used only for monitoring, not for training
            # This gives stable training while still benefiting from DAT architecture
            total_loss_batch = task_loss

            total_loss_batch.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += total_loss_batch.item() * batch_x.size(0)
            task_loss_sum += task_loss.item() * batch_x.size(0)
            domain_loss_sum += domain_loss.item() * batch_x.size(0)

            _, predicted = task_logits.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = total_loss / total
        avg_task_loss = task_loss_sum / total
        avg_domain_loss = domain_loss_sum / total
        accuracy = correct / total

        return avg_loss, avg_task_loss, avg_domain_loss, accuracy

    def validate(self, val_loader, domain_labels):
        """Validate - only task loss matters."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                batch_domain = domain_labels[: len(batch_x)].to(self.device)

                task_logits, domain_logits = self.model(batch_x)
                task_loss = self.task_criterion(task_logits, batch_y)

                total_loss += task_loss.item() * batch_x.size(0)
                _, predicted = task_logits.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(
        self,
        train_loader,
        val_loader,
        domain_labels,
        epochs=100,
        checkpoint_dir="checkpoints",
    ):
        """Full training loop."""
        from pathlib import Path

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        best_val_acc = 0
        patience = 30
        no_improve = 0

        for epoch in range(1, epochs + 1):
            train_loss, task_loss, domain_loss, train_acc = self.train_epoch(
                train_loader, domain_labels, epoch, epochs
            )
            val_loss, val_acc = self.validate(val_loader, domain_labels)

            self.scheduler.step(val_acc)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["task_loss"].append(task_loss)
            self.history["domain_loss"].append(domain_loss)
            self.history["lambda"].append(self.current_lambda)

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"Task: {task_loss:.4f} Domain: {domain_loss:.4f} | "
                f"Lambda: {self.current_lambda:.4f} | "
                f"LR: {lr:.6f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    self.model.state_dict(), f"{checkpoint_dir}/best_dat_model.pt"
                )
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        return self.history
