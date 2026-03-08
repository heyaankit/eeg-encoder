"""
Training Loop for EEGEncoder

Handles:
- Training loop
- Validation
- Early stopping
- Checkpoint saving
- Metrics logging

Per-subject training for prosthetics (personalized models)
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=30, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


class Trainer:
    """Trainer for EEGEncoder model"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        label_smoothing: float = 0.1,
        weight_decay: float = 0.0001
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 300,
        early_stopping_patience: int = 30,
        checkpoint_dir: str = 'checkpoints',
        subject: str = 'A01'
    ) -> Dict:
        """
        Full training loop
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            checkpoint_dir: Directory to save checkpoints
            subject: Subject ID for personalized model
            
        Returns:
            Training history
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        best_val_acc = 0
        
        print(f"\nTraining Subject: {subject}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("="*50)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Log history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            
            # Print progress
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(
                    checkpoint_dir / f"best_{subject}.pt",
                    epoch, val_acc
                )
            
            # Early stopping
            if early_stopping(val_acc):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Save final model
        self.save_checkpoint(
            checkpoint_dir / f"final_{subject}.pt",
            epoch, val_acc
        )
        
        # Save history
        history_path = checkpoint_dir / f"history_{subject}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, accuracy: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare data loaders
    
    Args:
        X: EEG data (n_trials, n_channels, n_times)
        y: Labels (n_trials,)
        batch_size: Batch size
        val_split: Validation split ratio
        random_seed: Random seed
        
    Returns:
        train_loader, val_loader
    """
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    n_val = int(len(X) * val_split)
    
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    
    # Convert to tensors
    # Add channel dimension: (n, c, t) -> (n, 1, c, t)
    X_train = torch.FloatTensor(X[train_idx]).unsqueeze(1)
    y_train = torch.LongTensor(y[train_idx])
    X_val = torch.FloatTensor(X[val_idx]).unsqueeze(1)
    y_val = torch.LongTensor(y[val_idx])
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_subject(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    subject: str,
    epochs: int = 300,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    label_smoothing: float = 0.1,
    checkpoint_dir: str = 'checkpoints'
) -> Dict:
    """
    Train model for a single subject
    
    Args:
        model: EEGEncoder model
        X: EEG data
        y: Labels
        subject: Subject ID
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        label_smoothing: Label smoothing
        checkpoint_dir: Checkpoint directory
        
    Returns:
        Training history
    """
    # Prepare data
    train_loader, val_loader = prepare_data(X, y, batch_size=batch_size)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=learning_rate,
        label_smoothing=label_smoothing
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        subject=subject
    )
    
    return history


# Test function
if __name__ == '__main__':
    print("Testing Training Loop")
    print("="*50)
    
    # Import model
    import sys
    sys.path.append('..')
    from models.eegencoder import create_eegencoder
    
    # Create model
    model = create_eegencoder(
        n_channels=22,
        n_times=1125,
        n_classes=4,
        n_branches=5,
        hidden_channels=16
    )
    
    # Create dummy data (simulate preprocessed EEG)
    n_samples = 200
    X = np.random.randn(n_samples, 22, 1125).astype(np.float32)
    y = np.random.randint(0, 4, n_samples)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Train for 2 epochs (test)
    history = train_subject(
        model=model,
        X=X,
        y=y,
        subject='test',
        epochs=2,
        batch_size=32,
        checkpoint_dir='checkpoints'
    )
    
    print("\n✓ Training loop test passed!")
