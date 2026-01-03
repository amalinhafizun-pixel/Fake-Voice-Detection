#!/usr/bin/env python3
"""
Train ResNet-18 model for Fake Voice Detection using Mel Spectrograms.

This script:
1. Downloads the dataset from Kaggle
2. Preprocesses audio files into mel spectrograms
3. Trains a ResNet-18 model
4. Saves the trained model

Uses MPS (Metal Performance Shaders) for acceleration on Apple Silicon.
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import lr_scheduler
from torchvision import models

# Handle different torchvision versions
try:
    from torchvision.models.resnet import ResNet18_Weights
except ImportError:
    # Fallback for older torchvision versions
    try:
        ResNet18_Weights = models.ResNet18_Weights
    except AttributeError:
        # Very old version - use None (will use default weights)
        ResNet18_Weights = None

import librosa as lb
import numpy as np
from tqdm import tqdm

# Configuration
DATASET_KAGGLE = "birdy654/deep-voice-deepfake-voice-recognition"
# Use local dataset if available, otherwise try Kaggle
LOCAL_DATASET_PATH = Path(__file__).parent / "AUDIO"
if LOCAL_DATASET_PATH.exists() and (LOCAL_DATASET_PATH / "REAL").exists() and (LOCAL_DATASET_PATH / "FAKE").exists():
    BASE_PATH = str(LOCAL_DATASET_PATH)
else:
    BASE_PATH = "KAGGLE/AUDIO"  # Fallback to Kaggle path
SAMPLE_TIME = 3
SAMPLE_RATE = 22050
N_MELS = 64
BATCH_SIZE = 32
NUM_EPOCHS = 30  # Increased from 20 for better generalization
LEARNING_RATE = 0.001
STEP_SIZE = 7
GAMMA = 0.1

# Model save directory
MODELS_DIR = Path(__file__).parent.parent / "models" / "resnet18"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def setup_device():
    """Setup and return the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using MPS (Metal Performance Shaders) for acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA for acceleration")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (no GPU acceleration available)")
    
    return device


def download_dataset():
    """Check for dataset (local or download from Kaggle)."""
    # Check if local dataset exists
    local_path = Path(__file__).parent / "AUDIO"
    if local_path.exists() and (local_path / "REAL").exists() and (local_path / "FAKE").exists():
        print(f"\n✅ Found local dataset at: {local_path}")
        return True
    
    print("\n📥 Local dataset not found. Checking for Kaggle dataset...")
    
    zip_path = "deep-voice-deepfake-voice-recognition.zip"
    
    # Check if already downloaded
    if os.path.exists(BASE_PATH) and os.listdir(BASE_PATH):
        print(f"✅ Dataset already exists at {BASE_PATH}")
        return True
    
    # Check if zip file exists
    if os.path.exists(zip_path):
        print(f"✅ Zip file found: {zip_path}")
    else:
        # Try to download using kaggle CLI
        try:
            print("Attempting to download using Kaggle CLI...")
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", DATASET_KAGGLE],
                check=True
            )
            print("✅ Download complete")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Kaggle CLI not found or download failed")
            print(f"Please download the dataset manually from:")
            print(f"https://www.kaggle.com/datasets/{DATASET_KAGGLE}")
            print(f"Or place the zip file '{zip_path}' in the current directory")
            return False
    
    # Extract zip file
    if os.path.exists(zip_path):
        print(f"📦 Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ Extraction complete")
        return True
    
    return False


def get_mel_spectrogram(y: np.ndarray, sr: int = 22050, n_mels: int = 64) -> tuple:
    """
    Compute a Mel spectrogram from an audio signal.
    
    Parameters:
    y (np.ndarray): Audio time series.
    sr (int): Sampling rate of `y`.
    n_mels (int): Number of Mel bands to generate.
    
    Returns:
    mel_spectrogram (np.ndarray): 2D array representing the Mel spectrogram.
    shape (tuple): Shape of the Mel spectrogram.
    """
    ms = lb.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    m = lb.power_to_db(ms, ref=np.max)
    return m, m.shape


class AudioSpectrogramDataset(Dataset):
    """Dataset for audio spectrograms."""
    
    def __init__(self, base_path, sample_time=3, sr=22050, n_mels=64):
        self.base_path = base_path
        self.labels = {'FAKE': 1, 'REAL': 0}
        self.file_paths = []
        self.file_labels = []
        self.sample_time = sample_time
        self.sr = sr
        self.n_mels = n_mels
        
        # Read the file paths and labels
        for label, value in self.labels.items():
            dir_path = os.path.join(self.base_path, label)
            if not os.path.exists(dir_path):
                continue
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.wav')]
            self.file_paths.extend(files)
            self.file_labels.extend([value] * len(files))
        
        self.samples = self._generate_samples()
        print(f"📊 Loaded {len(self.file_paths)} audio files, generating {len(self.samples)} samples")
    
    def _generate_samples(self):
        """Generate samples from audio files."""
        samples = []
        for file_path, label in zip(self.file_paths, self.file_labels):
            try:
                waveform, sample_rate = lb.load(file_path, sr=self.sr)
                sample_length = int(self.sample_time * sample_rate)
                full_samples_count = len(waveform) // sample_length
                
                for i in range(full_samples_count):
                    start = i * sample_length
                    end = start + sample_length
                    segment = waveform[start:end]
                    samples.append((segment, label))
            except Exception as e:
                print(f"⚠️  Warning: Could not load {file_path}: {e}")
                continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        segment, label = self.samples[idx]
        
        # Convert waveform to spectrogram
        spectrogram, _ = get_mel_spectrogram(segment, n_mels=self.n_mels, sr=self.sr)
        spectrogram = np.abs(spectrogram)
        spectrogram /= 80  # Normalize
        
        return torch.tensor(spectrogram, dtype=torch.float32), label


def create_model(device):
    """Create and return ResNet-18 model."""
    print("\n🏗️  Creating ResNet-18 model...")
    
    # Load pre-trained ResNet-18 and modify for 1-channel input
    if ResNet18_Weights is not None:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        # Fallback for older torchvision versions
        model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(
        1, 
        model.conv1.out_channels, 
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride, 
        padding=model.conv1.padding, 
        bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    model = model.to(device)
    
    print("✅ Model created and moved to device")
    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, step_size=7, gamma=0.1):
    """Train the model."""
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for inputs, labels in train_loader_tqdm:
            # Add channel dimension: (batch, 1, 64, time_frames)
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)
            
            # Update progress bar
            current_loss = running_loss / total_predictions
            current_acc = correct_predictions.float() / total_predictions
            current_lr = optimizer.param_groups[0]['lr']
            train_loader_tqdm.set_postfix(
                train_loss=f"{current_loss:.4f}", 
                train_acc=f"{current_acc.item():.4f}", 
                lr=f"{current_lr:.6f}"
            )
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions.float() / total_predictions
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())
        
        # Save best model
        if val_acc.item() > best_val_acc:
            best_val_acc = val_acc.item()
            best_model_state = model.state_dict().copy()
            print(f"✨ New best validation accuracy: {best_val_acc:.4f}")
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'LR: {current_lr:.6f}')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model with validation accuracy: {best_val_acc:.4f}")
    
    return train_losses, train_accuracies, val_losses, val_accuracies


def evaluate_model(model, loader, criterion, device):
    """Evaluate the model on a dataset."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)
    
    loss = running_loss / len(loader.dataset)
    acc = correct_predictions.float() / total_predictions
    return loss, acc


def save_model(model, train_accuracies, val_accuracies, device):
    """Save the trained model."""
    print("\n💾 Saving model...")
    
    # Save state dict
    model_path = MODELS_DIR / 'resnet18_spectrogram_weights.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model weights saved to: {model_path.absolute()}")
    
    # Save full model with metadata
    full_model_path = MODELS_DIR / 'resnet18_spectrogram_full.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'ResNet18_Spectrogram',
        'input_type': 'mel_spectrogram',
        'n_mels': N_MELS,
        'sample_rate': SAMPLE_RATE,
        'sample_time': SAMPLE_TIME,
        'num_classes': 2,
        'device': str(device),
        'accuracy': {
            'train': train_accuracies[-1] if train_accuracies else None,
            'val': val_accuracies[-1] if val_accuracies else None,
        }
    }, full_model_path)
    print(f"✅ Full model (with metadata) saved to: {full_model_path.absolute()}")
    
    return model_path, full_model_path


def main():
    """Main training function."""
    print("=" * 60)
    print("Fake Voice Detection - Model Training Script")
    print("=" * 60)
    
    # Setup device
    device = setup_device()
    
    # Check for dataset (local or download)
    if not download_dataset():
        print("❌ Failed to find or download dataset. Exiting.")
        return
    
    # Check if dataset exists
    if not os.path.exists(BASE_PATH):
        print(f"❌ Dataset path not found: {BASE_PATH}")
        print("Please ensure the dataset is in pretrain/AUDIO/ or download from Kaggle.")
        return
    
    # Verify dataset structure
    real_dir = os.path.join(BASE_PATH, "REAL")
    fake_dir = os.path.join(BASE_PATH, "FAKE")
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"❌ Dataset structure incorrect. Expected:")
        print(f"  {BASE_PATH}/REAL/")
        print(f"  {BASE_PATH}/FAKE/")
        return
    
    # Create dataset
    print("\n📂 Creating dataset...")
    dataset = AudioSpectrogramDataset(
        base_path=BASE_PATH,
        sample_time=SAMPLE_TIME,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )
    
    if len(dataset) == 0:
        print("❌ No samples found in dataset. Exiting.")
        return
    
    # Split dataset
    print("\n📊 Splitting dataset...")
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    model = create_model(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    # Add weight decay for regularization to reduce overfitting
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Train model
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=NUM_EPOCHS, step_size=STEP_SIZE, gamma=GAMMA
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("📊 Training Results:")
    print("=" * 60)
    print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}")
    
    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    model_path, full_model_path = save_model(model, train_accuracies, val_accuracies, device)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print(f"Model saved to: {model_path}")
    print(f"Full model with metadata: {full_model_path}")
    print("\nThe model is ready to use in the Fake Voice Detection application.")


if __name__ == "__main__":
    main()

