# Model Training Script

This directory contains scripts for training the fake voice detection model.

## Files

- `train_model.py` - Main training script (downloads dataset and trains ResNet-18 model)
- `DeepFake_Classification.ipynb` - Original Jupyter notebook (for reference)

## Quick Start

### Prerequisites

1. Install required packages:
```bash
pip install torch torchvision librosa numpy tqdm
```

2. (Optional) Install Kaggle CLI for automatic dataset download:
```bash
pip install kaggle
# Configure with your API credentials: https://www.kaggle.com/account
```

### Run Training

```bash
cd pretrain
python train_model.py
```

The script will:
1. ✅ Automatically detect and use MPS (Apple Silicon) for acceleration
2. 📥 Download the dataset from Kaggle (or use existing files)
3. 🔄 Preprocess audio files into mel spectrograms
4. 🚀 Train ResNet-18 model for 20 epochs
5. 💾 Save trained model to `../models/aasist/`

## Training Configuration

You can modify these settings in `train_model.py`:

```python
SAMPLE_TIME = 3          # Audio segment length (seconds)
SAMPLE_RATE = 22050      # Audio sample rate
N_MELS = 64              # Number of mel bands
BATCH_SIZE = 32          # Training batch size
NUM_EPOCHS = 20          # Number of training epochs
LEARNING_RATE = 0.001    # Initial learning rate
STEP_SIZE = 7            # LR scheduler step size
GAMMA = 0.1              # LR reduction factor
```

## Device Selection

The script automatically selects the best available device:
1. **MPS** (Apple Silicon) - Fastest on MacBook M4
2. **CUDA** (NVIDIA GPU) - If available
3. **CPU** - Fallback option

## Output

After training, you'll find:

- `../models/aasist/resnet18_spectrogram_weights.pth` - Model weights
- `../models/aasist/resnet18_spectrogram_full.pth` - Full model with metadata

## Manual Dataset Download

If Kaggle CLI is not available, you can manually download:

1. Visit: https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition
2. Download the dataset
3. Extract to `KAGGLE/AUDIO/` directory:
   ```
   KAGGLE/
   └── AUDIO/
       ├── FAKE/
       │   └── *.wav files
       └── REAL/
           └── *.wav files
   ```

## Training Progress

The script shows:
- Real-time training progress with tqdm
- Loss and accuracy for each epoch
- Best model tracking (saves model with highest validation accuracy)
- Final test set evaluation

## Expected Results

- **Training Accuracy**: ~99.8%
- **Validation Accuracy**: ~99.6%
- **Test Accuracy**: ~99.6%

Training time on MPS (M4 MacBook): ~20-30 minutes for 20 epochs

