#!/usr/bin/env python3
"""
Model Download Script for Fake Voice Detection System.

Downloads required model weights and trains models:
- AASIST pre-trained weights
- Whisper large-v2 speech recognition model
- Anomaly detection models (optional, requires genuine audio)

Usage:
    python models/download_models.py
    python models/download_models.py --skip-aasist
    python models/download_models.py --skip-whisper
    python models/download_models.py --train-anomaly /path/to/genuine/audio
    
Note: Only Whisper large-v2 model is downloaded (best accuracy).
"""

import os
import sys
import argparse
import urllib.request
import hashlib
from pathlib import Path
from typing import Optional
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Model URLs and checksums
# Note: AASIST weights need to be downloaded manually from:
# - Official repo: https://github.com/clovaai/aasist
# - HuggingFace: Search for "aasist" models
# - Research paper repositories
AASIST_MODELS = {
    'default': {
        'url': None,  # Set this if you have a direct download URL
        'filename': 'weights.pth',
        'sha256': None,
        'description': 'AASIST default weights',
        'instructions': '''
To get AASIST weights:
1. Visit: https://github.com/clovaai/aasist
2. Check releases or README for download links
3. Or search HuggingFace: https://huggingface.co/models?search=aasist
4. Download the .pth file and place it in: models/aasist/weights.pth

The system will work without AASIST weights using fallback detection.
        '''
    }
}

WHISPER_MODELS = ['large-v2']  # Only large-v2 model


def get_models_dir() -> Path:
    """Get the models directory path."""
    return Path(__file__).parent


def download_file(url: str, destination: Path, expected_sha256: Optional[str] = None) -> bool:
    """
    Download a file with progress indication.
    
    Args:
        url: URL to download from
        destination: Local path to save file
        expected_sha256: Expected SHA256 hash (optional)
        
    Returns:
        True if download successful
    """
    print(f"Downloading: {url}")
    print(f"To: {destination}")
    
    try:
        # Create directory if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def progress_hook(count, block_size, total_size):
            percent = min(100, count * block_size * 100 // total_size)
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f'\r  [{bar}] {percent}%', end='', flush=True)
        
        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print()  # New line after progress bar
        
        # Verify checksum if provided
        if expected_sha256:
            print("Verifying checksum...", end=' ')
            sha256 = hashlib.sha256()
            with open(destination, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            
            if sha256.hexdigest() == expected_sha256:
                print("OK")
            else:
                print("FAILED")
                print(f"  Expected: {expected_sha256}")
                print(f"  Got: {sha256.hexdigest()}")
                return False
        
        print(f"✓ Downloaded successfully: {destination.name}")
        return True
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def download_aasist(models_dir: Path, force: bool = False) -> bool:
    """
    Download AASIST model weights.
    
    Args:
        models_dir: Models directory path
        force: Force re-download even if exists
        
    Returns:
        True if successful
    """
    print("\n" + "=" * 50)
    print("AASIST Model")
    print("=" * 50)
    
    aasist_dir = models_dir / "aasist"
    weights_path = aasist_dir / "weights.pth"
    
    if weights_path.exists() and not force:
        file_size = weights_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ AASIST weights already exist: {weights_path}")
        print(f"  File size: {file_size:.2f} MB")
        return True
    
    model_info = AASIST_MODELS['default']
    
    if model_info['url'] is None:
        print("\n⚠ AASIST weights URL not configured.")
        print("\n" + "=" * 60)
        print("HOW TO GET AASIST WEIGHTS")
        print("=" * 60)
        print("\nOption 1: Official Repository")
        print("  1. Visit: https://github.com/clovaai/aasist")
        print("  2. Check the README or 'Releases' section")
        print("  3. Look for pre-trained model download links")
        print("  4. Download the weights file (usually .pth or .pt)")
        print(f"  5. Place it at: {weights_path}")
        
        print("\nOption 2: HuggingFace")
        print("  1. Visit: https://huggingface.co/models?search=aasist")
        print("  2. Browse available AASIST models")
        print("  3. Download the model file")
        print(f"  4. Place it at: {weights_path}")
        
        print("\nOption 3: Research Papers")
        print("  - Check arXiv papers about AASIST")
        print("  - Look for supplementary materials")
        print("  - Check conference proceedings")
        
        print("\n" + "-" * 60)
        print("IMPORTANT: The system will work WITHOUT AASIST weights!")
        print("It will use a fallback detection method based on signal features.")
        print("AASIST improves accuracy but is not required for basic functionality.")
        print("-" * 60)
        
        print(f"\nOnce you have the weights, place them at:")
        print(f"  {weights_path}")
        print("\nFor detailed instructions, see: models/SETUP_GUIDE.md")
        
        # Create placeholder
        aasist_dir.mkdir(parents=True, exist_ok=True)
        
        return False
    
    return download_file(
        model_info['url'],
        weights_path,
        model_info['sha256']
    )


def download_whisper(model_size: str = 'large-v2') -> bool:
    """
    Download Whisper model.
    
    Args:
        model_size: Model size (tiny, base, small, medium, large)
        
    Returns:
        True if successful
    """
    print("\n" + "=" * 50)
    print(f"Whisper Model ({model_size})")
    print("=" * 50)
    
    if model_size != 'large-v2':
        print(f"✗ Only 'large-v2' model is supported.")
        print(f"  Requested: {model_size}")
        print(f"  This script only downloads Whisper large-v2 (best accuracy model).")
        return False
    
    try:
        import whisper
        print(f"Downloading Whisper large-v2 model...")
        print("(This is the largest model ~3GB, may take a while)")
        print("Note: This script only downloads large-v2 for best accuracy.")
        
        model = whisper.load_model('large-v2')
        print(f"✓ Whisper large-v2 model downloaded and cached")
        
        # Show cache location
        cache_dir = Path.home() / ".cache" / "whisper"
        print(f"  Cache location: {cache_dir}")
        print(f"  Model size: ~3GB (1550M parameters)")
        
        return True
        
    except ImportError:
        print("✗ Whisper not installed. Install with:")
        print("  pip install openai-whisper")
        return False
    except Exception as e:
        print(f"✗ Failed to download Whisper: {e}")
        return False


def train_anomaly_models(genuine_audio_dir: str, models_dir: Path) -> bool:
    """
    Train anomaly detection models on genuine audio data.
    
    Args:
        genuine_audio_dir: Directory containing genuine audio files
        models_dir: Models directory path
        
    Returns:
        True if training successful
    """
    print("\n" + "=" * 50)
    print("Anomaly Detection Models - Training")
    print("=" * 50)
    
    audio_dir = Path(genuine_audio_dir)
    
    if not audio_dir.exists():
        print(f"✗ Directory not found: {audio_dir}")
        print("  Skipping anomaly model training.")
        return False
    
    print(f"Training anomaly models on genuine audio from: {audio_dir}")
    print("This will improve anomaly detection accuracy.")
    print("\nNote: This requires genuine (non-synthetic) speech samples.")
    print("The more samples you provide, the better the models will be.")
    
    try:
        # Import the training script function
        project_root = Path(__file__).parent.parent
        train_script = project_root / "train_anomaly_models.py"
        
        if not train_script.exists():
            print(f"✗ Training script not found: {train_script}")
            print("  Make sure train_anomaly_models.py is in the project root.")
            return False
        
        # Import the train_models function
        sys.path.insert(0, str(project_root))
        from train_anomaly_models import train_models
        
        output_dir = str(models_dir / "anomaly")
        success = train_models(str(audio_dir), output_dir)
        
        if success:
            print("\n✓ Anomaly models trained successfully!")
            print(f"  Models saved to: {output_dir}")
        else:
            print("\n⚠ Anomaly model training failed or incomplete.")
            print("  Check error messages above for details.")
        
        return success
        
    except ImportError as e:
        print(f"✗ Could not import training script: {e}")
        print("  Make sure train_anomaly_models.py is in the project root.")
        print("  And all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_dummy_anomaly_models(models_dir: Path) -> None:
    """Create placeholder anomaly detection models."""
    print("\n" + "=" * 50)
    print("Anomaly Detection Models")
    print("=" * 50)
    
    anomaly_dir = models_dir / "anomaly"
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    
    print("ℹ Anomaly detection models are trained on your data.")
    print("  They will be created automatically when you provide training data.")
    print("  For now, the system will use default parameters.")
    print(f"  Model directory: {anomaly_dir}")
    print("\nTo train anomaly models, run:")
    print("  python train_anomaly_models.py <genuine_audio_directory>")
    print("\nOr use the --train-anomaly option with this script:")
    print("  python models/download_models.py --train-anomaly <audio_dir>")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download model weights for Fake Voice Detection"
    )
    
    parser.add_argument(
        '--whisper-size',
        choices=WHISPER_MODELS,
        default='large-v2',
        help='Whisper model size to download (default: large-v2, only option)'
    )
    
    parser.add_argument(
        '--skip-aasist',
        action='store_true',
        help='Skip AASIST download'
    )
    
    parser.add_argument(
        '--skip-whisper',
        action='store_true',
        help='Skip Whisper download'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if models exist'
    )
    
    parser.add_argument(
        '--train-anomaly',
        type=str,
        metavar='DIR',
        help='Train anomaly detection models on genuine audio from directory'
    )
    
    args = parser.parse_args()
    
    models_dir = get_models_dir()
    
    print("\n" + "=" * 50)
    print("Fake Voice Detection - Model Downloader")
    print("=" * 50)
    print(f"Models directory: {models_dir}")
    
    success = True
    
    # Download AASIST
    if not args.skip_aasist:
        if not download_aasist(models_dir, args.force):
            success = False
    
    # Download Whisper
    if not args.skip_whisper:
        if not download_whisper(args.whisper_size):
            success = False
    
    # Train or create anomaly model placeholders
    if args.train_anomaly:
        if not train_anomaly_models(args.train_anomaly, models_dir):
            success = False
    else:
        create_dummy_anomaly_models(models_dir)
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    if success:
        print("✓ All models downloaded successfully!")
    else:
        print("⚠ Some models were not downloaded.")
        print("  The system will use fallback methods where needed.")
    
    print("\nYou can now run the application:")
    print("  python main.py --PyQT")
    print("  python main.py --terminal --file audio.mp3")
    
    if not args.train_anomaly:
        print("\nTo train anomaly models later, run:")
        print("  python train_anomaly_models.py <genuine_audio_directory>")
        print("  Or: python models/download_models.py --train-anomaly <audio_dir>")


if __name__ == '__main__':
    main()

