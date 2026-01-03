#!/usr/bin/env python3
"""
Train Anomaly Detection Models on Genuine Speech Data.

This script trains Isolation Forest, One-Class SVM, and LOF models
on genuine (non-synthetic) speech samples to improve anomaly detection.

Usage:
    python train_anomaly_models.py <genuine_audio_directory>
    python train_anomaly_models.py /path/to/genuine/audio --output models/anomaly
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def train_models(genuine_audio_dir: str, output_dir: str = None) -> bool:
    """
    Train anomaly detection models on genuine audio.
    
    Args:
        genuine_audio_dir: Directory containing genuine audio files
        output_dir: Output directory for models (default: models/anomaly)
        
    Returns:
        True if training successful
    """
    audio_dir = Path(genuine_audio_dir)
    output_path = Path(output_dir) if output_dir else Path(__file__).parent / "models" / "anomaly"
    
    if not audio_dir.exists():
        logger.error(f"Directory not found: {audio_dir}")
        return False
    
    # Import here to avoid errors if dependencies missing
    try:
        from audio.processor import AudioProcessor
        from detection.signal_features import SignalFeatureExtractor
        from detection.anomaly_detection import (
            AnomalyDetector,
            IsolationForestDetector,
            OneClassSVMDetector,
            LocalOutlierFactorDetector
        )
        from backend.model_loader import ModelLoader
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    
    processor = AudioProcessor()
    extractor = SignalFeatureExtractor()
    model_loader = ModelLoader(models_dir=str(output_path.parent))
    
    # Collect all audio files
    audio_files = []
    supported_exts = ['.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg']
    
    for ext in supported_exts:
        audio_files.extend(audio_dir.glob(f'*{ext}'))
        audio_files.extend(audio_dir.glob(f'**/*{ext}'))  # Recursive
    
    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        logger.info(f"Supported formats: {', '.join(supported_exts)}")
        return False
    
    logger.info(f"Found {len(audio_files)} audio files")
    logger.info("Extracting features from genuine speech samples...")
    
    # Extract features
    features_list = []
    failed = 0
    
    for i, file_path in enumerate(audio_files, 1):
        try:
            logger.info(f"  [{i}/{len(audio_files)}] Processing: {file_path.name}")
            audio = processor.load(file_path)
            features = extractor.extract(audio.waveform, audio.sample_rate)
            feature_vector = features.to_vector()
            features_list.append(feature_vector)
        except Exception as e:
            logger.warning(f"    Failed to process {file_path.name}: {e}")
            failed += 1
            continue
    
    if not features_list:
        logger.error("No features extracted. Check audio files and formats.")
        return False
    
    if failed > 0:
        logger.warning(f"Failed to process {failed} files")
    
    X_train = np.array(features_list)
    logger.info(f"\nTraining on {len(X_train)} genuine speech samples")
    logger.info(f"Feature vector dimension: {X_train.shape[1]}")
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Train individual models
    logger.info("\nTraining Isolation Forest...")
    iso_forest = IsolationForestDetector()
    iso_forest.fit(X_train)
    model_loader.save_model(iso_forest, 'isolation_forest', model_type='anomaly')
    logger.info("  ✓ Isolation Forest saved")
    
    logger.info("Training One-Class SVM...")
    one_class_svm = OneClassSVMDetector()
    one_class_svm.fit(X_train)
    model_loader.save_model(one_class_svm, 'one_class_svm', model_type='anomaly')
    logger.info("  ✓ One-Class SVM saved")
    
    logger.info("Training Local Outlier Factor...")
    lof = LocalOutlierFactorDetector()
    lof.fit(X_train)
    model_loader.save_model(lof, 'lof', model_type='anomaly')
    logger.info("  ✓ LOF saved")
    
    # Train ensemble
    logger.info("\nTraining ensemble detector...")
    detector = AnomalyDetector()
    detector.fit(X_train)
    model_loader.save_model(detector, 'anomaly_ensemble', model_type='anomaly')
    logger.info("  ✓ Ensemble detector saved")
    
    logger.info(f"\n✓ All models saved to: {output_path}")
    logger.info("\nYou can now use these models for improved anomaly detection!")
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train anomaly detection models on genuine speech data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on directory of genuine audio
  python train_anomaly_models.py /path/to/genuine/audio
  
  # Specify output directory
  python train_anomaly_models.py /path/to/genuine/audio --output custom/models
        """
    )
    
    parser.add_argument(
        'audio_directory',
        type=str,
        help='Directory containing genuine (non-synthetic) audio files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for models (default: models/anomaly)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Anomaly Detection Model Training")
    print("=" * 60)
    print(f"\nAudio directory: {args.audio_directory}")
    print(f"Output directory: {args.output or 'models/anomaly'}")
    print("\nThis will train models to recognize genuine speech patterns.")
    print("The more genuine samples you provide, the better the models will be.\n")
    
    success = train_models(args.audio_directory, args.output)
    
    if success:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print("\nThe trained models will be used automatically by the system.")
        print("You can now run detection with improved accuracy.")
    else:
        print("\n" + "=" * 60)
        print("Training Failed")
        print("=" * 60)
        print("\nPlease check the error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()

