"""
Terminal-based Command Line Interface for Fake Voice Detection.

Provides:
- File input via argument or interactive prompt
- Progress indicators
- Formatted text output
- JSON output option for scripting
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class CLIConfig:
    """Configuration for CLI."""
    verbose: bool = False
    json_output: bool = False
    show_details: bool = True
    color_output: bool = True


class ProgressIndicator:
    """Simple progress indicator for terminal."""
    
    SPINNER_CHARS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    def __init__(self, message: str = "Processing"):
        self.message = message
        self._idx = 0
        self._active = False
    
    def update(self, step: Optional[str] = None) -> None:
        """Update the progress indicator."""
        char = self.SPINNER_CHARS[self._idx % len(self.SPINNER_CHARS)]
        self._idx += 1
        
        msg = f"\r{char} {self.message}"
        if step:
            msg += f": {step}"
        msg += "   "  # Clear trailing chars
        
        sys.stdout.write(msg)
        sys.stdout.flush()
    
    def done(self, message: str = "Done") -> None:
        """Mark progress as complete."""
        sys.stdout.write(f"\r✓ {message}          \n")
        sys.stdout.flush()
    
    def error(self, message: str = "Error") -> None:
        """Mark progress as failed."""
        sys.stdout.write(f"\r✗ {message}          \n")
        sys.stdout.flush()


class TerminalColors:
    """ANSI color codes for terminal output."""
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    
    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Apply color to text."""
        color_code = getattr(cls, color.upper(), cls.RESET)
        return f"{color_code}{text}{cls.RESET}"
    
    @classmethod
    def risk_color(cls, risk_level: str) -> str:
        """Get color for risk level."""
        colors = {
            'LOW': cls.GREEN,
            'MEDIUM': cls.YELLOW,
            'HIGH': cls.RED
        }
        return colors.get(risk_level, cls.RESET)


class TerminalInterface:
    """
    Terminal interface for fake voice detection.
    """
    
    def __init__(self, config: Optional[CLIConfig] = None):
        """
        Initialize the terminal interface.
        
        Args:
            config: CLI configuration
        """
        self.config = config or CLIConfig()
        self._detector = None
    
    def _print(self, message: str, color: Optional[str] = None) -> None:
        """Print message with optional color."""
        if self.config.color_output and color:
            message = TerminalColors.colorize(message, color)
        print(message)
    
    def _print_header(self) -> None:
        """Print application header."""
        if self.config.json_output:
            return
        
        header = """
╔══════════════════════════════════════════════════════════════╗
║           🎤 Fake Voice Detection System 🎤                   ║
║                    v1.0.0                                     ║
╚══════════════════════════════════════════════════════════════╝
        """
        self._print(header, 'cyan')
    
    def _print_result(self, result: Dict[str, Any]) -> None:
        """Print detection result."""
        if self.config.json_output:
            print(json.dumps(result, indent=2))
            return
        
        print("\n" + "=" * 60)
        self._print("DETECTION RESULTS", 'bold')
        print("=" * 60)
        
        # Overall score
        score = result.get('overall_score', 0)
        risk = result.get('risk_level', 'UNKNOWN')
        confidence = result.get('confidence', 'UNKNOWN')
        
        risk_color = TerminalColors.risk_color(risk)
        print(f"\n  Overall Score: {score:.2%}")
        print(f"  Risk Level:    {risk_color}{risk}{TerminalColors.RESET}")
        print(f"  Confidence:    {confidence}")
        
        # Component scores
        if self.config.show_details and 'component_scores' in result:
            print("\n" + "-" * 40)
            self._print("Component Scores:", 'bold')
            for component, score in result['component_scores'].items():
                bar = self._score_bar(score)
                print(f"  {component:20} {bar} {score:.2%}")
        
        # Interpretation
        if 'interpretation' in result:
            print("\n" + "-" * 40)
            self._print("Interpretation:", 'bold')
            print(f"  {result['interpretation']}")
        
        print("\n" + "=" * 60 + "\n")
    
    def _score_bar(self, score: float, width: int = 20) -> str:
        """Create a visual score bar."""
        filled = int(score * width)
        empty = width - filled
        
        if score < 0.3:
            color = TerminalColors.GREEN
        elif score < 0.6:
            color = TerminalColors.YELLOW
        else:
            color = TerminalColors.RED
        
        bar = f"{color}{'█' * filled}{'░' * empty}{TerminalColors.RESET}"
        return f"[{bar}]"
    
    def _initialize_detector(self, backend: Optional[str] = None) -> None:
        """Initialize the detection pipeline."""
        from backend.device_manager import initialize_backend
        from detection.ensemble import EnsembleDetector
        
        # Initialize backend
        initialize_backend(backend)
        
        # Create ensemble detector
        self._detector = EnsembleDetector()
    
    def analyze_file(
        self,
        file_path: str,
        backend: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single audio file.
        
        Args:
            file_path: Path to audio file
            backend: Backend to use
            
        Returns:
            Detection result dictionary
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'error': f"File not found: {file_path}"}
        
        progress = ProgressIndicator("Analyzing audio")
        
        try:
            # Load audio
            progress.update("Loading audio file")
            from audio.processor import AudioProcessor
            processor = AudioProcessor()
            audio_data = processor.load(file_path)
            
            # Extract signal features
            progress.update("Extracting signal features")
            from detection.signal_features import SignalFeatureExtractor
            signal_extractor = SignalFeatureExtractor(sample_rate=audio_data.sample_rate)
            signal_features = signal_extractor.extract(audio_data.waveform)
            signal_score = signal_extractor.compute_anomaly_score(signal_features)
            
            # Deep learning detection
            progress.update("Running deep learning model")
            from detection.deep_learning import DeepLearningDetector
            from config import RESNET18_WEIGHTS_PATH, RESNET18_FULL_PATH
            import os
            # Prefer full model if available, otherwise use weights
            model_path = str(RESNET18_FULL_PATH) if os.path.exists(RESNET18_FULL_PATH) else str(RESNET18_WEIGHTS_PATH)
            dl_detector = DeepLearningDetector(model_path=model_path if os.path.exists(model_path) else None)
            dl_result = dl_detector.detect(audio_data.waveform, audio_data.sample_rate)
            
            # Behavioral analysis
            progress.update("Analyzing behavioral patterns")
            from detection.behavioral import BehavioralAnalyzer
            behavioral = BehavioralAnalyzer(sample_rate=audio_data.sample_rate)
            behavioral_result = behavioral.analyze(audio_data.waveform)
            
            # Linguistic analysis
            progress.update("Performing linguistic analysis")
            from detection.linguistic import LinguisticAnalyzer
            linguistic = LinguisticAnalyzer()
            linguistic_result = linguistic.analyze(audio_data.waveform, audio_data.sample_rate)
            
            # Anomaly detection
            progress.update("Running anomaly detection")
            feature_vector = signal_features.to_vector()
            from detection.anomaly_detection import AnomalyDetector
            anomaly_detector = AnomalyDetector()
            # Note: In production, you'd fit on training data
            # Pass signal_features object for better heuristic when not fitted
            anomaly_result = anomaly_detector.predict(feature_vector, signal_features=signal_features)
            
            # Ensemble scoring
            progress.update("Computing ensemble score")
            from detection.ensemble import EnsembleDetector
            ensemble = EnsembleDetector()
            
            result = ensemble.detect(
                deep_learning_score=dl_result.spoof_probability,
                behavioral_score=behavioral_result.anomaly_score,
                signal_score=signal_score,
                linguistic_score=linguistic_result.anomaly_score,
                anomaly_score=anomaly_result.anomaly_score,
                confidences={
                    'deep_learning': dl_result.confidence,
                    'behavioral': behavioral_result.confidence,
                    'linguistic': linguistic_result.confidence,
                    'anomaly': anomaly_result.confidence,
                }
            )
            
            progress.done("Analysis complete")
            
            return result.to_dict()
            
        except Exception as e:
            progress.error(str(e))
            logger.exception("Analysis failed")
            return {'error': str(e)}
    
    def run_interactive(self) -> None:
        """Run interactive mode."""
        self._print_header()
        
        while True:
            try:
                print("\nOptions:")
                print("  1. Analyze audio file")
                print("  2. Batch analyze directory")
                print("  3. Settings")
                print("  4. Exit")
                
                choice = input("\nEnter choice (1-4): ").strip()
                
                if choice == '1':
                    file_path = input("Enter audio file path: ").strip()
                    if file_path:
                        result = self.analyze_file(file_path)
                        self._print_result(result)
                
                elif choice == '2':
                    dir_path = input("Enter directory path: ").strip()
                    if dir_path:
                        self._batch_analyze(dir_path)
                
                elif choice == '3':
                    self._settings_menu()
                
                elif choice == '4':
                    print("\nGoodbye!")
                    break
                
                else:
                    print("Invalid choice. Please enter 1-4.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                break
    
    def _batch_analyze(self, directory: str) -> None:
        """Analyze all audio files in a directory."""
        dir_path = Path(directory)
        
        if not dir_path.is_dir():
            self._print(f"Not a directory: {directory}", 'red')
            return
        
        # Find audio files
        extensions = ['*.mp3', '*.wav', '*.aac', '*.m4a', '*.flac']
        audio_files = []
        for ext in extensions:
            audio_files.extend(dir_path.glob(ext))
        
        if not audio_files:
            self._print("No audio files found in directory.", 'yellow')
            return
        
        print(f"\nFound {len(audio_files)} audio files.")
        
        results = []
        for i, file_path in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {file_path.name}")
            result = self.analyze_file(str(file_path))
            result['file'] = str(file_path)
            results.append(result)
        
        # Summary
        print("\n" + "=" * 60)
        self._print("BATCH ANALYSIS SUMMARY", 'bold')
        print("=" * 60)
        
        high_risk = sum(1 for r in results if r.get('risk_level') == 'HIGH')
        medium_risk = sum(1 for r in results if r.get('risk_level') == 'MEDIUM')
        low_risk = sum(1 for r in results if r.get('risk_level') == 'LOW')
        
        print(f"\n  Total files:     {len(results)}")
        print(f"  High risk:       {TerminalColors.RED}{high_risk}{TerminalColors.RESET}")
        print(f"  Medium risk:     {TerminalColors.YELLOW}{medium_risk}{TerminalColors.RESET}")
        print(f"  Low risk:        {TerminalColors.GREEN}{low_risk}{TerminalColors.RESET}")
        
        # Export option
        export = input("\nExport results to JSON? (y/n): ").strip().lower()
        if export == 'y':
            output_path = dir_path / "analysis_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_path}")
    
    def _settings_menu(self) -> None:
        """Show settings menu."""
        print("\nSettings:")
        print(f"  1. Verbose mode: {'ON' if self.config.verbose else 'OFF'}")
        print(f"  2. Show details: {'ON' if self.config.show_details else 'OFF'}")
        print(f"  3. Color output: {'ON' if self.config.color_output else 'OFF'}")
        print("  4. Back")
        
        choice = input("\nToggle setting (1-4): ").strip()
        
        if choice == '1':
            self.config.verbose = not self.config.verbose
        elif choice == '2':
            self.config.show_details = not self.config.show_details
        elif choice == '3':
            self.config.color_output = not self.config.color_output


def run_cli(
    file_path: Optional[str] = None,
    backend: Optional[str] = None,
    json_output: bool = False,
    interactive: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Run the CLI application.
    
    Args:
        file_path: Audio file to analyze
        backend: Backend to use ('MSilicon', 'CUDA', 'ROCm')
        json_output: Output as JSON
        interactive: Run in interactive mode
        
    Returns:
        Detection result if file provided, None otherwise
    """
    config = CLIConfig(json_output=json_output)
    interface = TerminalInterface(config)
    
    if interactive or file_path is None:
        interface.run_interactive()
        return None
    else:
        result = interface.analyze_file(file_path, backend)
        interface._print_result(result)
        return result

