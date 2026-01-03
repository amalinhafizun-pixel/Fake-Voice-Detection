#!/usr/bin/env python3
"""
Fake Voice Detection System - Main Entry Point

A multi-modal fake voice detection system that analyzes audio files
to detect AI-generated or cloned speech.

Usage:
    # GUI mode (default)
    python main.py --PyQT
    
    # Terminal mode
    python main.py --terminal --file audio.mp3
    
    # With specific backend
    python main.py --MSilicon --terminal --file audio.mp3
    python main.py --CUDA --PyQT
    python main.py --ROCm --terminal --file audio.mp3

Backends:
    --MSilicon  Use Apple Silicon (MLX/MPS)
    --CUDA      Use NVIDIA CUDA
    --ROCm      Use AMD ROCm

Modes:
    --PyQT      Launch graphical user interface (default)
    --terminal  Use terminal/command-line interface
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fake Voice Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Backend selection (mutually exclusive)
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        '--MSilicon',
        action='store_true',
        help='Use Apple Silicon backend (MLX/MPS)'
    )
    backend_group.add_argument(
        '--CUDA',
        action='store_true',
        help='Use NVIDIA CUDA backend'
    )
    backend_group.add_argument(
        '--ROCm',
        action='store_true',
        help='Use AMD ROCm backend'
    )
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--PyQT',
        action='store_true',
        help='Launch PyQt6 graphical interface (default)'
    )
    mode_group.add_argument(
        '--terminal',
        action='store_true',
        help='Use terminal/command-line interface'
    )
    
    # File input (for terminal mode)
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Audio file to analyze (required for non-interactive terminal mode)'
    )
    
    # Output options
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON (terminal mode only)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for results'
    )
    
    # Misc options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Fake Voice Detection System v1.0.0'
    )
    
    return parser.parse_args()


def get_backend(args: argparse.Namespace) -> Optional[str]:
    """Determine the backend from arguments."""
    if args.MSilicon:
        return 'MSilicon'
    elif args.CUDA:
        return 'CUDA'
    elif args.ROCm:
        return 'ROCm'
    return None


def run_gui(backend: Optional[str] = None) -> None:
    """Launch the PyQt6 GUI."""
    try:
        from ui.main_window import run_gui as launch_gui
        launch_gui(backend)
    except ImportError as e:
        logger.error(f"Failed to import GUI module: {e}")
        logger.error("Make sure PyQt6 is installed: pip install PyQt6")
        sys.exit(1)


def run_terminal(
    file_path: Optional[str] = None,
    backend: Optional[str] = None,
    json_output: bool = False,
    output_path: Optional[str] = None
) -> None:
    """Run the terminal interface."""
    try:
        from terminal.cli import run_cli
        
        if file_path:
            # Analyze specific file
            result = run_cli(
                file_path=file_path,
                backend=backend,
                json_output=json_output,
                interactive=False
            )
            
            # Save to output file if specified
            if output_path and result:
                import json
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Results saved to: {output_path}")
        else:
            # Interactive mode
            run_cli(
                backend=backend,
                json_output=json_output,
                interactive=True
            )
            
    except ImportError as e:
        logger.error(f"Failed to import terminal module: {e}")
        sys.exit(1)


def initialize_backend(backend: Optional[str]) -> None:
    """Initialize the compute backend."""
    try:
        from backend.device_manager import initialize_backend as init_backend
        
        selected = init_backend(backend)
        logger.info(f"Backend initialized: {selected.value}")
        
    except ImportError:
        logger.warning("Backend module not found, using defaults")
    except Exception as e:
        logger.warning(f"Backend initialization failed: {e}")


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get backend
    backend = get_backend(args)
    
    # Initialize backend
    initialize_backend(backend)
    
    # Determine mode
    if args.terminal:
        # Terminal mode
        run_terminal(
            file_path=args.file,
            backend=backend,
            json_output=args.json,
            output_path=args.output
        )
    else:
        # GUI mode (default)
        run_gui(backend)


if __name__ == '__main__':
    main()

