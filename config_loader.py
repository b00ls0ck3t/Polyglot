"""
Configuration loader for Polyglot
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_cli_args():
    """Parse command-line arguments for config overrides"""
    parser = argparse.ArgumentParser(description='Polyglot - Real-time speech translation')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (default: config.yaml)')
    parser.add_argument('--diarization', type=str, choices=['ecapa', 'pyannote', 'none'],
                       help='Diarization method (overrides config)')
    parser.add_argument('--model', type=str, choices=['medium', 'large-v2', 'large-v3'],
                       help='Whisper model (overrides config)')
    parser.add_argument('--chunk-duration', type=int,
                       help='Audio chunk duration in seconds (overrides config)')
    
    return parser.parse_args()


def get_config() -> Dict[str, Any]:
    """
    Load config from file and apply CLI overrides
    Returns merged configuration
    """
    args = parse_cli_args()
    
    # Load base config
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.diarization:
        config['diarization']['method'] = args.diarization
    
    if args.model:
        config['transcription']['model'] = args.model
    
    if args.chunk_duration:
        config['transcription']['chunk_duration'] = args.chunk_duration
    
    return config


if __name__ == "__main__":
    # Test config loading
    config = get_config()
    print("Loaded configuration:")
    print(yaml.dump(config, default_flow_style=False))
