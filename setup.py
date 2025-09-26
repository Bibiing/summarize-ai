#!/usr/bin/env python3
"""
Setup and installation script for the Enhanced AI Audio Summarizer.
This script helps users set up the environment and test the system.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_conda():
    """Check if conda is installed and available."""
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Conda found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Conda not found or not working properly")
            return False
    except FileNotFoundError:
        print("❌ Conda not found in PATH")
        return False

def create_environment():
    """Create the conda environment from env.yml."""
    print("\n🔧 Creating conda environment...")
    try:
        result = subprocess.run(['conda', 'env', 'create', '-f', 'env.yml'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Environment created successfully")
            return True
        else:
            print(f"❌ Environment creation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return False

def check_api_key():
    """Check if API key is configured."""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if 'GOOGLE_API_KEY=' in content and len(content.split('GOOGLE_API_KEY=')[1].split('\n')[0].strip()) > 10:
                print("✓ Google API key found in .env file")
                return True
    
    print("⚠ Google API key not found or not configured")
    print("Please create a .env file with your Google Gemini API key:")
    print("GOOGLE_API_KEY=your_api_key_here")
    return False

def test_basic_imports():
    """Test if we can import the main modules."""
    print("\n🧪 Testing basic imports...")
    try:
        import soundfile
        import librosa
        import noisereduce
        import scipy
        print("✓ All audio processing libraries available")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_sample_structure():
    """Create the basic directory structure."""
    print("\n📁 Creating directory structure...")
    dirs = [
        "data/audio",
        "data/video", 
        "data/audio/converted",
        "data/audio/enhanced",
        "data/audio/test_samples"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure created")

def print_usage_examples():
    """Print usage examples."""
    print("\n" + "="*60)
    print("SETUP COMPLETE! 🎉")
    print("="*60)
    print("\n📖 USAGE EXAMPLES:")
    print("\n1. Basic usage:")
    print("   conda activate summarize-ai")
    print("   python apps.py --file data/audio/your_file.mp3")
    
    print("\n2. With noise reduction (recommended for poor quality audio):")
    print("   python apps.py --file data/audio/noisy_file.wav --denoise")
    
    print("\n3. Aggressive noise reduction (for very poor quality):")
    print("   python apps.py --file data/video/meeting.mp4 --aggressive-denoise")
    
    print("\n4. Force specific language:")
    print("   python apps.py --file data/audio/indonesian.mp3 --denoise --language id")
    
    print("\n5. Test the enhanced features:")
    print("   python test_enhanced_features.py")
    
    print("\n🎵 SUPPORTED FORMATS:")
    print("   Audio: MP3, WAV, M4A, FLAC, OGG")
    print("   Video: MP4 (extracts audio automatically)")
    
    print("\n🌍 SUPPORTED LANGUAGES:")
    print("   99+ languages including EN, ID, ZH, ES, FR, DE, JA, KO, RU...")
    print("   Use --help to see the complete list")
    
    print("\n💡 TIPS:")
    print("   - Use --denoise for recordings with background noise")
    print("   - Use --aggressive-denoise for very noisy audio")
    print("   - Let the system auto-detect language for best results")
    print("   - Check the enhanced/ folder to hear the improvements")

def main():
    """Main setup function."""
    print("AI AUDIO SUMMARIZER - ENHANCED SETUP")
    print("="*50)
    print("Setting up enhanced audio processing with noise reduction...")
    
    # Check prerequisites
    if not check_conda():
        print("\n❌ Setup failed: Conda is required")
        print("Please install Miniconda or Anaconda first:")
        print("https://docs.conda.io/en/latest/miniconda.html")
        return False
    
    # Check if environment already exists
    try:
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        if 'summarize-ai' in result.stdout:
            print("✓ Environment 'summarize-ai' already exists")
            print("To update: conda env update -f env.yml")
        else:
            if not create_environment():
                return False
    except Exception as e:
        print(f"❌ Error checking environment: {e}")
        return False
    
    # Create directory structure
    create_sample_structure()
    
    # Check API key
    check_api_key()
    
    # Print final instructions
    print_usage_examples()
    
    print(f"\n📂 Project directory: {Path.cwd()}")
    print("\n🚀 Ready to process audio with advanced noise reduction!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)